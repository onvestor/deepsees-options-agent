"""Read the decision log and shape it into the four views.

**Every number on the dashboard comes from this file, and this file reads one
thing: ``decision_log.jsonl``.** No broker calls, no config reads that affect a
figure, no recomputation of a decision. If a fact is not in the log then the
dashboard cannot show it, and that constraint is the point -- a dashboard that
reached for the broker to fill a gap would be showing something the log cannot
prove, which is the opposite of an audit trail.

The consequence to keep in mind while reading the "live" view: it shows the
**last state the log recorded**, not the state of the account right now. If the
session process died an hour ago, this shows an hour-old world and says so
through ``stale_seconds`` rather than quietly looking current.

**Traces are grouped by ``trace_id`` where present.** Entry scans re-run every
few minutes, so a symbol is scanned many times in a session; grouping a causal
chain by symbol alone would splice unrelated attempts into one chain that never
happened. Older records predate the id, so there is a documented fallback --
and it is labelled as inferred, because a reconstructed chain and a recorded one
are not the same evidence.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# Kinds that mean a guardrail fired. The safety story is made of these.
GUARDRAIL_KINDS = frozenset({
    "agent_override", "cap_override", "killswitch", "skip",
})

AGENT_ORDER = ["a2_context", "a1_regime", "a4_contract", "a3_risk", "a5_exit", "a6_review"]

# The pipeline, in the order a candidate passes through it. Used to lay a trace
# out as a chain rather than a pile of timestamps.
STAGE_OF_KIND = {
    "agent_call": "agent",
    "signal_eval": "signal",
    "prefilter": "prefilter",
    "sizing": "sizing",
    "cap_override": "caps",
    "agent_override": "override",
    "order": "order",
    "skip": "skip",
    "killswitch": "killswitch",
    "session": "session",
}


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


@dataclass(frozen=True)
class Record:
    """One decision, as written. Never modified here."""

    raw: dict[str, Any]

    @property
    def seq(self) -> int:
        return int(self.raw.get("seq", 0))

    @property
    def kind(self) -> str:
        return self.raw.get("kind") or self.raw["payload"]["kind"]

    @property
    def payload(self) -> dict[str, Any]:
        return self.raw["payload"]

    @property
    def symbol(self) -> str | None:
        return self.raw.get("symbol")

    @property
    def action(self) -> str:
        return self.raw.get("action", "")

    @property
    def trace_id(self) -> str | None:
        return self.raw.get("trace_id")

    @property
    def ts_et(self) -> str:
        return self.raw.get("ts_et", "")

    @property
    def clock(self) -> str:
        return self.ts_et[11:19] if len(self.ts_et) >= 19 else ""

    @property
    def latency_ms(self) -> float | None:
        value = self.raw.get("latency_ms")
        return float(value) if value is not None else None

    @property
    def is_guardrail(self) -> bool:
        return self.kind in GUARDRAIL_KINDS

    def moment(self) -> datetime | None:
        stamp = self.raw.get("ts_utc")
        if not stamp:
            return None
        try:
            return datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        except ValueError:
            return None


@dataclass
class Log:
    """Every record from one or more session files, in sequence order."""

    records: list[Record] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, paths: Iterable[Path]) -> "Log":
        records: list[Record] = []
        sources: list[str] = []
        for path in paths:
            if not path.is_file():
                continue
            sources.append(str(path))
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    records.append(Record(json.loads(line)))
                except json.JSONDecodeError:
                    # A half-written final line is normal while a session is
                    # still appending. Skipped rather than raised: a dashboard
                    # that 500s because it caught the writer mid-flush is
                    # worse than one record late.
                    continue
        records.sort(key=lambda r: (r.raw.get("session_date", ""), r.seq))
        return cls(records=records, sources=sources)

    # -- selection ----------------------------------------------------------

    @property
    def sessions(self) -> list[str]:
        return sorted({r.raw.get("session_date", "") for r in self.records if r.raw.get("session_date")})

    def for_session(self, session: str | None) -> list[Record]:
        if not session:
            return self.records
        return [r for r in self.records if r.raw.get("session_date") == session]

    # -- view 1: timeline ---------------------------------------------------

    def timeline(self, session: str | None = None) -> list[dict[str, Any]]:
        """Every decision, chronologically, flattened for a table."""
        out = []
        for r in self.for_session(session):
            out.append({
                "seq": r.seq,
                "time": r.clock,
                "kind": r.kind,
                "stage": STAGE_OF_KIND.get(r.kind, r.kind),
                "actor": _actor(r),
                "symbol": r.symbol or "",
                "action": r.action,
                "verdict": _verdict(r),
                "latency_ms": round(r.latency_ms) if r.latency_ms is not None else None,
                "trace_id": r.trace_id,
                "guardrail": r.is_guardrail,
            })
        return out

    # -- view 2: traces -----------------------------------------------------

    def traces(self, session: str | None = None) -> list[dict[str, Any]]:
        """One causal chain per entry attempt or exit decision."""
        records = self.for_session(session)
        groups: dict[str, list[Record]] = {}
        inferred: dict[str, list[Record]] = {}

        for r in records:
            if r.trace_id:
                groups.setdefault(r.trace_id, []).append(r)
            elif r.symbol:
                # Fallback for records written before trace_id was threaded.
                # Chains are split whenever a symbol's scan restarts, so two
                # attempts on the same symbol do not merge.
                inferred.setdefault(r.symbol, []).append(r)

        for symbol, rows in inferred.items():
            for index, chunk in enumerate(_split_attempts(rows), start=1):
                groups[f"inferred-{index:03d}-{symbol}"] = chunk

        out = []
        for trace_id, rows in groups.items():
            rows.sort(key=lambda r: r.seq)
            out.append(_trace_summary(trace_id, rows))
        out.sort(key=lambda t: t["first_seq"])
        return out

    def trace(self, trace_id: str, session: str | None = None) -> dict[str, Any] | None:
        for candidate in self.traces(session):
            if candidate["trace_id"] == trace_id:
                return candidate
        return None

    # -- view 3: guardrails -------------------------------------------------

    def guardrails(self, session: str | None = None) -> dict[str, Any]:
        """Every clamp, force, cap, fail-closed skip, retry and switch verdict.

        Grouped by what the reader is actually asking. A clamp and a force are
        counted apart because they mean different things -- a clamp says the
        model returned something invalid, a force says it returned something
        legal and a rule overrode it. Collapsing them would make a well-behaved
        model on a choppy day look identical to one emitting garbage.
        """
        records = self.for_session(session)
        clamps, forces, caps, skips, switches, retries = [], [], [], [], [], []

        for r in records:
            if r.kind == "agent_override":
                row = {
                    "time": r.clock, "seq": r.seq, "agent": r.payload.get("agent"),
                    "symbol": r.symbol or "", "field": r.payload.get("field"),
                    "model_value": r.payload.get("model_value"),
                    "applied_value": r.payload.get("applied_value"),
                    "rule": r.payload.get("rule"), "detail": r.payload.get("detail", ""),
                    "trace_id": r.trace_id,
                }
                (clamps if r.payload.get("override") == "clamp" else forces).append(row)
            elif r.kind == "cap_override":
                caps.append({
                    "time": r.clock, "seq": r.seq, "symbol": r.symbol or "",
                    "cap": r.payload.get("cap_name"), "stage": r.payload.get("stage"),
                    "requested": r.payload.get("requested"),
                    "cap_value": r.payload.get("cap_value"),
                    "applied": r.payload.get("applied"), "trace_id": r.trace_id,
                })
            elif r.kind == "skip":
                skips.append({
                    "time": r.clock, "seq": r.seq, "symbol": r.symbol or "",
                    "stage": r.payload.get("stage"), "reason": r.payload.get("reason"),
                    "trace_id": r.trace_id,
                })
            elif r.kind == "killswitch":
                switches.append({
                    "time": r.clock, "seq": r.seq, "switch": r.payload.get("switch"),
                    "threshold": r.payload.get("threshold"),
                    "observed": r.payload.get("observed"),
                    "fired": bool(r.payload.get("fired")),
                    "halts": bool(r.payload.get("halts_new_entries", True)),
                })
            elif r.kind == "agent_call":
                validation = r.payload.get("validation") or {}
                if int(validation.get("attempt", 1)) > 1:
                    retries.append({
                        "time": r.clock, "seq": r.seq,
                        "agent": r.payload.get("agent"), "symbol": r.symbol or "",
                        "attempt": validation.get("attempt"),
                        "status": validation.get("status"),
                        "errors": validation.get("errors", []),
                        "trace_id": r.trace_id,
                    })

        return {
            "clamps": clamps,
            "forces": forces,
            "cap_overrides": caps,
            "skips": skips,
            "schema_retries": retries,
            "killswitch_evaluations": switches,
            "summary": {
                "clamps": len(clamps), "forces": len(forces),
                "cap_overrides": len(caps), "skips": len(skips),
                "schema_retries": len(retries),
                "killswitch_evaluations": len(switches),
                "killswitches_fired": sorted(
                    {s["switch"] for s in switches if s["fired"]}
                ),
            },
        }

    # -- view 4: status -----------------------------------------------------

    def status(self, session: str | None = None) -> dict[str, Any]:
        """Last recorded state. **Not** a live broker read -- see the module docstring."""
        records = self.for_session(session)
        sessions = [r for r in records if r.kind == "session"]
        orders = [r for r in records if r.kind == "order"]
        calls = [r for r in records if r.kind == "agent_call"]

        last_session = sessions[-1] if sessions else None
        last_record = records[-1] if records else None
        age = None
        if last_record is not None:
            moment = last_record.moment()
            if moment is not None:
                age = max(0.0, (_now() - moment).total_seconds())

        # Positions reconstructed from fills, so the view is provable from the
        # log alone. Open premium is at entry price, matching how the caps
        # measure it.
        open_positions: dict[str, dict[str, Any]] = {}
        realized = 0.0
        for r in orders:
            p = r.payload
            legs = p.get("legs") or []
            filled = float(p.get("filled_qty") or 0)
            price = p.get("filled_avg_price")
            if not legs or filled <= 0 or price is None:
                continue
            contract = legs[0]
            if p.get("intent") == "buy_to_open":
                held = open_positions.setdefault(
                    contract, {"contract": contract, "symbol": r.symbol,
                               "qty": 0, "entry": float(price), "opened": r.clock}
                )
                held["qty"] += int(filled)
                held["entry"] = float(price)
            elif p.get("intent") == "sell_to_close":
                held = open_positions.get(contract)
                if held:
                    realized += (float(price) - held["entry"]) * min(filled, held["qty"]) * 100.0
                    held["qty"] -= int(filled)
                    if held["qty"] <= 0:
                        open_positions.pop(contract, None)

        # Kill-switch headroom, from the most recent evaluation of each switch.
        latest_switch: dict[str, dict[str, Any]] = {}
        for r in records:
            if r.kind == "killswitch":
                p = r.payload
                latest_switch[p["switch"]] = {
                    "switch": p["switch"],
                    "threshold": float(p.get("threshold") or 0.0),
                    "observed": float(p.get("observed") or 0.0),
                    "headroom": float(p.get("threshold") or 0.0) - float(p.get("observed") or 0.0),
                    "fired": bool(p.get("fired")),
                    "time": r.clock,
                }

        # Agent reliability. The same input the runner's kill switch reads.
        per_agent: dict[str, dict[str, Any]] = {}
        for r in calls:
            agent = r.payload.get("agent", "?")
            entry = per_agent.setdefault(
                agent, {"agent": agent, "calls": 0, "failed": 0, "clamped": 0,
                        "latencies": []}
            )
            entry["calls"] += 1
            status = (r.payload.get("validation") or {}).get("status")
            if status in ("failed", "timeout"):
                entry["failed"] += 1
            elif status == "clamped":
                entry["clamped"] += 1
            if r.latency_ms is not None:
                entry["latencies"].append(r.latency_ms)
        agents = []
        for agent in sorted(per_agent, key=lambda a: AGENT_ORDER.index(a) if a in AGENT_ORDER else 99):
            entry = per_agent[agent]
            lat = sorted(entry.pop("latencies"))
            entry["failure_rate"] = round(entry["failed"] / entry["calls"], 4) if entry["calls"] else 0.0
            entry["median_ms"] = round(lat[len(lat) // 2]) if lat else None
            entry["p95_ms"] = round(lat[int(len(lat) * 0.95)]) if lat else None
            agents.append(entry)

        halted = any(s["fired"] for s in latest_switch.values())
        return {
            "session": (last_record.raw.get("session_date") if last_record else None),
            "records": len(records),
            "last_record_at": last_record.ts_et if last_record else None,
            "stale_seconds": round(age) if age is not None else None,
            "equity": (last_session.payload.get("equity") if last_session else None),
            "open_positions_recorded": (
                last_session.payload.get("open_positions") if last_session else None
            ),
            "positions_from_fills": list(open_positions.values()),
            "realized_pnl_from_fills": round(realized, 2),
            "orders": len(orders),
            "fills": sum(
                1 for r in orders if float(r.payload.get("filled_qty") or 0) > 0
            ),
            "killswitches": sorted(latest_switch.values(), key=lambda s: s["switch"]),
            "halted": halted,
            "agents": agents,
        }


# --- helpers ---------------------------------------------------------------


def _actor(r: Record) -> str:
    if r.kind == "agent_call":
        return r.payload.get("agent", "agent")
    if r.kind == "agent_override":
        return r.payload.get("agent", "agent")
    if r.kind == "skip":
        return f"skip:{r.payload.get('stage', '?')}"
    if r.kind == "killswitch":
        return r.payload.get("switch", "killswitch")
    return r.kind


def _verdict(r: Record) -> str:
    """A one-line human reading of what this record decided."""
    p = r.payload
    if r.kind == "agent_call":
        parsed = p.get("response_parsed")
        status = (p.get("validation") or {}).get("status")
        if not parsed:
            errors = (p.get("validation") or {}).get("errors") or []
            return f"{status}: {errors[0][:70] if errors else 'no decision'}"
        return _summarise(parsed)
    if r.kind == "skip":
        return str(p.get("reason", ""))[:110]
    if r.kind == "order":
        status = str(p.get("status") or "").split(".")[-1]
        return (f"{p.get('intent')} {p.get('qty')} @ {p.get('limit_price')} "
                f"-> {status} filled {p.get('filled_qty')} @ {p.get('filled_avg_price')}")
    if r.kind == "prefilter":
        return f"{p.get('survivors')} survivors of {p.get('total_contracts')}"
    if r.kind == "signal_eval":
        blocked = [g for g, ok in (p.get("gates") or {}).items() if not ok]
        return ("triggered " + str(p.get("direction"))) if p.get("triggered") \
            else f"no signal (blocked: {', '.join(blocked) or 'n/a'})"
    if r.kind == "sizing":
        return (f"base {p.get('base_contracts')} -> final {p.get('final_contracts')} "
                f"({p.get('premium_per_contract')}/contract)")
    if r.kind == "cap_override":
        return f"{p.get('cap_name')}: requested {p.get('requested')} -> {p.get('applied')}"
    if r.kind == "agent_override":
        return (f"{p.get('override')} {p.get('field')}: {p.get('model_value')} "
                f"-> {p.get('applied_value')} [{p.get('rule')}]")
    if r.kind == "killswitch":
        return (f"{p.get('observed')} vs {p.get('threshold')} "
                f"{'FIRED' if p.get('fired') else 'ok'}")
    if r.kind == "session":
        return f"{p.get('event')} equity={p.get('equity')} {p.get('notes') or ''}"
    return ""


def _summarise(parsed: dict) -> str:
    if "regime" in parsed:
        sp = parsed.get("signal_profile", {})
        return (f"{parsed['regime']} conf={parsed['confidence']} "
                f"dir={sp.get('allowed_direction')}")
    if "eligible" in parsed:
        return (f"eligible={parsed['eligible']} bias={parsed['directional_bias']}"
                f"/{parsed['bias_strength']} iv={parsed['iv_assessment']}")
    if "size_multiplier" in parsed:
        return f"multiplier={parsed['size_multiplier']}"
    if "structure" in parsed:
        return f"{parsed['structure']} {parsed['primary_symbol']}"
    if "action" in parsed:
        return f"{parsed['action']} stop={parsed.get('new_stop_pct')}"
    if "observations" in parsed:
        return f"{len(parsed['observations'])} observation(s)"
    return json.dumps(parsed)[:80]


def _split_attempts(rows: list[Record]) -> list[list[Record]]:
    """Split a symbol's records wherever a new scan clearly begins.

    A new attempt starts at an ``a2_context``/``a1_regime`` call or a
    ``signal_eval`` following a terminal record. Approximate by construction --
    which is why traces built this way are labelled inferred.
    """
    rows.sort(key=lambda r: r.seq)
    chunks: list[list[Record]] = []
    current: list[Record] = []
    for r in rows:
        starts = r.kind == "signal_eval" or (
            r.kind == "agent_call" and r.payload.get("agent") == "a2_context"
        )
        if starts and current:
            chunks.append(current)
            current = []
        current.append(r)
    if current:
        chunks.append(current)
    return chunks


def _trace_summary(trace_id: str, rows: list[Record]) -> dict[str, Any]:
    symbols = [r.symbol for r in rows if r.symbol]
    orders = [r for r in rows if r.kind == "order"]
    skips = [r for r in rows if r.kind == "skip"]
    filled = [r for r in orders if float(r.payload.get("filled_qty") or 0) > 0]

    if filled:
        outcome, outcome_detail = "filled", _verdict(filled[-1])
    elif orders:
        outcome, outcome_detail = "order not filled", _verdict(orders[-1])
    elif skips:
        outcome = f"skipped at {skips[-1].payload.get('stage')}"
        outcome_detail = str(skips[-1].payload.get("reason", ""))[:120]
    else:
        outcome, outcome_detail = "incomplete", ""

    return {
        "trace_id": trace_id,
        "inferred": trace_id.startswith("inferred-"),
        "symbol": symbols[0] if symbols else "",
        "first_seq": rows[0].seq,
        "start": rows[0].clock,
        "end": rows[-1].clock,
        "steps": len(rows),
        "outcome": outcome,
        "outcome_detail": outcome_detail,
        "guardrails": sum(1 for r in rows if r.is_guardrail),
        "chain": [
            {
                "seq": r.seq, "time": r.clock, "kind": r.kind,
                "stage": STAGE_OF_KIND.get(r.kind, r.kind),
                "actor": _actor(r), "action": r.action,
                "verdict": _verdict(r),
                "latency_ms": round(r.latency_ms) if r.latency_ms is not None else None,
                "guardrail": r.is_guardrail,
                "payload": r.payload,
            }
            for r in rows
        ],
    }


def discover(log_dir: Path, session: str | None = None) -> list[Path]:
    """Log files, newest last. One per session when rotation is on."""
    if session:
        exact = list(log_dir.glob(f"*{session}*.jsonl"))
        if exact:
            return sorted(exact)
    return sorted(log_dir.glob("decision_log*.jsonl"))
