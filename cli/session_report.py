"""Reconstruct a session from its decision log.

    python -m cli.session_report
    python -m cli.session_report --log logs/decision_log-2026-08-28.jsonl
    python -m cli.session_report --timeline

The decision log is the artifact this system exists to produce -- it backs the
demo, the write-up, and any question about why the agent did something. This
reads it back and answers the questions a reader actually has:

* what did each agent say, and what did the system do with it
* where did every candidate die, and why
* what orders went out, what filled, and at what price
* which kill switches were evaluated, and how close the others came

**Nothing here reads the broker.** If a fact is not in the log, the log has a
gap and this says so rather than filling it in from a live call -- an audit
trail that silently repairs itself is not an audit trail.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Iterable

STAGE_ORDER = [
    "a2_earnings", "a2", "a1", "signal", "prefilter", "a4", "a3",
    "order", "exit", "entry", "reconcile", "a5",
]


def load(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"no decision log at {path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _kind(row: dict) -> str:
    return row.get("kind") or row["payload"]["kind"]


def by_kind(rows: Iterable[dict], kind: str) -> list[dict]:
    return [r for r in rows if _kind(r) == kind]


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def report(rows: list[dict], timeline: bool = False) -> None:
    if not rows:
        raise SystemExit("the log is empty")

    sessions = sorted({r["session_date"] for r in rows})
    first, last = rows[0]["ts_et"][11:19], rows[-1]["ts_et"][11:19]

    section(f"SESSION {', '.join(sessions)}   {first} - {last} ET")
    print(f"  {len(rows)} records")
    counts = collections.Counter(_kind(r) for r in rows)
    for kind, n in counts.most_common():
        print(f"    {kind:<16} {n}")
    print("  actions: " + ", ".join(
        f"{a}={n}" for a, n in collections.Counter(r["action"] for r in rows).most_common()
    ))

    # -- agent calls --------------------------------------------------------
    calls = by_kind(rows, "agent_call")
    section(f"AGENT CALLS ({len(calls)})")
    if not calls:
        print("  none -- no agent was invoked this session")
    else:
        per = collections.defaultdict(lambda: collections.Counter())
        for row in calls:
            p = row["payload"]
            per[p["agent"]][p["validation"]["status"]] += 1
        print(f"  {'agent':<14}{'calls':>6}{'ok':>6}{'clamped':>9}{'failed':>8}"
              f"{'timeout':>9}{'median ms':>11}")
        for agent in sorted(per):
            statuses = per[agent]
            lat = sorted(
                r.get("latency_ms") or 0.0 for r in calls
                if r["payload"]["agent"] == agent
            )
            median = lat[len(lat) // 2] if lat else 0.0
            print(f"  {agent:<14}{sum(statuses.values()):>6}{statuses['ok']:>6}"
                  f"{statuses['clamped']:>9}{statuses['failed']:>8}"
                  f"{statuses['timeout']:>9}{median:>11.0f}")

        errors = [
            e for r in calls for e in r["payload"]["validation"].get("errors", [])
        ]
        if errors:
            print("\n  validation errors:")
            for err, n in collections.Counter(errors).most_common(6):
                print(f"    {n:>3}x {err[:100]}")

    # -- what the models actually decided -----------------------------------
    decided = [r for r in calls if r["payload"].get("response_parsed")]
    if decided:
        section(f"MODEL DECISIONS ({len(decided)})")
        for row in decided:
            p = row["payload"]
            print(f"  {row['ts_et'][11:19]}  {p['agent']:<12} "
                  f"{(row.get('symbol') or '-'):<6} {_summarise(p['response_parsed'])}")

    # -- overrides ----------------------------------------------------------
    overrides = by_kind(rows, "agent_override")
    if overrides:
        section(f"OVERRIDES ({len(overrides)})")
        print("  A clamp means the model returned something invalid.")
        print("  A force means it returned something legal and a rule overrode it.\n")
        for row in overrides:
            p = row["payload"]
            print(f"  {row['ts_et'][11:19]}  {p['override']:<6} {p['agent']:<12} "
                  f"{p['field']}: {p['model_value']} -> {p['applied_value']}  [{p['rule']}]")

    # -- skips --------------------------------------------------------------
    skips = by_kind(rows, "skip")
    section(f"SKIPS ({len(skips)})")
    if not skips:
        print("  none recorded")
    else:
        per_stage = collections.Counter(r["payload"]["stage"] for r in skips)
        print("  by stage (where the candidate died):")
        for stage in sorted(per_stage, key=lambda s: (
            STAGE_ORDER.index(s) if s in STAGE_ORDER else 99, s
        )):
            print(f"    {stage:<14} {per_stage[stage]}")
        print()
        for row in skips:
            p = row["payload"]
            print(f"  {row['ts_et'][11:19]}  [{p['stage']:<12}] "
                  f"{(row.get('symbol') or '-'):<6} {p['reason'][:96]}")

    # -- signals ------------------------------------------------------------
    signals = by_kind(rows, "signal_eval")
    if signals:
        section(f"SIGNAL EVALUATIONS ({len(signals)})")
        fired = [r for r in signals if r["payload"]["triggered"]]
        print(f"  triggered {len(fired)} of {len(signals)}")
        blocked = collections.Counter(
            gate for r in signals
            for gate, ok in r["payload"]["gates"].items() if not ok
        )
        if blocked:
            print("  blocking gates:")
            for gate, n in blocked.most_common():
                print(f"    {gate:<26} {n}")

    # -- prefilter ----------------------------------------------------------
    scans = by_kind(rows, "prefilter")
    if scans:
        section(f"CHAIN SCANS ({len(scans)})")
        print(f"  {'time':<10}{'symbol':<8}{'scanned':>9}{'survivors':>11}  top reasons")
        for row in scans:
            p = row["payload"]
            reasons = ", ".join(
                f"{k}={v}" for k, v in list(p.get("reason_counts", {}).items())[:3]
            )
            print(f"  {row['ts_et'][11:19]:<10}{(row.get('symbol') or '-'):<8}"
                  f"{p['total_contracts']:>9}{p['survivors']:>11}  {reasons}")

    # -- sizing and caps ----------------------------------------------------
    sizings = by_kind(rows, "sizing")
    if sizings:
        section(f"SIZING ({len(sizings)})")
        print(f"  {'time':<10}{'symbol':<8}{'premium':>10}{'budget':>10}"
              f"{'base':>6}{'final':>7}")
        for row in sizings:
            p = row["payload"]
            print(f"  {row['ts_et'][11:19]:<10}{(row.get('symbol') or '-'):<8}"
                  f"{p['premium_per_contract']:>10.2f}{p['risk_per_trade']:>10.2f}"
                  f"{p['base_contracts']:>6}{p['final_contracts']:>7}")

    caps = by_kind(rows, "cap_override")
    if caps:
        section(f"CAPS BOUND ({len(caps)})")
        for row in caps:
            p = row["payload"]
            print(f"  {row['ts_et'][11:19]}  {(row.get('symbol') or '-'):<6} "
                  f"{p['cap_name']:<34} requested {p['requested']:.2f} "
                  f"-> allowed {p['applied']:.2f}")

    # -- orders -------------------------------------------------------------
    orders = by_kind(rows, "order")
    section(f"ORDERS ({len(orders)})")
    if not orders:
        print("  none placed")
    else:
        print(f"  {'time':<10}{'intent':<15}{'contract':<24}{'qty':>4}"
              f"{'limit':>8}{'status':>12}{'filled':>7}{'avg':>8}")
        for row in orders:
            p = row["payload"]
            legs = ",".join(p.get("legs") or []) or "-"
            status = str(p.get("status") or "").split(".")[-1]
            print(f"  {row['ts_et'][11:19]:<10}{p['intent']:<15}{legs:<24}"
                  f"{p['qty']:>4}{(p.get('limit_price') or 0):>8.2f}{status:>12}"
                  f"{(p.get('filled_qty') or 0):>7.0f}"
                  f"{(p.get('filled_avg_price') or 0):>8.2f}")
        fills = [
            r for r in orders
            if (r["payload"].get("filled_qty") or 0) > 0
        ]
        print(f"\n  {len(fills)} of {len(orders)} order(s) filled")
        _round_trips(fills)

    # -- kill switches ------------------------------------------------------
    switches = by_kind(rows, "killswitch")
    section(f"KILL SWITCHES ({len(switches)} evaluations)")
    if not switches:
        print("  none evaluated")
    else:
        latest: dict[str, dict] = {}
        ever_fired: set[str] = set()
        for row in switches:
            p = row["payload"]
            latest[p["switch"]] = p
            if p["fired"]:
                ever_fired.add(p["switch"])
        print("  Every switch is evaluated every time, fired or not -- 'we were one")
        print("  trade from the halt' is what a review needs to know.\n")
        print(f"  {'switch':<28}{'threshold':>12}{'last observed':>15}"
              f"{'headroom':>11}  fired")
        for name in sorted(latest):
            p = latest[name]
            head = p["threshold"] - p["observed"]
            print(f"  {name:<28}{p['threshold']:>12.4f}{p['observed']:>15.4f}"
                  f"{head:>11.4f}  {'YES' if name in ever_fired else 'no'}")
        if ever_fired:
            print(f"\n  FIRED THIS SESSION: {', '.join(sorted(ever_fired))}")

    # -- session lifecycle --------------------------------------------------
    lifecycle = by_kind(rows, "session")
    if lifecycle:
        section(f"SESSION RECORDS ({len(lifecycle)})")
        for row in lifecycle[:3] + (["..."] if len(lifecycle) > 6 else []) + lifecycle[-3:]:
            if row == "...":
                print("  ...")
                continue
            p = row["payload"]
            print(f"  {row['ts_et'][11:19]}  {p['event']:<8} equity={p.get('equity')} "
                  f"positions={p.get('open_positions')}  {p.get('notes') or ''}")

    if timeline:
        section("FULL TIMELINE")
        for row in rows:
            print(f"  {row['seq']:>5} {row['ts_et'][11:19]} {_kind(row):<16}"
                  f"{row['action']:<10}{(row.get('symbol') or '-'):<7}"
                  f"{_one_line(row['payload'])}")


def _round_trips(fills: list[dict]) -> None:
    """Pair entries with exits on the same contract and price the round trip."""
    opens: dict[str, dict] = {}
    for row in fills:
        p = row["payload"]
        legs = p.get("legs") or []
        if not legs:
            continue
        symbol = legs[0]
        if p["intent"] == "buy_to_open":
            opens[symbol] = p
        elif p["intent"] == "sell_to_close" and symbol in opens:
            entry = opens.pop(symbol)
            a, b = entry.get("filled_avg_price"), p.get("filled_avg_price")
            if a and b:
                print(f"  round trip {symbol}: {a:.2f} -> {b:.2f} "
                      f"({(b - a) / a * 100:+.2f}% of premium)")
    for symbol in opens:
        print(f"  still open at the end of the log: {symbol}")


def _summarise(parsed: dict) -> str:
    if "regime" in parsed:
        sp = parsed.get("signal_profile", {})
        return (f"{parsed['regime']} conf={parsed['confidence']} "
                f"dir={sp.get('allowed_direction')} ema={sp.get('ema_fast')}")
    if "eligible" in parsed:
        return (f"eligible={parsed['eligible']} bias={parsed['directional_bias']}"
                f"/{parsed['bias_strength']} iv={parsed['iv_assessment']} "
                f"event={parsed['event_risk']}")
    if "size_multiplier" in parsed:
        return f"multiplier={parsed['size_multiplier']} -- {parsed.get('reason', '')[:50]}"
    if "structure" in parsed:
        return f"{parsed['structure']} {parsed['primary_symbol']} hold={parsed.get('expected_hold_sessions')}"
    if "action" in parsed:
        return f"{parsed['action']} stop={parsed.get('new_stop_pct')}"
    if "observations" in parsed:
        return f"{len(parsed['observations'])} observation(s)"
    return json.dumps(parsed)[:90]


def _one_line(payload: dict) -> str:
    skip = {"kind"}
    return " ".join(
        f"{k}={v}" for k, v in payload.items()
        if k not in skip and not isinstance(v, (dict, list))
    )[:110]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m cli.session_report")
    p.add_argument("--log", type=Path, default=None)
    p.add_argument("--timeline", action="store_true", help="every record, in order")
    args = p.parse_args(argv)

    path = args.log
    if path is None:
        from src.config import load_config

        config = load_config()
        candidates = sorted(config.log_dir.glob("decision_log-*.jsonl"))
        if not candidates:
            raise SystemExit(f"no decision log found in {config.log_dir}")
        path = candidates[-1]

    print(f"reading {path}")
    report(load(path), timeline=args.timeline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
