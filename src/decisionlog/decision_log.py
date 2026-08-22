"""Append-only JSONL decision log: writer, credential scrubber, reader.

The log is the single most valuable artifact this project produces. It is
written once, never rewritten, and read by things that were not built yet when
it was written.

**Append-only is enforced, not assumed.** The file is opened in ``"a"`` mode,
which the OS guarantees appends even under concurrent writers, and there is no
code path in this module that opens it any other way. There is no update, no
delete, and no seek.

**Nothing secret can reach the log.** Two independent defences:

* The schema has no field for prompt text or credentials -- see
  ``schema.AgentCallPayload``. You cannot log a prompt because there is
  nowhere to put one.
* Every record is passed through :class:`Redactor` on the way out, which
  replaces the live credential values and anything matching a credential
  shape. That covers the case where a secret arrives inside a model response
  or a broker error string, which no schema can prevent.

Belt and braces is the right call here: the repository goes public, and a
leaked key in a committed log is unrecoverable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from collections.abc import Iterator
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from src.config import Config
from src.decisionlog.schema import (
    SCHEMA_VERSION,
    DecisionRecord,
    Payload,
    new_trace_id,
)

log = logging.getLogger(__name__)

try:  # pragma: no cover - trivial import shim
    from zoneinfo import ZoneInfo

    ET = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    ET = timezone.utc  # type: ignore[assignment]

__all__ = ["DecisionLog", "Redactor", "prompt_hash", "read_records", "reconstruct_session"]

REDACTION = "<redacted>"

# Credential shapes, for the case where a secret arrives inside a model
# response or a broker error rather than a field we control.
#
# The last pattern is the general high-entropy catch-all, and it needs care:
# a naive `[A-Za-z0-9+/]{40,}` also matches a sha256 hex digest, which would
# silently destroy `prompt_hash` -- the one field that makes prompts auditable
# without disclosing them. The lookaheads require upper, lower AND digit, so a
# lowercase hex digest is preserved while a real mixed-case key still matches.
# `_HASH_FIELDS` below is the second, independent guard on the same mistake.
_SECRET_PATTERNS = (
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\bPK[A-Z0-9]{16,}\b"),
    re.compile(r"\bAK[A-Z0-9]{16,}\b"),
    re.compile(
        r"\b(?=[A-Za-z0-9+/]*[A-Z])(?=[A-Za-z0-9+/]*[a-z])(?=[A-Za-z0-9+/]*[0-9])"
        r"[A-Za-z0-9+/]{40,}={0,2}\b"
    ),
)

# Fields whose whole purpose is to be a digest or an opaque id. Literal secret
# values are still replaced inside them -- that can never be wrong -- but the
# shape heuristics are not applied, because a hash looking like a secret is
# exactly what a hash is supposed to look like.
_HASH_FIELDS = frozenset(
    {
        "prompt_hash",
        "prompt_template_hash",
        "config_fingerprint",
        "record_id",
        "trace_id",
        "order_id",
    }
)


def prompt_hash(text: str) -> str:
    """sha256 of a rendered prompt. The only representation ever recorded."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Redactor:
    """Scrubs known credential values and credential-shaped strings.

    Known values come from :class:`~src.config.Env` and are matched literally,
    which catches a key however it was embedded. The patterns catch keys this
    process has never seen -- a key quoted back inside a broker error, for
    instance.
    """

    def __init__(self, secrets: Iterator[str] | None = None, enabled: bool = True) -> None:
        self.enabled = enabled
        # Longest first, so a key containing another as a prefix redacts whole.
        self._literals = sorted(
            {s for s in (secrets or []) if s and len(s) >= 8}, key=len, reverse=True
        )

    @classmethod
    def from_config(cls, config: Config) -> "Redactor":
        env = config.env
        enabled = config.limits.get_bool("decision_log.redact_env_keys")
        return cls(
            secrets=[
                value
                for value in (
                    env.alpaca_api_key,
                    env.alpaca_secret_key,
                    env.anthropic_api_key,
                )
                if value
            ],
            enabled=enabled,
        )

    def scrub_text(self, text: str, apply_patterns: bool = True) -> str:
        """Replace known secrets always; apply shape heuristics when asked."""
        if not self.enabled:
            return text
        for literal in self._literals:
            if literal in text:
                text = text.replace(literal, REDACTION)
        if apply_patterns:
            for pattern in _SECRET_PATTERNS:
                text = pattern.sub(REDACTION, text)
        return text

    def scrub(self, value: Any, key: str | None = None) -> Any:
        """Recursively scrub strings anywhere in a JSON-shaped structure.

        ``key`` carries the field name down so digest fields can opt out of
        the shape heuristics without opting out of literal replacement.
        """
        if not self.enabled:
            return value
        if isinstance(value, str):
            return self.scrub_text(value, apply_patterns=key not in _HASH_FIELDS)
        if isinstance(value, dict):
            return {name: self.scrub(item, key=name) for name, item in value.items()}
        if isinstance(value, list):
            return [self.scrub(item, key=key) for item in value]
        return value


class DecisionLog:
    """Append-only JSONL writer. One record per line, one line per decision.

    Not a context manager by design: the log outlives any single block and is
    held open for the session. ``close()`` exists for tests and shutdown.
    """

    def __init__(
        self,
        path: Path,
        redactor: Redactor | None = None,
        fsync: bool = False,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._redactor = redactor or Redactor(enabled=False)
        self._fsync = fsync
        self._lock = threading.Lock()
        # Continue the sequence rather than restarting it, so seq is monotonic
        # across a restart mid-session.
        self._seq = self._count_existing_lines()
        self._handle = open(self.path, "a", encoding="utf-8", newline="\n")

    @classmethod
    def from_config(cls, config: Config, session_date: date | None = None) -> "DecisionLog":
        filename = config.limits.get_str("decision_log.filename")
        if config.limits.get_bool("decision_log.rotate_daily"):
            day = (session_date or datetime.now(tz=ET).date()).isoformat()
            stem, _, suffix = filename.rpartition(".")
            filename = f"{stem or filename}-{day}.{suffix or 'jsonl'}"
        return cls(
            path=config.ensure_log_dir() / filename,
            redactor=Redactor.from_config(config),
            fsync=config.limits.get_bool("decision_log.fsync_every_record"),
        )

    def _count_existing_lines(self) -> int:
        if not self.path.exists():
            return 0
        with open(self.path, "r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)

    def write(
        self,
        payload: Payload,
        action: str,
        *,
        symbol: str | None = None,
        reasons: list[str] | None = None,
        trace_id: str | None = None,
        latency_ms: float | None = None,
        at: datetime | None = None,
    ) -> DecisionRecord:
        """Append one decision. Returns the record as written.

        ``action`` is what actually happened -- the decision's effect, not the
        model's suggestion. When a cap overrides a model, the action is the
        capped outcome and both values live in the payload.
        """
        moment = at or datetime.now(tz=timezone.utc)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        eastern = moment.astimezone(ET)

        with self._lock:
            self._seq += 1
            record = DecisionRecord(
                seq=self._seq,
                ts_utc=moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                ts_et=eastern.isoformat(),
                session_date=eastern.date().isoformat(),
                trace_id=trace_id,
                symbol=symbol.upper() if symbol else None,
                action=action,
                reasons=list(reasons or []),
                latency_ms=latency_ms,
                payload=payload,
            )
            line = json.dumps(
                self._redactor.scrub(record.model_dump(mode="json")),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            if "\n" in line:  # pragma: no cover - json.dumps escapes newlines
                raise ValueError("record serialised to multiple lines")
            self._handle.write(line + "\n")
            self._handle.flush()
            if self._fsync:
                import os

                os.fsync(self._handle.fileno())
        return record

    def close(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.close()

    @property
    def records_written(self) -> int:
        return self._seq

    def __repr__(self) -> str:
        return f"DecisionLog(path={self.path.name!r}, records={self._seq})"


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def read_records(path: Path, strict: bool = False) -> list[dict[str, Any]]:
    """Read a log back.

    A truncated final line -- the process died mid-write -- is skipped rather
    than fatal, because the other 4,000 decisions are still worth having.
    ``strict=True`` raises instead, for tests.
    """
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                if strict:
                    raise
                log.warning("%s:%d is not valid JSON -- skipping", path.name, lineno)
    return records


def reconstruct_session(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Rebuild a session summary from the log alone.

    This is the acceptance criterion made executable: everything here is
    derived from the log with no access to config, to the code that produced
    it, or to the broker. If a number the write-up needs cannot be computed
    here, the schema is missing a field.
    """
    versions = {r.get("schema_version") for r in records}
    unknown = versions - {SCHEMA_VERSION}
    if unknown:
        log.warning("log contains unknown schema versions: %s", sorted(unknown))

    by_kind: dict[str, int] = {}
    for record in records:
        by_kind[record["kind"]] = by_kind.get(record["kind"], 0) + 1

    def of_kind(kind: str) -> list[dict[str, Any]]:
        return [r for r in records if r["kind"] == kind]

    signals = of_kind("signal_eval")
    orders = of_kind("order")
    agent_calls = of_kind("agent_call")

    gate_failures: dict[str, int] = {}
    for record in signals:
        for gate, passed in record["payload"].get("gates", {}).items():
            if not passed:
                gate_failures[gate] = gate_failures.get(gate, 0) + 1

    prefilter_reasons: dict[str, int] = {}
    for record in of_kind("prefilter"):
        for reason, count in record["payload"].get("reason_counts", {}).items():
            prefilter_reasons[reason] = prefilter_reasons.get(reason, 0) + count

    fills = [r for r in orders if r["payload"].get("status") == "filled"]
    traces = {r["trace_id"] for r in records if r.get("trace_id")}

    latencies = [r["latency_ms"] for r in agent_calls if r.get("latency_ms") is not None]

    return {
        "schema_versions": sorted(v for v in versions if v is not None),
        "records": len(records),
        "sessions": sorted({r["session_date"] for r in records}),
        "symbols": sorted({r["symbol"] for r in records if r.get("symbol")}),
        "by_kind": dict(sorted(by_kind.items())),
        "traces": len(traces),
        "signals_evaluated": len(signals),
        "signals_triggered": sum(1 for r in signals if r["payload"].get("triggered")),
        "signal_gate_failures": dict(sorted(gate_failures.items(), key=lambda kv: -kv[1])),
        "prefilter_reason_counts": dict(sorted(prefilter_reasons.items(), key=lambda kv: -kv[1])),
        "agent_calls": len(agent_calls),
        "agent_validation": {
            status: sum(
                1 for r in agent_calls if r["payload"]["validation"]["status"] == status
            )
            for status in sorted(
                {r["payload"]["validation"]["status"] for r in agent_calls}
            )
        },
        "agent_latency_ms": {
            "count": len(latencies),
            "mean": round(sum(latencies) / len(latencies), 2) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "cap_overrides": len(of_kind("cap_override")),
        "killswitch_fires": sum(
            1 for r in of_kind("killswitch") if r["payload"].get("fired")
        ),
        "orders_submitted": len(orders),
        "orders_filled": len(fills),
        "actions": dict(
            sorted(
                {
                    action: sum(1 for r in records if r["action"] == action)
                    for action in {r["action"] for r in records}
                }.items()
            )
        ),
        "sequence_is_contiguous": [r["seq"] for r in records] == list(range(1, len(records) + 1)),
    }


def make_trace_id() -> str:
    """Re-exported for callers that only import this module."""
    return new_trace_id()
