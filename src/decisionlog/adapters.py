"""Adapters from live objects to log payloads.

These live here rather than beside the things they describe because
``src/signals/`` must not import the logging layer -- purity is enforced by
test, and a signal module that knows how to log itself is no longer a pure
function over a dataframe.

The direction of dependency is deliberate: the log knows about the engine, the
engine knows nothing about the log.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping

from src.decisionlog.schema import (
    AgentCallPayload,
    PrefilterPayload,
    SignalEvalPayload,
    ValidationResult,
)

__all__ = [
    "agent_call_payload",
    "config_fingerprint",
    "prefilter_payload",
    "signal_eval_payload",
    "signal_action",
]


def signal_eval_payload(
    evaluation: Any,
    profile: Any,
    bar_ts: str,
    bar_count: int,
    profile_source: str = "default",
) -> SignalEvalPayload:
    """Wire ``SignalEvaluation`` straight through, gates included.

    Every gate is carried whether it passed or failed. A suppressed signal is
    a decision and must be as reconstructable as a triggered one -- "why was
    there no trade all morning" is the question the log most often has to
    answer.
    """
    return SignalEvalPayload(
        bar_ts=bar_ts,
        bar_count=bar_count,
        direction=evaluation.direction,
        triggered=evaluation.triggered,
        gates=dict(evaluation.gates),
        metrics={k: float(v) for k, v in evaluation.metrics.items()},
        profile={
            "ema_fast": profile.ema_fast,
            "confirmation_bars": profile.confirmation_bars,
            "require_vwap_alignment": profile.require_vwap_alignment,
            "min_atr_multiple": profile.min_atr_multiple,
            "allowed_direction": profile.allowed_direction,
        },
        profile_source=profile_source,  # type: ignore[arg-type]
    )


def signal_action(evaluation: Any) -> str:
    """The action actually taken, in the log's vocabulary."""
    return "signal_triggered" if evaluation.triggered else "signal_suppressed"


def prefilter_payload(
    candidates: Iterable[Any],
    thresholds: Mapping[str, float],
    underlying_price: float | None = None,
    detail: str = "boundary",
    near_boundary_pct: float = 0.20,
    keep_symbols: Iterable[str] = (),
) -> PrefilterPayload:
    """Multi-label prefilter outcome from a list of evaluated candidates.

    ``detail`` controls how much per-contract evidence is retained, because
    the aggregate is cheap and the per-contract rows are not -- a full scan is
    hundreds of contracts, several times a session, across the universe.

    * ``"aggregate"`` -- counts only.
    * ``"boundary"``  -- counts, plus per-contract rows for the ranked set and
      for **single-reason rejects that came within ``near_boundary_pct`` of
      passing**. Those are the only rejects a threshold change would move; a
      contract that failed on four counts tells you nothing about where to put
      a cap. This is the default.
    * ``"full"``      -- every rejected contract. Diagnostic use.

    Reason counts and ``sole_reason`` are always complete regardless of detail,
    so the aggregate view never depends on the retention setting.
    """
    candidates = list(candidates)
    keep = {s.upper() for s in keep_symbols}
    reason_counts: dict[str, int] = {}
    sole_reason: dict[str, int] = {}
    rejections: dict[str, list[str]] = {}
    survivors: list[str] = []

    for candidate in candidates:
        failures = list(getattr(candidate, "failures", ()) or ())
        symbol = candidate.symbol
        if not failures:
            survivors.append(symbol)
            continue

        for reason in failures:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if len(failures) == 1:
            sole_reason[failures[0]] = sole_reason.get(failures[0], 0) + 1

        if detail == "full":
            rejections[symbol] = failures
        elif detail == "boundary":
            distance = getattr(candidate, "boundary_distance", None)
            near = len(failures) == 1 and distance is not None and distance <= near_boundary_pct
            if near or symbol.upper() in keep:
                rejections[symbol] = failures

    return PrefilterPayload(
        underlying_price=underlying_price,
        total_contracts=len(candidates),
        survivors=len(survivors),
        rejected=len(candidates) - len(survivors),
        reason_counts=dict(sorted(reason_counts.items(), key=lambda kv: -kv[1])),
        sole_reason=sole_reason,
        survivor_symbols=sorted(survivors),
        rejections=rejections,
        thresholds={k: float(v) for k, v in thresholds.items()},
    )


def agent_call_payload(
    agent: str,
    model: str,
    rendered_prompt: str,
    response_raw: str | None,
    validation: ValidationResult,
    template_text: str | None = None,
    response_parsed: dict[str, Any] | None = None,
    max_response_chars: int | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    fallback_used: bool = False,
) -> AgentCallPayload:
    """Build an agent record from a prompt that is hashed and then discarded.

    ``rendered_prompt`` is consumed to produce a hash and a length and is not
    retained in the payload. There is no field it could be stored in.
    """
    from src.decisionlog.decision_log import prompt_hash

    truncated = False
    if response_raw is not None and max_response_chars and len(response_raw) > max_response_chars:
        response_raw = response_raw[:max_response_chars]
        truncated = True

    return AgentCallPayload(
        agent=agent,  # type: ignore[arg-type]
        model=model,
        prompt_hash=prompt_hash(rendered_prompt),
        prompt_template_hash=prompt_hash(template_text) if template_text else None,
        prompt_chars=len(rendered_prompt),
        response_raw=response_raw,
        response_parsed=response_parsed,
        response_truncated=truncated,
        validation=validation,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        fallback_used=fallback_used,
    )


def config_fingerprint(*sections: Mapping[str, Any]) -> str:
    """A hash of the thresholds in force, recorded instead of the values.

    ``config/limits.yaml`` is never committed, so a log that simply named its
    thresholds would leak them the moment the log was shared. The fingerprint
    proves two sessions ran on identical configuration without disclosing what
    that configuration was.
    """
    blob = json.dumps(list(sections), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
