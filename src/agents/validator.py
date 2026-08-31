"""Parse, clamp, force, retry, fail closed. The layer between model and system.

Nothing here trusts a model response. Every value that reaches the rest of the
system has either been checked against the schema or replaced by this module,
and every replacement is recorded with both values.

**Clamping and forcing are different operations and must stay distinguishable.**

* A **clamp** moves an out-of-range value to the nearest permitted one. The
  model returned something invalid: ``ema_fast: 7`` when the allowed set is
  ``[5, 8, 9, 12, 21]``. Clamps are evidence about model quality.
* A **force** overrides a perfectly valid value because a rule fired.
  ``allowed_direction: long_calls`` is a legal answer; it becomes ``none``
  because confidence sat below the floor, or because the regime is on the
  forced list. Forces say nothing about the model's output quality -- they are
  the system asserting policy over a well-formed answer.

Counting them together would make a well-behaved model on a choppy day
indistinguishable from one emitting garbage, so they are separate kinds in the
log and separate counters here.

**Retry is exactly once.** A validation failure is fed back to the caller as
text to append to the prompt, and the model gets one more attempt. A second
failure is a skip -- never a partial parse, never a best guess, never a
default-filled object. Failing closed is the only safe reading of "the model
did not answer the question".
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable, Sequence

from pydantic import BaseModel, ValidationError

from src.agents.schemas import (
    AGENT_SCHEMAS,
    ContextDecision,
    ContractDecision,
    Direction,
    ExitAction,
    ExitDecision,
    Observation,
    RegimeDecision,
    ReviewDecision,
    RiskDecision,
)

log = logging.getLogger(__name__)

AGENT_LOG_NAMES = {
    "a1": "a1_regime", "a2": "a2_context", "a3": "a3_risk",
    "a4": "a4_contract", "a5": "a5_exit", "a6": "a6_review",
}


class OverrideKind(str, Enum):
    CLAMP = "clamp"
    FORCE = "force"


@dataclass(frozen=True)
class Override:
    """One replacement, with both values side by side.

    ``model_value`` is what the model actually said and ``applied_value`` is
    what the system used. Neither is derivable from the other, so the log
    carries both and the reader never has to guess which one a downstream
    action was based on.
    """

    kind: OverrideKind
    field: str
    model_value: Any
    applied_value: Any
    rule: str
    detail: str = ""

    def as_payload_kwargs(self, agent_key: str) -> dict[str, Any]:
        return {
            "agent": AGENT_LOG_NAMES[agent_key],
            "override": self.kind.value,
            "field": self.field,
            "model_value": _jsonable(self.model_value),
            "applied_value": _jsonable(self.applied_value),
            "rule": self.rule,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ValidationOutcome:
    agent: str
    decision: BaseModel | None
    overrides: tuple[Override, ...] = ()
    errors: tuple[str, ...] = ()
    attempts: int = 1

    @property
    def ok(self) -> bool:
        return self.decision is not None

    @property
    def clamps(self) -> tuple[Override, ...]:
        return tuple(o for o in self.overrides if o.kind is OverrideKind.CLAMP)

    @property
    def forces(self) -> tuple[Override, ...]:
        return tuple(o for o in self.overrides if o.kind is OverrideKind.FORCE)

    @property
    def status(self) -> str:
        """Summary for ``ValidationResult.status``.

        ``clamped`` is reserved for a response the model got wrong. A response
        that was only forced is reported ``ok``: the model answered correctly
        and policy overrode it, which is not a model failure.
        """
        if self.decision is None:
            return "failed"
        return "clamped" if self.clamps else "ok"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


class _Collector:
    """Accumulates overrides while rules run."""

    def __init__(self) -> None:
        self.items: list[Override] = []

    def clamp(self, field_name: str, was: Any, now: Any, rule: str, detail: str = "") -> None:
        if was == now:
            return
        self.items.append(Override(OverrideKind.CLAMP, field_name, was, now, rule, detail))

    def force(self, field_name: str, was: Any, now: Any, rule: str, detail: str = "") -> None:
        # A rule that fires without changing anything is not an override. The
        # model already agreed; recording it would bury real overrides in
        # no-ops.
        if was == now:
            return
        self.items.append(Override(OverrideKind.FORCE, field_name, was, now, rule, detail))


# --- parsing ---------------------------------------------------------------


def parse(agent_key: str, raw: Any) -> tuple[BaseModel | None, str | None]:
    """Raw model output to a validated contract, or an error string.

    Accepts a mapping or a JSON string. A model that wraps its JSON in prose or
    a fenced block has not followed the contract, and the error says so rather
    than trying to excavate an object from the text -- salvaging malformed
    output is how a best guess reaches the broker.
    """
    model = AGENT_SCHEMAS.get(agent_key)
    if model is None:
        return None, f"unknown agent {agent_key!r}"

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            return None, f"response is not valid JSON: {exc}"
    if not isinstance(raw, dict):
        return None, f"expected a JSON object, got {type(raw).__name__}"

    try:
        return model.model_validate(raw), None
    except ValidationError as exc:
        return None, _readable(exc)


def _readable(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)


# --- per-agent rules -------------------------------------------------------


def _nearest(value: float, allowed: Sequence[float]) -> float:
    return min(allowed, key=lambda a: (abs(a - value), a))


def _rules_a1(d: RegimeDecision, limits: Any, ctx: dict[str, Any], c: _Collector) -> RegimeDecision:
    allowed_ema = sorted(limits.get_int_set("agents.a1.allowed_ema_fast"))
    allowed_bars = sorted(limits.get_int_set("agents.a1.allowed_confirmation_bars"))
    floor = limits.get_float("agents.a1.min_atr_multiple_floor")
    ceiling = limits.get_float("agents.a1.min_atr_multiple_ceiling")
    confidence_floor = limits.get_float("agents.a1.confidence_floor")
    forced_regimes = {r.lower() for r in limits.get_str_set("agents.a1.force_none_regimes")}

    p = d.signal_profile
    updates: dict[str, Any] = {}

    if p.ema_fast not in allowed_ema:
        picked = int(_nearest(p.ema_fast, allowed_ema))
        c.clamp("signal_profile.ema_fast", p.ema_fast, picked,
                "agents.a1.allowed_ema_fast", f"allowed {allowed_ema}")
        updates["ema_fast"] = picked
    if p.confirmation_bars not in allowed_bars:
        picked = int(_nearest(p.confirmation_bars, allowed_bars))
        c.clamp("signal_profile.confirmation_bars", p.confirmation_bars, picked,
                "agents.a1.allowed_confirmation_bars", f"allowed {allowed_bars}")
        updates["confirmation_bars"] = picked
    # require_vwap_alignment is FORCED OFF, 31 Aug 2026. The indicator is
    # degenerate on the daily frames this system evaluates -- a session-anchored
    # VWAP over daily bars is that bar's own typical price -- so the option is
    # removed from Agent 1's effective choices rather than left as a switch the
    # model can flip to no purpose. Kept in the schema because the field is a
    # published contract and the gate becomes meaningful again the day an
    # intraday frame is evaluated. A force, not a clamp: the model's answer was
    # legal and policy overrode it.
    if p.require_vwap_alignment and not limits.get_bool("agents.a1.allow_vwap_alignment"):
        c.force("signal_profile.require_vwap_alignment", True, False,
                "agents.a1.allow_vwap_alignment",
                "vwap is degenerate on a daily frame; gate disabled")
        updates["require_vwap_alignment"] = False

    if not floor <= p.min_atr_multiple <= ceiling:
        picked = min(max(p.min_atr_multiple, floor), ceiling)
        c.clamp("signal_profile.min_atr_multiple", p.min_atr_multiple, picked,
                "agents.a1.min_atr_multiple_floor/ceiling", f"band [{floor}, {ceiling}]")
        updates["min_atr_multiple"] = picked

    # Forces: the value was legal, a rule overrode it.
    direction = updates.get("allowed_direction", p.allowed_direction)
    if d.regime.value.lower() in forced_regimes and direction is not Direction.NONE:
        c.force("signal_profile.allowed_direction", direction, Direction.NONE,
                "agents.a1.force_none_regimes", f"regime {d.regime.value!r} is on the forced list")
        updates["allowed_direction"] = Direction.NONE
    elif d.confidence < confidence_floor and direction is not Direction.NONE:
        c.force("signal_profile.allowed_direction", direction, Direction.NONE,
                "agents.a1.confidence_floor",
                f"confidence {d.confidence} below floor {confidence_floor}")
        updates["allowed_direction"] = Direction.NONE

    if not updates:
        return d
    return d.model_copy(update={"signal_profile": p.model_copy(update=updates)})


def _rules_a2(d: ContextDecision, limits: Any, ctx: dict[str, Any], c: _Collector) -> ContextDecision:
    blocking = {v.lower() for v in limits.get_str_set("agents.a2.blocking_event_risk")}
    blocked_iv = {v.lower() for v in limits.get_str_set("agents.a2.blocked_iv_assessments")}
    min_bias = limits.get_float("agents.a2.min_bias_strength")

    if not d.eligible:
        return d

    if d.event_risk.value.lower() in blocking:
        c.force("eligible", True, False, "agents.a2.blocking_event_risk",
                f"event_risk {d.event_risk.value!r} blocks entry")
        return d.model_copy(update={"eligible": False})
    if d.iv_assessment.value.lower() in blocked_iv:
        c.force("eligible", True, False, "agents.a2.blocked_iv_assessments",
                f"iv_assessment {d.iv_assessment.value!r} blocks entry")
        return d.model_copy(update={"eligible": False})
    if d.bias_strength < min_bias:
        c.force("eligible", True, False, "agents.a2.min_bias_strength",
                f"bias_strength {d.bias_strength} below {min_bias}")
        return d.model_copy(update={"eligible": False})
    return d


def _rules_a3(d: RiskDecision, limits: Any, ctx: dict[str, Any], c: _Collector) -> RiskDecision:
    veto_below = limits.get_float("agents.a3.veto_below_multiplier")
    if 0.0 < d.size_multiplier < veto_below:
        # Not a clamp: the value was inside [0, 1] and legal. The rule says a
        # position this small is not worth its friction, so it becomes a veto.
        c.force("size_multiplier", d.size_multiplier, 0.0,
                "agents.a3.veto_below_multiplier",
                f"below {veto_below}; too small to be worth the round trip")
        return d.model_copy(update={"size_multiplier": 0.0})
    return d


def _rules_a4(d: ContractDecision, limits: Any, ctx: dict[str, Any], c: _Collector) -> ContractDecision:
    allowed = {s.lower() for s in limits.get_str_set("agents.a4.allowed_structures")}
    max_hold = limits.get_int("agents.a4.max_expected_hold_sessions")

    if d.structure.value.lower() not in allowed:
        # Deliberately not clamped. Rewriting a vertical into a single leg
        # would change the position the model reasoned about, so this fails and
        # the caller falls back to the deterministic survivor.
        raise _RuleFailure(
            f"structure {d.structure.value!r} not in allowed_structures {sorted(allowed)}"
        )

    survivors = ctx.get("survivors")
    if survivors is not None:
        allowed_symbols = {s.upper() for s in survivors}
        for name in ("primary_symbol", "short_symbol", "alternate_symbol"):
            value = getattr(d, name)
            if value and value.upper() not in allowed_symbols:
                raise _RuleFailure(f"{name} {value!r} is not in the survivor set")

    if d.expected_hold_sessions > max_hold:
        c.clamp("expected_hold_sessions", d.expected_hold_sessions, max_hold,
                "agents.a4.max_expected_hold_sessions", f"max {max_hold} sessions")
        return d.model_copy(update={"expected_hold_sessions": max_hold})
    return d


def _rules_a5(d: ExitDecision, limits: Any, ctx: dict[str, Any], c: _Collector) -> ExitDecision:
    max_stop = limits.get_float("agents.a5.max_stop_pct")
    min_tighten = limits.get_float("agents.a5.min_stop_tightening_pct")
    allow_scale = limits.get_bool("agents.a5.allow_scale_out")

    if d.action is ExitAction.SCALE_OUT_HALF and not allow_scale:
        c.force("action", d.action, ExitAction.HOLD, "agents.a5.allow_scale_out",
                "scale-out disabled")
        return d.model_copy(update={"action": ExitAction.HOLD})

    if d.action is not ExitAction.TIGHTEN_STOP:
        return d

    proposed = d.new_stop_pct
    assert proposed is not None  # schema guarantees this pairing
    if proposed < max_stop:
        c.clamp("new_stop_pct", proposed, max_stop, "agents.a5.max_stop_pct",
                f"floor {max_stop}")
        proposed = max_stop

    current = ctx.get("current_stop_pct")
    if current is None:
        raise _RuleFailure(
            "current_stop_pct is required to validate a tighten_stop -- "
            "tightening cannot be checked against an unknown stop"
        )

    if proposed <= current:
        # The monotone invariant. Not clamped to "a bit tighter": the model
        # asked to widen, and inventing a tightening it did not request would
        # be a best guess in the one place they are least acceptable.
        c.force("action", d.action, ExitAction.HOLD, "monotone_stop_invariant",
                f"proposed {proposed} does not tighten current {current}")
        return d.model_copy(update={"action": ExitAction.HOLD, "new_stop_pct": None})

    if proposed - current < min_tighten:
        c.force("action", d.action, ExitAction.HOLD, "agents.a5.min_stop_tightening_pct",
                f"tightening {proposed - current:.2f} below minimum {min_tighten}")
        return d.model_copy(update={"action": ExitAction.HOLD, "new_stop_pct": None})

    if proposed != d.new_stop_pct:
        return d.model_copy(update={"new_stop_pct": proposed})
    return d


def _rules_a6(d: ReviewDecision, limits: Any, ctx: dict[str, Any], c: _Collector) -> ReviewDecision:
    max_obs = limits.get_int("agents.a6.max_observations")
    max_chars = limits.get_int("agents.a6.max_observation_chars")
    max_expiry = limits.get_int("agents.a6.max_expires_after_sessions")
    max_global = limits.get_int("agents.a6.max_global_scope_observations")

    kept: list[Observation] = []
    globals_kept = 0
    for obs in d.observations:
        if len(kept) >= max_obs:
            c.clamp("observations", len(d.observations), max_obs,
                    "agents.a6.max_observations", "surplus observations dropped")
            break
        if obs.scope.lower() == "global":
            if globals_kept >= max_global:
                c.clamp("observations[global]", obs.text, None,
                        "agents.a6.max_global_scope_observations", "surplus global observation dropped")
                continue
            globals_kept += 1
        updates: dict[str, Any] = {}
        if len(obs.text) > max_chars:
            c.clamp("observation.text", len(obs.text), max_chars,
                    "agents.a6.max_observation_chars", "text truncated")
            updates["text"] = obs.text[:max_chars]
        if obs.expires_after_sessions > max_expiry:
            c.clamp("observation.expires_after_sessions", obs.expires_after_sessions, max_expiry,
                    "agents.a6.max_expires_after_sessions", "")
            updates["expires_after_sessions"] = max_expiry
        kept.append(obs.model_copy(update=updates) if updates else obs)

    if tuple(kept) == d.observations:
        return d
    return d.model_copy(update={"observations": tuple(kept)})


class _RuleFailure(Exception):
    """A rule that cannot be satisfied by clamping. Fails the response."""


_RULES: dict[str, Callable[..., BaseModel]] = {
    "a1": _rules_a1, "a2": _rules_a2, "a3": _rules_a3,
    "a4": _rules_a4, "a5": _rules_a5, "a6": _rules_a6,
}


# --- entry points ----------------------------------------------------------


def validate(agent_key: str, raw: Any, limits: Any, **context: Any) -> ValidationOutcome:
    """One attempt: parse, then apply this agent's rules."""
    decision, error = parse(agent_key, raw)
    if decision is None:
        return ValidationOutcome(agent=agent_key, decision=None, errors=(error or "unparseable",))

    collector = _Collector()
    try:
        decision = _RULES[agent_key](decision, limits, context, collector)
    except _RuleFailure as exc:
        return ValidationOutcome(agent=agent_key, decision=None, errors=(str(exc),),
                                 overrides=tuple(collector.items))
    return ValidationOutcome(agent=agent_key, decision=decision,
                             overrides=tuple(collector.items))


def validate_with_retry(
    agent_key: str,
    call: Callable[[str | None], Any],
    limits: Any,
    on_override: Callable[[str, Override], None] | None = None,
    **context: Any,
) -> ValidationOutcome:
    """Validate, and on failure retry exactly once with the error fed back.

    ``call(feedback)`` returns raw model output. It is invoked first with
    ``None`` and, if validation fails, once more with the validation error so
    the prompt can carry it. A second failure returns a failed outcome -- the
    caller skips. There is no third attempt and no partial result.

    ``on_override`` fires once per override, in order, so each becomes its own
    decision-log entry rather than being rolled into a summary.
    """
    outcome = validate(agent_key, call(None), limits, **context)
    if not outcome.ok:
        first_errors = outcome.errors
        retry = validate(agent_key, call("; ".join(first_errors)), limits, **context)
        outcome = ValidationOutcome(
            agent=agent_key,
            decision=retry.decision,
            overrides=retry.overrides,
            errors=first_errors + retry.errors,
            attempts=2,
        )
        if not outcome.ok:
            log.warning("%s failed validation twice, skipping: %s", agent_key,
                        "; ".join(outcome.errors))

    if on_override:
        for item in outcome.overrides:
            on_override(agent_key, item)
    return outcome


def truncate_eligible(
    decisions: Iterable[ContextDecision], limits: Any
) -> tuple[tuple[ContextDecision, ...], tuple[Override, ...]]:
    """Rank the eligible set by bias strength and cut it to the configured max.

    A set-level rule: no single response is wrong, so this is a force rather
    than a clamp. Ranking is by ``bias_strength`` descending, with symbol as a
    tiebreak so the cut is deterministic across runs.
    """
    max_eligible = limits.get_int("agents.a2.max_eligible_symbols")
    eligible = [d for d in decisions if d.eligible]
    ranked = sorted(eligible, key=lambda d: (-d.bias_strength, d.symbol))
    if len(ranked) <= max_eligible:
        return tuple(ranked), ()
    kept, dropped = ranked[:max_eligible], ranked[max_eligible:]
    override = Override(
        OverrideKind.FORCE, "eligible_set",
        [d.symbol for d in ranked], [d.symbol for d in kept],
        "agents.a2.max_eligible_symbols",
        f"dropped {[d.symbol for d in dropped]} ranked below the top {max_eligible}",
    )
    return tuple(kept), (override,)
