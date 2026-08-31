"""Validator. Weighted to what happens when the model misbehaves.

The happy path is one test. Everything else is malformed output, out-of-range
values, rules firing, and the retry-then-skip path.
"""
from __future__ import annotations

import pytest

from src.agents.schemas import (
    Bias,
    ContextDecision,
    Direction,
    EventRisk,
    ExitAction,
    IvAssessment,
    Structure,
)
from src.agents.validator import (
    OverrideKind,
    parse,
    truncate_eligible,
    validate,
    validate_with_retry,
)


@pytest.fixture(scope="module")
def limits():
    from src.config import load_config

    return load_config().limits


def a1_payload(**kw):
    # False so the vwap force does not fire: these tests are about other
    # rules, and an unrelated override would mask the one under test.
    profile = dict(ema_fast=9, confirmation_bars=2, require_vwap_alignment=False,
                   min_atr_multiple=0.6, allowed_direction="long_calls")
    profile.update(kw.pop("profile", {}))
    base = dict(symbol="NVDA", regime="trending_up", confidence=0.9,
                signal_profile=profile, rationale="trend")
    base.update(kw)
    return base


# --- clamp vs force: the distinction the log depends on --------------------


def test_a_clamp_records_both_values_and_the_rule(limits):
    out = validate("a1", a1_payload(profile={"ema_fast": 7}), limits)
    assert out.ok
    [clamp] = out.clamps
    assert clamp.kind is OverrideKind.CLAMP
    assert clamp.field == "signal_profile.ema_fast"
    assert clamp.model_value == 7          # what the model said
    assert clamp.applied_value == 8        # what the system used
    assert clamp.rule == "agents.a1.allowed_ema_fast"
    assert out.decision.signal_profile.ema_fast == 8


def test_a_force_is_not_a_clamp(limits):
    """Confidence below the floor overrides a legal value. Not model error."""
    out = validate("a1", a1_payload(confidence=0.1), limits)
    assert out.ok
    assert out.clamps == ()
    [force] = out.forces
    assert force.kind is OverrideKind.FORCE
    assert force.field == "signal_profile.allowed_direction"
    assert force.model_value is Direction.LONG_CALLS
    assert force.applied_value is Direction.NONE
    assert force.rule == "agents.a1.confidence_floor"
    assert out.decision.signal_profile.allowed_direction is Direction.NONE


def test_status_reflects_clamps_but_not_forces(limits):
    """A forced-only response is 'ok' -- the model answered correctly."""
    assert validate("a1", a1_payload(confidence=0.1), limits).status == "ok"
    assert validate("a1", a1_payload(profile={"ema_fast": 7}), limits).status == "clamped"


def test_a_choppy_regime_forces_direction_none(limits):
    out = validate("a1", a1_payload(regime="choppy"), limits)
    [force] = out.forces
    assert force.rule == "agents.a1.force_none_regimes"
    assert out.decision.signal_profile.allowed_direction is Direction.NONE


def test_a_rule_that_changes_nothing_is_not_logged(limits):
    """Choppy plus direction already none: the rule fires, nothing changed."""
    out = validate("a1", a1_payload(regime="choppy",
                                    profile={"allowed_direction": "none"}), limits)
    assert out.overrides == ()
    assert out.status == "ok"


def test_min_atr_multiple_is_clamped_into_the_band(limits):
    out = validate("a1", a1_payload(profile={"min_atr_multiple": 9.0}), limits)
    [clamp] = out.clamps
    assert clamp.model_value == 9.0
    assert clamp.applied_value == 1.5


# --- malformed output ------------------------------------------------------


@pytest.mark.parametrize("raw,fragment", [
    ("not json at all", "not valid JSON"),
    ('```json\n{"symbol": "NVDA"}\n```', "not valid JSON"),
    ("[1, 2, 3]", "expected a JSON object"),
    ("null", "expected a JSON object"),
])
def test_unparseable_responses_fail_closed(raw, fragment, limits):
    out = validate("a1", raw, limits)
    assert not out.ok and out.decision is None
    assert fragment in out.errors[0]


def test_a_fenced_block_is_not_excavated(limits):
    """Salvaging JSON out of prose is how a best guess reaches the broker."""
    out = validate("a1", '```\n{"size_multiplier": 0.5}\n```', limits)
    assert not out.ok


def test_an_invented_field_fails_rather_than_being_ignored(limits):
    payload = a1_payload()
    payload["confidence_level"] = "very high"
    out = validate("a1", payload, limits)
    assert not out.ok
    assert "confidence_level" in out.errors[0]


def test_the_error_names_the_offending_field(limits):
    out = validate("a1", a1_payload(confidence=5.0), limits)
    assert not out.ok
    assert "confidence" in out.errors[0]


# --- retry: exactly once ---------------------------------------------------


def test_retry_feeds_the_error_back_and_succeeds(limits):
    seen = []

    def call(feedback):
        seen.append(feedback)
        return a1_payload() if feedback else "garbage"

    out = validate_with_retry("a1", call, limits)
    assert out.ok and out.attempts == 2
    assert seen[0] is None
    assert "not valid JSON" in seen[1]      # the error reached the prompt


def test_a_second_failure_is_a_skip_not_a_guess(limits):
    calls = []

    def call(feedback):
        calls.append(feedback)
        return "still garbage"

    out = validate_with_retry("a1", call, limits)
    assert not out.ok
    assert out.decision is None            # no partial, no defaults
    assert out.attempts == 2
    assert len(calls) == 2                 # never a third attempt


def test_a_clean_first_answer_is_not_retried(limits):
    calls = []

    def call(feedback):
        calls.append(feedback)
        return a1_payload()

    out = validate_with_retry("a1", call, limits)
    assert out.ok and out.attempts == 1 and len(calls) == 1


def test_a_timeout_propagates_rather_than_becoming_a_default(limits):
    """The runner decides what a timeout means; it must not parse as success."""
    def call(feedback):
        raise TimeoutError("model call timed out")

    with pytest.raises(TimeoutError):
        validate_with_retry("a1", call, limits)


def test_every_override_is_offered_for_logging_in_order(limits):
    seen = []
    out = validate_with_retry(
        "a1", lambda fb: a1_payload(confidence=0.1, profile={"ema_fast": 7}),
        limits, on_override=lambda agent, o: seen.append((agent, o.kind, o.field)),
    )
    assert out.ok
    assert [k for _, k, _ in seen] == [OverrideKind.CLAMP, OverrideKind.FORCE]
    assert all(agent == "a1" for agent, _, _ in seen)


# --- Agent 2 ---------------------------------------------------------------


def a2_payload(**kw):
    base = dict(symbol="AMD", eligible=True, hard_blocks=[], directional_bias="bullish",
                bias_strength=0.8, event_risk="low", iv_assessment="fair", notes="")
    base.update(kw)
    return base


@pytest.mark.parametrize("field,value,rule", [
    ("event_risk", "high", "agents.a2.blocking_event_risk"),
    ("iv_assessment", "rich", "agents.a2.blocked_iv_assessments"),
    ("bias_strength", 0.1, "agents.a2.min_bias_strength"),
])
def test_a2_rules_force_ineligible(field, value, rule, limits):
    out = validate("a2", a2_payload(**{field: value}), limits)
    assert out.ok and out.decision.eligible is False
    [force] = out.forces
    assert force.kind is OverrideKind.FORCE
    assert force.model_value is True and force.applied_value is False
    assert force.rule == rule


def test_truncation_keeps_the_strongest_and_names_the_dropped(limits):
    decisions = [ContextDecision.model_validate(a2_payload(symbol=s, bias_strength=b))
                 for s, b in [("A", 0.9), ("B", 0.5), ("C", 0.7), ("D", 0.6)]]
    kept, overrides = truncate_eligible(decisions, limits)
    assert [d.symbol for d in kept] == ["A", "C", "D"]      # max_eligible_symbols = 3
    [ov] = overrides
    assert ov.kind is OverrideKind.FORCE                    # no response was wrong
    assert ov.applied_value == ["A", "C", "D"]
    assert "B" in ov.detail


def test_truncation_below_the_cap_is_not_an_override(limits):
    decisions = [ContextDecision.model_validate(a2_payload(symbol="A"))]
    kept, overrides = truncate_eligible(decisions, limits)
    assert len(kept) == 1 and overrides == ()


# --- Agent 3 ---------------------------------------------------------------


def test_a_tiny_multiplier_becomes_a_veto(limits):
    out = validate("a3", {"size_multiplier": 0.1, "reason": "small"}, limits)
    assert out.ok and out.decision.size_multiplier == 0.0
    [force] = out.forces
    assert force.kind is OverrideKind.FORCE     # 0.1 was legal, policy vetoed it
    assert force.rule == "agents.a3.veto_below_multiplier"


def test_an_explicit_zero_is_left_alone(limits):
    out = validate("a3", {"size_multiplier": 0.0, "reason": "veto"}, limits)
    assert out.ok and out.overrides == ()


# --- Agent 4 ---------------------------------------------------------------


def a4_payload(**kw):
    base = dict(structure="single_leg", primary_symbol="NVDA261016C00185000",
                expected_hold_sessions=3, reason="r")
    base.update(kw)
    return base


def test_a_symbol_outside_the_survivor_set_fails(limits):
    out = validate("a4", a4_payload(), limits, survivors=["SPY261016C00765000"])
    assert not out.ok
    assert "survivor set" in out.errors[0]


def test_a_symbol_inside_the_survivor_set_passes(limits):
    out = validate("a4", a4_payload(), limits, survivors=["NVDA261016C00185000"])
    assert out.ok


def test_an_over_long_hold_is_clamped(limits):
    """The schema caps at 5; config could cap lower."""
    out = validate("a4", a4_payload(expected_hold_sessions=5), limits)
    assert out.ok and out.decision.expected_hold_sessions == 5


def test_alternate_symbol_is_also_checked_against_survivors(limits):
    out = validate("a4", a4_payload(alternate_symbol="XXX261016C00100000"), limits,
                   survivors=["NVDA261016C00185000"])
    assert not out.ok
    assert "alternate_symbol" in out.errors[0]


# --- Agent 5: the monotone invariant ---------------------------------------


def test_widening_a_stop_is_forced_to_hold(limits):
    """Not clamped to a token tightening -- the model asked to widen."""
    out = validate("a5", {"action": "tighten_stop", "new_stop_pct": -45.0, "reason": "r"},
                   limits, current_stop_pct=-40.0)
    assert out.ok
    assert out.decision.action is ExitAction.HOLD
    assert out.decision.new_stop_pct is None
    [force] = out.forces
    assert force.rule == "monotone_stop_invariant"


def test_a_trivial_tightening_is_forced_to_hold(limits):
    out = validate("a5", {"action": "tighten_stop", "new_stop_pct": -39.5, "reason": "r"},
                   limits, current_stop_pct=-40.0)
    assert out.decision.action is ExitAction.HOLD
    [force] = out.forces
    assert force.rule == "agents.a5.min_stop_tightening_pct"


def test_a_real_tightening_survives(limits):
    out = validate("a5", {"action": "tighten_stop", "new_stop_pct": -25.0, "reason": "r"},
                   limits, current_stop_pct=-40.0)
    assert out.ok and out.decision.action is ExitAction.TIGHTEN_STOP
    assert out.decision.new_stop_pct == -25.0
    assert out.overrides == ()


def test_a_stop_beyond_the_floor_is_clamped_then_judged(limits):
    out = validate("a5", {"action": "tighten_stop", "new_stop_pct": -80.0, "reason": "r"},
                   limits, current_stop_pct=-90.0)
    assert out.ok
    [clamp] = out.clamps
    assert clamp.model_value == -80.0 and clamp.applied_value == -50.0


def test_tightening_without_a_known_current_stop_fails(limits):
    """Unknown current stop means tightening cannot be verified. Fail closed."""
    out = validate("a5", {"action": "tighten_stop", "new_stop_pct": -25.0, "reason": "r"},
                   limits)
    assert not out.ok
    assert "current_stop_pct" in out.errors[0]


def test_exit_now_needs_no_stop_context(limits):
    out = validate("a5", {"action": "exit_now", "reason": "stop breached"}, limits)
    assert out.ok and out.decision.action is ExitAction.EXIT_NOW


# --- Agent 6 ---------------------------------------------------------------


def test_surplus_observations_are_dropped(limits):
    payload = {"observations": [
        {"scope": "AMD", "text": f"o{i}", "expires_after_sessions": 3} for i in range(12)
    ]}
    out = validate("a6", payload, limits)
    assert out.ok and len(out.decision.observations) == 8      # max_observations
    assert any(o.rule == "agents.a6.max_observations" for o in out.clamps)


def test_surplus_global_observations_are_dropped(limits):
    payload = {"observations": [
        {"scope": "global", "text": f"g{i}", "expires_after_sessions": 3} for i in range(5)
    ]}
    out = validate("a6", payload, limits)
    scopes = [o.scope for o in out.decision.observations]
    assert scopes.count("global") == 3                        # max_global = 3


def test_an_over_long_expiry_is_clamped(limits):
    payload = {"observations": [
        {"scope": "AMD", "text": "o", "expires_after_sessions": 19}]}
    out = validate("a6", payload, limits)
    assert out.decision.observations[0].expires_after_sessions == 5
    assert any(o.rule == "agents.a6.max_expires_after_sessions" for o in out.clamps)


# --- generic ---------------------------------------------------------------


def test_an_unknown_agent_key_fails(limits):
    decision, error = parse("a9", {})
    assert decision is None and "unknown agent" in error
