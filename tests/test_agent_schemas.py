"""Agent contracts. The fail-closed paths matter more than the happy path.

Every test here is a thing a model might plausibly return. The question is
never "does the good case parse" but "does the bad case get rejected loudly".
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.agents.schemas import (
    AGENT_SCHEMAS,
    Bias,
    ContextDecision,
    ContractDecision,
    Direction,
    EventRisk,
    ExitAction,
    ExitDecision,
    IvAssessment,
    Observation,
    Regime,
    RegimeDecision,
    ReviewDecision,
    RiskDecision,
    SignalProfile,
    Structure,
)


def _profile(**kw):
    base = dict(ema_fast=9, confirmation_bars=2, require_vwap_alignment=True,
                min_atr_multiple=0.6, allowed_direction=Direction.LONG_CALLS)
    return SignalProfile(**{**base, **kw})


def _regime(**kw):
    base = dict(symbol="NVDA", regime=Regime.TRENDING_UP, confidence=0.7,
                signal_profile=_profile(), rationale="trend intact")
    return RegimeDecision(**{**base, **kw})


# --- structural invariants -------------------------------------------------


def test_widening_is_not_representable():
    """The monotone invariant is a property of the type, not a check."""
    values = {a.value for a in ExitAction}
    assert values == {"hold", "tighten_stop", "scale_out_half", "exit_now"}
    for forbidden in ("widen_stop", "add_to_position", "reverse", "double_down"):
        with pytest.raises(ValidationError):
            ExitDecision(action=forbidden, reason="x")


def test_size_multiplier_cannot_increase_exposure():
    for bad in (1.01, 2.0, 100.0):
        with pytest.raises(ValidationError):
            RiskDecision(size_multiplier=bad, reason="bigger")
    assert RiskDecision(size_multiplier=1.0, reason="full").size_multiplier == 1.0
    assert RiskDecision(size_multiplier=0.0, reason="veto").size_multiplier == 0.0


@pytest.mark.parametrize("key", sorted(AGENT_SCHEMAS))
def test_every_contract_forbids_unknown_fields(key):
    """A model that invents a field has not answered the question asked."""
    model = AGENT_SCHEMAS[key]
    with pytest.raises(ValidationError, match="[Ee]xtra"):
        model.model_validate({"definitely_not_a_field": 1})


@pytest.mark.parametrize("key", sorted(AGENT_SCHEMAS))
def test_contracts_are_immutable(key):
    """Nothing downstream may quietly rewrite a model's answer."""
    assert AGENT_SCHEMAS[key].model_config["frozen"] is True


# --- Agent 1 ---------------------------------------------------------------


def test_confidence_out_of_range_is_rejected():
    for bad in (-0.1, 1.1, 42.0):
        with pytest.raises(ValidationError):
            _regime(confidence=bad)


def test_invented_regime_is_rejected():
    with pytest.raises(ValidationError):
        _regime(regime="melt_up")


def test_rationale_over_length_is_rejected():
    with pytest.raises(ValidationError):
        _regime(rationale="x" * 201)


def test_negative_ema_span_is_rejected():
    with pytest.raises(ValidationError):
        _profile(ema_fast=0)


# --- Agent 2 ---------------------------------------------------------------


def _context(**kw):
    base = dict(symbol="AMD", eligible=True, hard_blocks=(), directional_bias=Bias.BULLISH,
                bias_strength=0.6, event_risk=EventRisk.LOW,
                iv_assessment=IvAssessment.FAIR, notes="")
    return ContextDecision(**{**base, **kw})


def test_a_named_blocker_beats_the_eligible_flag():
    """The contradiction resolves against the blocker, never the flag."""
    with pytest.raises(ValidationError, match="blocker always wins"):
        _context(eligible=True, hard_blocks=("earnings in 2 sessions",))
    assert _context(eligible=False, hard_blocks=("earnings",)).eligible is False


def test_notes_over_length_is_rejected():
    with pytest.raises(ValidationError):
        _context(notes="x" * 301)


# --- Agent 4 ---------------------------------------------------------------


def test_vertical_without_a_short_leg_is_rejected():
    with pytest.raises(ValidationError, match="requires a short_symbol"):
        ContractDecision(structure=Structure.DEBIT_VERTICAL,
                         primary_symbol="NVDA260904C00185000",
                         expected_hold_sessions=3, reason="spread")


def test_single_leg_carrying_a_short_leg_is_rejected():
    with pytest.raises(ValidationError, match="must not carry"):
        ContractDecision(structure=Structure.SINGLE_LEG,
                         primary_symbol="NVDA260904C00185000",
                         short_symbol="NVDA260904C00190000",
                         expected_hold_sessions=3, reason="oops")


def test_a_vertical_cannot_be_two_of_the_same_contract():
    with pytest.raises(ValidationError, match="must differ"):
        ContractDecision(structure=Structure.DEBIT_VERTICAL,
                         primary_symbol="NVDA260904C00185000",
                         short_symbol="NVDA260904C00185000",
                         expected_hold_sessions=3, reason="same leg twice")


def test_hold_sessions_outside_the_horizon_is_rejected():
    """Max hold is five sessions; a model asking for more is out of contract."""
    for bad in (0, 6, 30):
        with pytest.raises(ValidationError):
            ContractDecision(structure=Structure.SINGLE_LEG,
                             primary_symbol="X", expected_hold_sessions=bad, reason="r")


# --- Agent 5 ---------------------------------------------------------------


def test_a_positive_stop_is_not_a_stop():
    with pytest.raises(ValidationError):
        ExitDecision(action=ExitAction.TIGHTEN_STOP, new_stop_pct=10.0, reason="up")


def test_tighten_without_a_stop_is_rejected():
    with pytest.raises(ValidationError, match="requires new_stop_pct"):
        ExitDecision(action=ExitAction.TIGHTEN_STOP, reason="tighter")


def test_a_stop_on_a_non_tighten_action_is_rejected():
    """Otherwise hold could smuggle a stop change past a caller."""
    with pytest.raises(ValidationError, match="only meaningful"):
        ExitDecision(action=ExitAction.HOLD, new_stop_pct=-20.0, reason="hold but")


@pytest.mark.parametrize("current,proposed,expected", [
    (-40.0, -25.0, True),
    (-40.0, -40.0, False),
    (-25.0, -40.0, False),
])
def test_tightens_is_strict(current, proposed, expected):
    d = ExitDecision(action=ExitAction.TIGHTEN_STOP, new_stop_pct=proposed, reason="r")
    assert d.tightens(current) is expected


@pytest.mark.parametrize("action", [ExitAction.HOLD, ExitAction.EXIT_NOW,
                                    ExitAction.SCALE_OUT_HALF])
def test_non_tighten_actions_never_report_tightening(action):
    """A caller gating on tightens() cannot widen a stop by accident."""
    assert ExitDecision(action=action, reason="r").tightens(-40.0) is False


# --- Agent 6 ---------------------------------------------------------------


def test_observation_text_and_expiry_are_bounded():
    with pytest.raises(ValidationError):
        Observation(scope="AMD", text="x" * 201, expires_after_sessions=5)
    with pytest.raises(ValidationError):
        Observation(scope="AMD", text="ok", expires_after_sessions=0)


def test_empty_review_is_valid():
    assert ReviewDecision().observations == ()


# --- malformed payloads, as a model would actually produce them ------------


@pytest.mark.parametrize("payload", [
    {},
    {"symbol": "NVDA"},
    {"symbol": "NVDA", "regime": None, "confidence": 0.5,
     "signal_profile": None, "rationale": ""},
    {"symbol": "NVDA", "regime": "trending_up", "confidence": "high",
     "signal_profile": {}, "rationale": ""},
])
def test_malformed_regime_payloads_are_rejected(payload):
    with pytest.raises(ValidationError):
        RegimeDecision.model_validate(payload)


def test_numeric_strings_are_still_accepted():
    """Models emit JSON; a quoted number is a formatting quirk, not a lie."""
    d = RiskDecision.model_validate({"size_multiplier": "0.5", "reason": "half"})
    assert d.size_multiplier == 0.5


def test_a_quoted_out_of_range_number_is_still_rejected():
    with pytest.raises(ValidationError):
        RiskDecision.model_validate({"size_multiplier": "1.5", "reason": "no"})
