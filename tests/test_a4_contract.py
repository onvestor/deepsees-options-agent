"""Agent 4 against stubs. The fallback and the survivor-set boundary."""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date

import pytest

from src.agents.a4_contract import ContractInputs, ContractSelector
from src.agents.runner import AgentRunner
from src.agents.schemas import Structure

TEMPLATE = (
    "Select a contract for $symbol at $spot (atr $atr). Regime $regime "
    "confidence $confidence, bias $directional_bias $bias_strength, "
    "iv $iv_assessment. Expiry $target_expiry at $session_dte sessions, "
    "spans_earnings=$spans_earnings.\n"
    "$survivor_count survivors:\n$survivors\nObservations:\n$observations\n"
)


# --- stand-ins for prefilter Candidates ------------------------------------


@dataclass(frozen=True)
class FakeSpec:
    symbol: str
    strike: float
    expiry: date
    open_interest: int


@dataclass(frozen=True)
class FakeQuote:
    delta: float
    mid: float
    spread_pct_of_mid: float


@dataclass(frozen=True)
class FakeMetrics:
    pnl_to_spread_ratio: float


@dataclass(frozen=True)
class FakeCandidate:
    spec: FakeSpec
    quote: FakeQuote
    metrics: FakeMetrics
    expiry_type: str = "monthly"


def survivor(symbol, ratio=2.0, strike=760.0):
    return FakeCandidate(
        spec=FakeSpec(symbol, strike, date(2026, 10, 16), 900),
        quote=FakeQuote(0.62, 20.0, 0.012),
        metrics=FakeMetrics(ratio),
    )


def survivors(n=3):
    """Ranked best-first, as the prefilter hands them over."""
    return tuple(
        survivor(f"SPY261016C0076{i}000", ratio=3.0 - i * 0.5, strike=760.0 + i)
        for i in range(n)
    )


@pytest.fixture
def config(tmp_path, monkeypatch):
    from src.config import load_config

    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "a4_contract.txt").write_text(TEMPLATE, encoding="utf-8")
    monkeypatch.setenv("DEEPSEES_PROMPT_DIR", str(prompts))
    return load_config()


class RecordingLog:
    def __init__(self):
        self.records = []

    def write(self, payload, action, **kw):
        self.records.append({"payload": payload, "action": action, **kw})

    def of_action(self, action):
        return [r for r in self.records if r["action"] == action]

    def of_kind(self, kind):
        return [r for r in self.records if r["payload"].kind == kind]


@pytest.fixture
def selector(config):
    log = RecordingLog()
    runner = AgentRunner(config, decision_log=log)
    s = ContractSelector(config, runner)
    s.recording = log
    yield s
    runner.close()


def inputs(surv=None, **kw):
    base = dict(
        symbol="SPY", spot=766.0, atr=6.6, survivors=surv if surv is not None else survivors(),
        regime="trending_up", confidence=0.8, directional_bias="bullish",
        bias_strength=0.7, iv_assessment="fair", target_expiry="2026-10-16",
        session_dte=37,
    )
    base.update(kw)
    return ContractInputs(**base)


def pick(symbol, **kw):
    base = {"structure": "single_leg", "primary_symbol": symbol,
            "expected_hold_sessions": 3, "reason": "best delta"}
    base.update(kw)
    return base


# --- the model choosing ----------------------------------------------------


def test_a_valid_choice_is_honoured_and_reads_as_model(selector):
    target = survivors()[1].spec.symbol
    res = selector.select(inputs(), lambda p, f: pick(target))
    assert res.ok and res.source == "model"
    assert res.used_fallback is False
    assert res.decision.primary_symbol == target
    assert selector.recording.of_action("agent_fallback") == []


def test_choosing_the_top_ranked_survivor_is_still_a_model_choice(selector):
    """Same contract the fallback would pick -- but the model chose it."""
    top = survivors()[0].spec.symbol
    res = selector.select(inputs(), lambda p, f: pick(top))
    assert res.source == "model"
    assert res.decision.primary_symbol == top
    assert selector.recording.of_action("agent_fallback") == []


# --- the fallback reads differently ----------------------------------------


def test_a_failed_response_falls_back_to_the_best_ratio_survivor(selector):
    res = selector.select(inputs(), lambda p, f: "{not json")
    assert res.ok                                   # a working outcome
    assert res.source == "fallback"
    assert res.used_fallback is True
    assert res.decision.primary_symbol == survivors()[0].spec.symbol
    assert res.decision.structure is Structure.SINGLE_LEG


def test_the_fallback_emits_its_own_record(selector):
    """Otherwise there is no way to count how often the model is overridden."""
    selector.select(inputs(), lambda p, f: "{not json")
    [rec] = selector.recording.of_action("agent_fallback")
    payload = rec["payload"]
    assert payload.rule == "a4_deterministic_fallback"
    assert payload.model_value is None              # the model produced nothing usable
    assert payload.applied_value == survivors()[0].spec.symbol
    assert "not honoured" in payload.detail


def test_a_model_choice_emits_no_fallback_record(selector):
    selector.select(inputs(), lambda p, f: pick(survivors()[0].spec.symbol))
    assert selector.recording.of_action("agent_fallback") == []


def test_the_fallback_is_single_leg_not_a_vertical(selector):
    """A vertical is a judgment, and the fallback exists because none is available."""
    res = selector.select(inputs(), lambda p, f: "garbage")
    assert res.decision.structure is Structure.SINGLE_LEG
    assert res.decision.short_symbol is None


def test_the_fallback_carries_the_reason(selector):
    res = selector.select(inputs(), lambda p, f: "{not json")
    assert res.fallback_reason
    assert "JSON" in res.fallback_reason


@pytest.mark.parametrize("bad", ["{not json", "", "[1,2,3]"])
def test_every_malformed_shape_falls_back(selector, bad):
    res = selector.select(inputs(), lambda p, f, b=bad: b)
    assert res.source == "fallback" and res.ok


def test_a_timeout_falls_back(config):
    runner = AgentRunner(config, decision_log=RecordingLog())
    runner.timeout = 0.05
    s = ContractSelector(config, runner)

    def slow(prompt, feedback):
        time.sleep(0.5)
        return pick(survivors()[0].spec.symbol)

    res = s.select(inputs(), slow)
    assert res.source == "fallback"
    assert res.run.timed_out is True
    runner.close()


# --- the survivor set is a boundary, not a suggestion ----------------------


def test_a_symbol_outside_the_set_is_a_failure_not_a_fetch(selector):
    res = selector.select(inputs(), lambda p, f: pick("SPY261016C09999000"))
    assert res.source == "fallback"                 # rejected, then fallen back
    assert res.decision.primary_symbol == survivors()[0].spec.symbol
    assert "survivor set" in res.fallback_reason


def test_a_short_leg_outside_the_set_is_also_rejected(selector):
    res = selector.select(inputs(), lambda p, f: pick(
        survivors()[0].spec.symbol, structure="debit_vertical",
        short_symbol="SPY261016C09999000"))
    assert res.source == "fallback"
    assert "survivor set" in res.fallback_reason


def test_a_vertical_within_the_set_is_honoured(selector):
    surv = survivors()
    res = selector.select(inputs(), lambda p, f: pick(
        surv[0].spec.symbol, structure="debit_vertical",
        short_symbol=surv[1].spec.symbol))
    assert res.source == "model"
    assert res.decision.structure is Structure.DEBIT_VERTICAL


def test_the_model_only_sees_the_capped_set(selector):
    """Cap is 12. The model is held to what it was shown, not the full set."""
    many = survivors(20)
    seen = {}

    def capture(prompt, feedback):
        seen["p"] = prompt
        return pick(many[0].spec.symbol)

    res = selector.select(inputs(surv=many), capture)
    assert res.survivors_total == 20
    assert len(res.offered) == 12
    assert many[15].spec.symbol not in seen["p"]     # beyond the cap, unseen


def test_a_symbol_beyond_the_cap_is_rejected(selector):
    """Even a real survivor is invalid if the model was never shown it."""
    many = survivors(20)
    res = selector.select(inputs(surv=many), lambda p, f: pick(many[15].spec.symbol))
    assert res.source == "fallback"
    assert "survivor set" in res.fallback_reason


def test_the_full_survivor_count_is_retained(selector):
    many = survivors(20)
    res = selector.select(inputs(surv=many), lambda p, f: pick(many[0].spec.symbol))
    assert res.survivors_total == 20                # nothing lost to analysis
    assert len(res.offered) == 12


# --- nothing to choose from ------------------------------------------------


def test_an_empty_survivor_set_selects_nothing(selector):
    """Not a failure and not a fallback -- there is no contract to choose."""
    res = selector.select(inputs(surv=()), lambda p, f: pick("X"))
    assert res.decision is None
    assert res.source == "none"
    assert res.ok is False


def test_an_empty_survivor_set_calls_no_model(selector):
    calls = []
    selector.select(inputs(surv=()), lambda p, f: calls.append(1))
    assert calls == []


# --- clamping still applies ------------------------------------------------


def test_an_over_long_hold_is_clamped_not_failed(selector):
    top = survivors()[0].spec.symbol
    res = selector.select(inputs(), lambda p, f: pick(top, expected_hold_sessions=5))
    assert res.source == "model"
    assert res.decision.expected_hold_sessions == 5


def test_a_hold_outside_the_schema_falls_back(selector):
    top = survivors()[0].spec.symbol
    res = selector.select(inputs(), lambda p, f: pick(top, expected_hold_sessions=99))
    assert res.source == "fallback"


def test_the_fallback_hold_comes_from_config(selector):
    res = selector.select(inputs(), lambda p, f: "garbage")
    assert res.decision.expected_hold_sessions == 3


# --- prompt handling --------------------------------------------------------


def test_the_survivor_table_reaches_the_prompt(selector):
    seen = {}

    def capture(prompt, feedback):
        seen["p"] = prompt
        return pick(survivors()[0].spec.symbol)

    selector.select(inputs(), capture)
    assert survivors()[0].spec.symbol in seen["p"]
    assert "pnl_to_spread" in seen["p"]
    assert "$survivors" not in seen["p"]


def test_the_prompt_never_reaches_the_log(selector):
    selector.select(inputs(), lambda p, f: pick(survivors()[0].spec.symbol))
    for rec in selector.recording.of_kind("agent_call"):
        assert "Select a contract" not in rec["payload"].model_dump_json()
