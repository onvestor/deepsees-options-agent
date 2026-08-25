"""Agent 1 against stub responses. No prompts, no provider, no network."""
from __future__ import annotations

import time
from datetime import date

import pytest

from src.agents.a1_regime import RegimeInputs, RegimeProfiler
from src.agents.runner import AgentRunner
from src.agents.schemas import Direction
from src.config import ConfigError

SESSION = date(2026, 8, 25)

GOOD = {
    "symbol": "NVDA", "regime": "trending_up", "confidence": 0.9,
    "signal_profile": {"ema_fast": 9, "confirmation_bars": 2,
                       "require_vwap_alignment": True, "min_atr_multiple": 0.6,
                       "allowed_direction": "long_calls"},
    "rationale": "higher highs, ATR expanding",
}

TEMPLATE = (
    "Classify the regime for $symbol at $spot. ATR $atr ($atr_pct_of_spot of spot), "
    "RSI $rsi, realized vol $realized_vol, 20d trend $trend_pct_20d, "
    "ema $ema_fast_value/$ema_slow_value, above_vwap=$above_vwap.\n"
    "Prior observations:\n$observations\n"
)


@pytest.fixture
def config(tmp_path, monkeypatch):
    from src.config import load_config

    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "a1_regime.txt").write_text(TEMPLATE, encoding="utf-8")
    monkeypatch.setenv("DEEPSEES_PROMPT_DIR", str(prompts))
    return load_config()


@pytest.fixture
def profiler(config):
    runner = AgentRunner(config)
    yield RegimeProfiler(config, runner)
    runner.close()


def inputs(**kw):
    base = dict(symbol="NVDA", spot=209.6, atr=6.4, atr_pct_of_spot=0.031,
                realized_vol=0.42, rsi=61.2, ema_fast_value=207.1,
                ema_slow_value=201.4, trend_pct_20d=0.084, above_vwap=True)
    base.update(kw)
    return RegimeInputs(**base)


def stub(response):
    return lambda prompt, feedback: response


# --- happy path ------------------------------------------------------------


def test_a_valid_response_becomes_a_decision(profiler):
    res = profiler.profile(inputs(), SESSION, stub(GOOD))
    assert res.ok and res.cached is False
    assert res.decision.regime.value == "trending_up"
    assert res.decision.signal_profile.ema_fast == 9


def test_the_prompt_is_rendered_with_the_computed_inputs(profiler):
    seen = {}

    def capture(prompt, feedback):
        seen["prompt"] = prompt
        return GOOD

    profiler.profile(inputs(), SESSION, capture)
    assert "NVDA" in seen["prompt"]
    assert "209.60" in seen["prompt"]        # spot, pre-formatted
    assert "$symbol" not in seen["prompt"]   # every placeholder substituted


def test_observations_are_rendered_and_absent_reads_as_none(profiler):
    seen = {}

    def capture(prompt, feedback):
        seen["p"] = prompt
        return GOOD

    profiler.profile(inputs(observations=("AMD gapped twice",)), SESSION, capture)
    assert "- AMD gapped twice" in seen["p"]

    profiler.clear()
    profiler.profile(inputs(), SESSION, capture)
    assert "(none)" in seen["p"]


# --- fail closed: this is the ENTRY path -----------------------------------


def test_malformed_output_blocks_the_trade(profiler):
    res = profiler.profile(inputs(), SESSION, stub("{not json"))
    assert not res.ok
    assert res.decision is None
    assert res.blocks_action is True       # entry path: no profile, no trade


def test_valid_json_failing_the_schema_blocks(profiler):
    res = profiler.profile(inputs(), SESSION, stub({"symbol": "NVDA", "regime": "moon"}))
    assert not res.ok and res.blocks_action is True


def test_an_empty_response_blocks(profiler):
    res = profiler.profile(inputs(), SESSION, stub(""))
    assert not res.ok and res.blocks_action is True


def test_a_timeout_blocks(config):
    runner = AgentRunner(config)
    runner.timeout = 0.05
    p = RegimeProfiler(config, runner)

    def slow(prompt, feedback):
        time.sleep(0.5)
        return GOOD

    res = p.profile(inputs(), SESSION, slow)
    assert res.run.timed_out is True
    assert not res.ok and res.blocks_action is True
    runner.close()


def test_there_is_no_default_profile(profiler):
    """A default here would be a guess about market regime."""
    res = profiler.profile(inputs(), SESSION, stub("garbage"))
    assert res.decision is None


# --- out-of-range values are clamped, not rejected -------------------------


def test_an_out_of_range_ema_is_clamped_and_still_usable(profiler):
    payload = {**GOOD, "signal_profile": {**GOOD["signal_profile"], "ema_fast": 7}}
    res = profiler.profile(inputs(), SESSION, stub(payload))
    assert res.ok
    assert res.decision.signal_profile.ema_fast == 8
    assert res.run.outcome.status == "clamped"


def test_low_confidence_forces_direction_none(profiler):
    res = profiler.profile(inputs(), SESSION, stub({**GOOD, "confidence": 0.1}))
    assert res.ok
    assert res.decision.signal_profile.allowed_direction is Direction.NONE
    assert res.run.outcome.status == "ok"      # forced, not a model error


def test_a_choppy_regime_forces_direction_none(profiler):
    res = profiler.profile(inputs(), SESSION, stub({**GOOD, "regime": "choppy"}))
    assert res.decision.signal_profile.allowed_direction is Direction.NONE


def test_confidence_outside_zero_to_one_is_a_failure_not_a_clamp(profiler):
    """Out of the schema's range entirely -- the model answered badly."""
    res = profiler.profile(inputs(), SESSION, stub({**GOOD, "confidence": 5.0}))
    assert not res.ok and res.blocks_action is True


# --- the session lock ------------------------------------------------------


def test_the_profile_is_locked_for_the_session(profiler):
    calls = []

    def counting(prompt, feedback):
        calls.append(1)
        return GOOD

    first = profiler.profile(inputs(), SESSION, counting)
    second = profiler.profile(inputs(), SESSION, counting)
    assert len(calls) == 1                 # no second model call
    assert second.cached is True
    assert second.decision == first.decision


def test_the_lock_stores_the_accepted_decision_not_the_raw_output(profiler):
    """Otherwise a clamped-away value re-enters through the cache."""
    payload = {**GOOD, "signal_profile": {**GOOD["signal_profile"], "ema_fast": 7}}
    profiler.profile(inputs(), SESSION, stub(payload))
    cached = profiler.profile(inputs(), SESSION, stub(payload))
    assert cached.cached is True
    assert cached.decision.signal_profile.ema_fast == 8      # not 7


def test_a_failed_call_does_not_poison_the_session(profiler):
    """A failure must leave the next cycle free to try again."""
    bad = profiler.profile(inputs(), SESSION, stub("garbage"))
    assert not bad.ok
    good = profiler.profile(inputs(), SESSION, stub(GOOD))
    assert good.ok and good.cached is False


def test_the_lock_is_per_symbol_and_per_session(profiler):
    calls = []

    def counting(prompt, feedback):
        calls.append(1)
        return GOOD

    profiler.profile(inputs(symbol="NVDA"), SESSION, counting)
    profiler.profile(inputs(symbol="SPY"), SESSION, counting)
    profiler.profile(inputs(symbol="NVDA"), date(2026, 8, 26), counting)
    assert len(calls) == 3


# --- prompt loading is operator-supplied and fails closed ------------------


def test_a_missing_prompt_names_the_file(config):
    runner = AgentRunner(config)
    p = RegimeProfiler(config, runner, prompt_name="does_not_exist.txt")
    with pytest.raises(ConfigError, match="does_not_exist.txt"):
        p.profile(inputs(), SESSION, stub(GOOD))
    runner.close()


def test_a_template_referencing_an_unknown_field_fails_loudly(config, tmp_path):
    bad = tmp_path / "prompts" / "bad.txt"
    bad.write_text("Regime for $symbol with $not_a_field", encoding="utf-8")
    runner = AgentRunner(config)
    p = RegimeProfiler(config, runner, prompt_name="bad.txt")
    with pytest.raises(ConfigError, match="not_a_field"):
        p.profile(inputs(), SESSION, stub(GOOD))
    runner.close()


def test_the_prompt_text_never_reaches_the_log(config):
    class Recording:
        def __init__(self):
            self.records = []

        def write(self, payload, action, **kw):
            self.records.append(payload)

    log = Recording()
    runner = AgentRunner(config, decision_log=log)
    p = RegimeProfiler(config, runner)
    p.profile(inputs(), SESSION, stub(GOOD))
    calls = [r for r in log.records if r.kind == "agent_call"]
    assert calls
    for record in calls:
        assert "Classify the regime" not in record.model_dump_json()
        assert record.prompt_hash
    runner.close()
