"""Agent 3 against stubs. The model may only shrink; caps land after it."""
from __future__ import annotations

import time

import pytest

from src.agents.a3_risk import RiskAllocator, RiskInputs
from src.agents.runner import AgentRunner
from src.risk.sizing import AccountState

TEMPLATE = (
    "Size $symbol / $contract_symbol. Base $base_contracts contracts at "
    "$cost_per_contract each, risk $max_risk_per_contract, budget $risk_budget, "
    "equity $equity. Open $open_positions positions, $open_premium premium. "
    "Regime $regime conf $confidence bias $bias_strength iv $iv_assessment "
    "spans_earnings=$spans_earnings.\nObservations:\n$observations\n"
)


@pytest.fixture
def config(tmp_path, monkeypatch):
    from src.config import load_config

    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "a3_risk.txt").write_text(TEMPLATE, encoding="utf-8")
    monkeypatch.setenv("DEEPSEES_PROMPT_DIR", str(prompts))
    return load_config()


class RecordingLog:
    def __init__(self):
        self.records = []

    def write(self, payload, action, **kw):
        self.records.append({"payload": payload, "action": action, **kw})

    def of_kind(self, kind):
        return [r for r in self.records if r["payload"].kind == kind]


@pytest.fixture
def allocator(config):
    log = RecordingLog()
    runner = AgentRunner(config, decision_log=log)
    a = RiskAllocator(config, runner)
    a.recording = log
    yield a
    runner.close()


ACCOUNT = AccountState(equity=100_000.0, options_buying_power=100_000.0)


def inputs(**kw):
    base = dict(symbol="SPY", contract_symbol="SPY261016C00760000",
                base_contracts=3, cost_per_contract=200.0,
                max_risk_per_contract=200.0, risk_budget=1000.0,
                equity=100_000.0, open_positions=0, open_premium=0.0,
                regime="trending_up", confidence=0.8, bias_strength=0.7,
                iv_assessment="fair")
    base.update(kw)
    return RiskInputs(**base)


def mult(value, reason="ok"):
    return {"size_multiplier": value, "reason": reason}


# --- the model can only shrink ---------------------------------------------


def test_a_multiplier_of_one_leaves_the_base_size(allocator):
    base = allocator.base_size(200.0, 200.0, ACCOUNT).final_contracts
    res = allocator.allocate(inputs(), ACCOUNT, lambda p, f: mult(1.0))
    assert res.ok and res.contracts == base


def test_a_multiplier_shrinks(allocator):
    base = allocator.base_size(200.0, 200.0, ACCOUNT).final_contracts
    res = allocator.allocate(inputs(), ACCOUNT, lambda p, f: mult(0.5))
    assert res.contracts < base
    assert res.contracts == int(base * 0.5)


def test_a_multiplier_above_one_is_a_schema_failure(allocator):
    """Not clamped to 1.0 -- out of the type's range entirely."""
    res = allocator.allocate(inputs(), ACCOUNT, lambda p, f: mult(2.0))
    assert not res.ok and res.blocked is True


def test_a_tiny_multiplier_becomes_a_veto(allocator):
    res = allocator.allocate(inputs(), ACCOUNT, lambda p, f: mult(0.1))
    assert res.ok
    assert res.multiplier == 0.0             # forced by veto_below_multiplier
    assert res.contracts == 0
    assert res.vetoed_by_model is True


def test_an_explicit_zero_is_a_veto_not_a_failure(allocator):
    res = allocator.allocate(inputs(), ACCOUNT, lambda p, f: mult(0.0, "too rich"))
    assert res.ok                            # the model answered
    assert res.contracts == 0
    assert res.vetoed_by_model is True


def test_no_multiplier_can_exceed_the_base(allocator):
    """The monotone invariant, swept."""
    base = allocator.base_size(200.0, 200.0, ACCOUNT).final_contracts
    for value in (0.0, 0.25, 0.3, 0.5, 0.75, 0.99, 1.0):
        res = allocator.allocate(inputs(), ACCOUNT, lambda p, f, v=value: mult(v))
        assert res.contracts <= base


# --- caps run after the model ----------------------------------------------


def test_caps_still_bind_after_a_full_multiplier(allocator):
    """A 1.0 multiplier cannot restore what a cap removes."""
    expensive = inputs(cost_per_contract=2_600.0, max_risk_per_contract=2_600.0)
    res = allocator.allocate(expensive, ACCOUNT, lambda p, f: mult(1.0))
    assert res.contracts == 0


def test_a_trade_capped_to_zero_calls_no_model(allocator):
    """Asking whether to make zero smaller has no reachable outcome."""
    calls = []
    expensive = inputs(cost_per_contract=50_000.0, max_risk_per_contract=50_000.0)
    res = allocator.allocate(expensive, ACCOUNT, lambda p, f: calls.append(1))
    assert calls == []
    assert res.blocked is True and res.contracts == 0


def test_the_sizing_record_keeps_every_stage(allocator):
    res = allocator.allocate(inputs(), ACCOUNT, lambda p, f: mult(0.5))
    assert res.sizing.base_contracts >= res.sizing.final_contracts
    assert res.sizing.model_multiplier == 0.5
    assert res.sizing.caps                   # every cap's verdict retained


# --- failure is a skip, not a default size ---------------------------------


def test_malformed_output_blocks_rather_than_defaulting(allocator):
    res = allocator.allocate(inputs(), ACCOUNT, lambda p, f: "{not json")
    assert not res.ok and res.blocked is True
    assert res.multiplier is None            # neither 1.0 nor 0.0
    assert res.vetoed_by_model is False      # a skip, not a model veto


def test_a_timeout_blocks_rather_than_sizing_full(config):
    """Silence must not read as approval."""
    runner = AgentRunner(config, decision_log=RecordingLog())
    runner.timeout = 0.05
    a = RiskAllocator(config, runner)

    def slow(prompt, feedback):
        time.sleep(0.5)
        return mult(1.0)

    res = a.allocate(inputs(), ACCOUNT, slow)
    assert res.blocked is True
    assert res.multiplier is None
    assert res.run.timed_out is True
    runner.close()


def test_an_empty_response_blocks(allocator):
    res = allocator.allocate(inputs(), ACCOUNT, lambda p, f: "")
    assert res.blocked is True and res.multiplier is None


def test_a_failed_call_is_distinguishable_from_a_veto(allocator):
    """Both size zero. Only one is a decision the model made."""
    failed = allocator.allocate(inputs(), ACCOUNT, lambda p, f: "garbage")
    vetoed = allocator.allocate(inputs(), ACCOUNT, lambda p, f: mult(0.0))
    assert failed.contracts == vetoed.contracts == 0
    assert failed.vetoed_by_model is False and vetoed.vetoed_by_model is True
    assert failed.blocked is True and vetoed.blocked is False


def test_the_base_size_survives_a_failure_for_the_log(allocator):
    """What the code decided is still recorded even when the model failed."""
    res = allocator.allocate(inputs(), ACCOUNT, lambda p, f: "garbage")
    assert res.sizing is not None
    assert res.sizing.final_contracts > 0    # the pre-model size


# --- prompt handling --------------------------------------------------------


def test_the_deterministic_size_reaches_the_prompt(allocator):
    seen = {}

    def capture(prompt, feedback):
        seen["p"] = prompt
        return mult(1.0)

    allocator.allocate(inputs(base_contracts=3), ACCOUNT, capture)
    assert "Base 3 contracts" in seen["p"]
    assert "$base_contracts" not in seen["p"]


def test_the_prompt_never_reaches_the_log(allocator):
    allocator.allocate(inputs(), ACCOUNT, lambda p, f: mult(1.0))
    for rec in allocator.recording.of_kind("agent_call"):
        assert "Size SPY" not in rec["payload"].model_dump_json()
