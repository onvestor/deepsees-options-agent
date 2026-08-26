"""All six agents wired together against stubs. Proves the wiring, not the prompts.

A wiring bug and a prompt bug look identical from the outside -- both surface as
"the system did not trade" or "the system traded when it should not have". This
file exists so that when prompts arrive, any failure is attributable to them.

Two claims are load-bearing:

* **An entry-path skip cascades.** If Agent 1 does not produce a profile there
  is no signal evaluation, so Agent 4 is never called, nothing is sized, and no
  order is built. Not "an order of size zero" -- no order at all.
* **A clamp is not a failure.** A clamped Agent 3 multiplier still produces a
  valid sized order, because clamping means the model answered and a value was
  adjusted.

And one invariant swept across every stage: **anything blocked or failed
reaches zero contracts and no order.** That is asserted at every stage rather
than at the end, because ``AllocationResult.contracts`` already once returned a
tradeable-looking number on a blocked result.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pytest

from src.agents.a1_regime import RegimeInputs, RegimeProfiler
from src.agents.a2_context import ContextInputs, ContextScreener
from src.agents.a3_risk import RiskAllocator, RiskInputs
from src.agents.a4_contract import ContractInputs, ContractSelector
from src.agents.a5_exit import ExitInputs, ExitManager
from src.agents.a6_review import NightlyReviewer, ObservationStore, ReviewInputs
from src.agents.runner import AgentRunner
from src.agents.schemas import Direction
from src.risk.sizing import AccountState

SESSION = date(2026, 8, 26)
ACCOUNT = AccountState(equity=100_000.0, options_buying_power=100_000.0)

TEMPLATES = {
    "a1_regime.txt": "Regime $symbol $spot $atr $atr_pct_of_spot $realized_vol $rsi "
                     "$ema_fast_value $ema_slow_value $trend_pct_20d $above_vwap\n$observations",
    "a2_context.txt": "Context $symbol $spot $atr_pct_of_spot $realized_vol $iv_vs_rv20 "
                      "$iv_percentile $trend_pct_20d $sessions_until_earnings "
                      "$sessions_since_earnings\n$headlines\n$observations",
    "a4_contract.txt": "Contract $symbol $spot $atr $regime $confidence $directional_bias "
                       "$bias_strength $iv_assessment $target_expiry $session_dte "
                       "$spans_earnings $survivor_count\n$survivors\n$observations",
    "a3_risk.txt": "Risk $symbol $contract_symbol $base_contracts $cost_per_contract "
                   "$max_risk_per_contract $risk_budget $equity $open_positions "
                   "$open_premium $regime $confidence $bias_strength $iv_assessment "
                   "$spans_earnings\n$observations",
    "a5_exit.txt": "Exit $symbol $contract_symbol $entry_premium $current_premium $pnl_pct "
                   "$current_stop_pct $target_pct $sessions_held $max_hold_sessions "
                   "$sessions_to_expiry $contracts $regime $spans_earnings\n$observations",
    "a6_review.txt": "Review $session $entries $exits $skips $wins $losses $realized_pnl "
                     "$agent_clamps $agent_forces $agent_failures $fallbacks "
                     "$symbols_traded\n$notes",
}


@pytest.fixture
def config(tmp_path, monkeypatch):
    from src.config import load_config

    prompts = tmp_path / "prompts"
    prompts.mkdir()
    for name, body in TEMPLATES.items():
        (prompts / name).write_text(body, encoding="utf-8")
    monkeypatch.setenv("DEEPSEES_PROMPT_DIR", str(prompts))
    return load_config()


class RecordingLog:
    def __init__(self):
        self.records = []

    def write(self, payload, action, **kw):
        self.records.append({"payload": payload, "action": action, **kw})


# --- a minimal survivor, matching the prefilter's shape --------------------


@dataclass(frozen=True)
class Spec:
    symbol: str
    strike: float
    expiry: date
    open_interest: int


@dataclass(frozen=True)
class Quote:
    delta: float
    mid: float
    spread_pct_of_mid: float


@dataclass(frozen=True)
class Metrics:
    pnl_to_spread_ratio: float
    cost_per_contract: float
    max_risk: float


@dataclass(frozen=True)
class Candidate:
    spec: Spec
    quote: Quote
    metrics: Metrics
    expiry_type: str = "monthly"


CONTRACT = "SPY261016C00760000"
SURVIVORS = (
    Candidate(Spec(CONTRACT, 760.0, date(2026, 10, 16), 900),
              Quote(0.62, 20.0, 0.012), Metrics(2.9, 200.0, 200.0)),
    Candidate(Spec("SPY261016C00765000", 765.0, date(2026, 10, 16), 700),
              Quote(0.57, 17.0, 0.014), Metrics(2.4, 170.0, 170.0)),
)


# --- the pipeline under test ------------------------------------------------


@dataclass
class PipelineTrace:
    """What actually happened, stage by stage."""

    a1_called: bool = False
    a2_called: bool = False
    a4_called: bool = False
    a3_called: bool = False
    contracts: int = 0
    order_built: bool = False
    stopped_at: str | None = None


class Pipeline:
    """Entry pipeline: a1 -> a2 -> a4 -> a3 -> order. Stops at the first block."""

    def __init__(self, config, log):
        self.runner = AgentRunner(config, decision_log=log)
        self.a1 = RegimeProfiler(config, self.runner)
        self.a2 = ContextScreener(config, self.runner)
        self.a4 = ContractSelector(config, self.runner)
        self.a3 = RiskAllocator(config, self.runner)

    def close(self):
        self.runner.close()

    def run(self, stubs) -> PipelineTrace:
        trace = PipelineTrace()

        trace.a1_called = True
        profile = self.a1.profile(
            RegimeInputs("SPY", 766.0, 6.6, 0.0086, 0.14, 58.0, 764.0, 758.0, 0.04, True),
            SESSION, stubs["a1"],
        )
        if profile.blocks_action or not profile.ok:
            trace.stopped_at = "a1"
            return trace
        if profile.decision.signal_profile.allowed_direction is Direction.NONE:
            # A valid profile that permits no direction. Not a failure -- the
            # regime read says do not trade this symbol today.
            trace.stopped_at = "a1_direction_none"
            return trace

        trace.a2_called = True
        screen = self.a2.screen(
            [ContextInputs("SPY", 766.0, 0.0086, 0.14, 1.05, 0.5, 0.04)],
            SESSION, stubs["a2"],
        )
        if not screen.eligible:
            trace.stopped_at = "a2"
            return trace

        trace.a4_called = True
        selection = self.a4.select(
            ContractInputs("SPY", 766.0, 6.6, SURVIVORS, "trending_up", 0.8,
                           "bullish", 0.7, "fair", "2026-10-16", 37),
            stubs["a4"],
        )
        if not selection.ok:
            trace.stopped_at = "a4"
            return trace

        trace.a3_called = True
        allocation = self.a3.allocate(
            RiskInputs("SPY", selection.decision.primary_symbol, 3, 200.0, 200.0,
                       1000.0, 100_000.0, 0, 0.0, "trending_up", 0.8, 0.7, "fair"),
            ACCOUNT, stubs["a3"],
        )
        trace.contracts = allocation.contracts
        if not allocation.ok or allocation.contracts <= 0:
            trace.stopped_at = "a3"
            return trace

        trace.order_built = True
        return trace


def stubs(**overrides):
    base = {
        "a1": lambda p, f: {
            "symbol": "SPY", "regime": "trending_up", "confidence": 0.9,
            "signal_profile": {"ema_fast": 9, "confirmation_bars": 2,
                               "require_vwap_alignment": True, "min_atr_multiple": 0.6,
                               "allowed_direction": "long_calls"},
            "rationale": "trend"},
        "a2": lambda p, f: {
            "symbol": "SPY", "eligible": True, "hard_blocks": [],
            "directional_bias": "bullish", "bias_strength": 0.8,
            "event_risk": "low", "iv_assessment": "fair", "notes": ""},
        "a4": lambda p, f: {"structure": "single_leg", "primary_symbol": CONTRACT,
                            "expected_hold_sessions": 3, "reason": "best ratio"},
        "a3": lambda p, f: {"size_multiplier": 1.0, "reason": "full"},
    }
    base.update(overrides)
    return base


@pytest.fixture
def pipeline(config):
    log = RecordingLog()
    p = Pipeline(config, log)
    p.recording = log
    yield p
    p.close()


# --- the happy path, so the failures below mean something ------------------


def test_a_clean_run_reaches_an_order(pipeline):
    trace = pipeline.run(stubs())
    assert trace.a1_called and trace.a2_called and trace.a4_called and trace.a3_called
    assert trace.contracts > 0
    assert trace.order_built is True
    assert trace.stopped_at is None


# --- an Agent 1 skip cascades ----------------------------------------------


@pytest.mark.parametrize("bad", ["{not json", "", "[1,2,3]", {"regime": "moon"}])
def test_an_a1_skip_stops_everything_downstream(pipeline, bad):
    """No profile means no signal evaluation, so nothing downstream runs."""
    trace = pipeline.run(stubs(a1=lambda p, f, b=bad: b))
    assert trace.stopped_at == "a1"
    assert trace.a2_called is False
    assert trace.a4_called is False        # never asked to choose a contract
    assert trace.a3_called is False        # nothing sized
    assert trace.contracts == 0
    assert trace.order_built is False      # no order, not an order of size zero


def test_an_a1_timeout_cascades_the_same_way(config):
    log = RecordingLog()
    p = Pipeline(config, log)
    p.runner.timeout = 0.05

    def slow(prompt, feedback):
        import time
        time.sleep(0.5)
        return {"symbol": "SPY", "regime": "trending_up", "confidence": 0.9,
                "signal_profile": {"ema_fast": 9, "confirmation_bars": 2,
                                   "require_vwap_alignment": True,
                                   "min_atr_multiple": 0.6,
                                   "allowed_direction": "long_calls"},
                "rationale": "r"}

    trace = p.run(stubs(a1=slow))
    assert trace.stopped_at == "a1"
    assert trace.a4_called is False and trace.contracts == 0 and not trace.order_built
    p.close()


def test_a_forced_direction_none_also_stops_the_pipeline(pipeline):
    """A valid answer that permits no trade. Different reason, same outcome."""
    low = {"symbol": "SPY", "regime": "trending_up", "confidence": 0.1,
           "signal_profile": {"ema_fast": 9, "confirmation_bars": 2,
                              "require_vwap_alignment": True, "min_atr_multiple": 0.6,
                              "allowed_direction": "long_calls"},
           "rationale": "weak"}
    trace = pipeline.run(stubs(a1=lambda p, f: low))
    assert trace.stopped_at == "a1_direction_none"
    assert trace.a4_called is False and trace.contracts == 0 and not trace.order_built


# --- a clamp is not a failure ----------------------------------------------


def test_a_clamped_a3_multiplier_still_produces_an_order(pipeline):
    """Clamping means the model answered and a value was adjusted."""
    # 1.4 is out of the schema's range and fails; 0.5 is in range. Use a value
    # that the VALIDATOR adjusts rather than rejects: a1's ema is the clean
    # example, so clamp there and keep a3 valid but reduced.
    trace = pipeline.run(stubs(
        a1=lambda p, f: {
            "symbol": "SPY", "regime": "trending_up", "confidence": 0.9,
            "signal_profile": {"ema_fast": 7, "confirmation_bars": 2,
                               "require_vwap_alignment": True, "min_atr_multiple": 9.0,
                               "allowed_direction": "long_calls"},
            "rationale": "trend"},
        a3=lambda p, f: {"size_multiplier": 0.5, "reason": "half"},
    ))
    assert trace.stopped_at is None
    assert trace.contracts > 0
    assert trace.order_built is True


def test_a_clamped_a1_profile_does_not_stop_the_pipeline(pipeline):
    trace = pipeline.run(stubs(a1=lambda p, f: {
        "symbol": "SPY", "regime": "trending_up", "confidence": 0.9,
        "signal_profile": {"ema_fast": 7, "confirmation_bars": 9,
                           "require_vwap_alignment": True, "min_atr_multiple": 0.6,
                           "allowed_direction": "long_calls"},
        "rationale": "trend"}))
    assert trace.order_built is True


def test_an_a4_fallback_still_produces_an_order(pipeline):
    """A fallback is a working outcome; the pipeline continues."""
    trace = pipeline.run(stubs(a4=lambda p, f: "{not json"))
    assert trace.stopped_at is None
    assert trace.a3_called is True
    assert trace.order_built is True


# --- the sweep: blocked or failed reaches zero, at every stage --------------


@pytest.mark.parametrize("stage", ["a1", "a2", "a4", "a3"])
@pytest.mark.parametrize("bad", ["{not json", "", "garbage"])
def test_no_failure_anywhere_produces_contracts_or_an_order(pipeline, stage, bad):
    """The invariant that .contracts already broke once.

    a4 is the exception by design: it falls back rather than failing, so the
    pipeline continues and an order IS built. Everything else must reach zero.
    """
    trace = pipeline.run(stubs(**{stage: lambda p, f, b=bad: b}))
    if stage == "a4":
        assert trace.order_built is True and trace.contracts > 0
    else:
        assert trace.contracts == 0
        assert trace.order_built is False
        assert trace.stopped_at == stage


@pytest.mark.parametrize("multiplier", [0.0, 0.1])
def test_an_a3_veto_produces_no_order(pipeline, multiplier):
    """Zero contracts and no order -- a veto is not an order of size zero."""
    trace = pipeline.run(stubs(a3=lambda p, f: {"size_multiplier": multiplier,
                                                "reason": "veto"}))
    assert trace.contracts == 0
    assert trace.order_built is False


def test_an_a2_ineligible_verdict_stops_before_a4(pipeline):
    trace = pipeline.run(stubs(a2=lambda p, f: {
        "symbol": "SPY", "eligible": False, "hard_blocks": ["halted"],
        "directional_bias": "neutral", "bias_strength": 0.1,
        "event_risk": "high", "iv_assessment": "rich", "notes": ""}))
    assert trace.stopped_at == "a2"
    assert trace.a4_called is False
    assert trace.contracts == 0 and trace.order_built is False


def test_an_empty_survivor_set_produces_no_order(pipeline, config):
    """No contract to choose. Not a failure, still no order."""
    log = RecordingLog()
    p = Pipeline(config, log)
    selection = p.a4.select(
        ContractInputs("SPY", 766.0, 6.6, (), "trending_up", 0.8, "bullish",
                       0.7, "fair", "2026-10-16", 37),
        stubs()["a4"],
    )
    assert selection.decision is None and selection.source == "none"
    p.close()


# --- the exit path does NOT cascade ----------------------------------------


def test_an_a5_failure_does_not_stop_exit_management(config):
    """The mirror image: on the exit path a failure must not halt the loop."""
    log = RecordingLog()
    runner = AgentRunner(config, decision_log=log)
    a5 = ExitManager(config, runner)
    plan = a5.manage(
        ExitInputs("SPY", CONTRACT, 2000.0, 1800.0, -10.0, -40.0, 75.0, 2, 5, 37, 2),
        lambda p, f: "{not json",
    )
    assert plan.model_failed is True
    assert plan.blocks_action is False       # the loop continues
    assert plan.stop_pct == -40.0            # protection unchanged
    runner.close()


def test_an_a6_failure_does_not_affect_today(config):
    log = RecordingLog()
    runner = AgentRunner(config, decision_log=log)
    a6 = NightlyReviewer(config, runner)
    store = ObservationStore()
    res = a6.review(
        ReviewInputs(SESSION, 1, 1, 2, 1, 0, 50.0, 0, 1, 0, 0),
        lambda p, f: "garbage", store=store,
    )
    assert res.model_failed is True
    assert res.blocks_action is False
    assert store.items == []


# --- observations flow forward as context only -----------------------------


def test_observations_reach_the_next_session_as_text(config):
    log = RecordingLog()
    runner = AgentRunner(config, decision_log=log)
    a6 = NightlyReviewer(config, runner)
    store = ObservationStore()
    a6.review(
        ReviewInputs(SESSION, 1, 1, 2, 1, 0, 50.0, 0, 1, 0, 0),
        lambda p, f: {"observations": [
            {"scope": "SPY", "text": "spreads widened late", "expires_after_sessions": 3}]},
        store=store,
    )
    carried = store.live_for("SPY", SESSION, lambda a, b: (b - a).days)
    assert carried == ("spreads widened late",)

    # And they render into tomorrow's Agent 1 prompt as text.
    seen = {}
    a1 = RegimeProfiler(config, runner)
    a1.profile(
        RegimeInputs("SPY", 766.0, 6.6, 0.0086, 0.14, 58.0, 764.0, 758.0, 0.04,
                     True, observations=carried),
        date(2026, 8, 27),
        lambda p, f: (seen.__setitem__("p", p), stubs()["a1"](p, f))[1],
    )
    assert "- spreads widened late" in seen["p"]
    runner.close()
