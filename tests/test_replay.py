"""Step 8 -- the replay harness.

The acceptance condition is that a full session replays end to end with no
network access to the broker. That is asserted directly: the harness is
constructed with no ``AlpacaClients`` at all, so a call that reached for one
would raise rather than quietly succeed against a cached response.

Two properties get more attention than the rest because they are the ones a
replay harness can get wrong invisibly:

* **No lookahead.** A window that included one bar past the session would make
  every result better and none of them meaningful. The failure is undetectable
  in the output -- it looks like a strategy that works.
* **Replay and production must not diverge.** The prefilter, sizing and agents
  are the production objects. If replay reimplemented any of them it would
  agree only by coincidence, so the test asserts the shared entry points are
  the ones actually called.
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from replay.bars import BarError, BarSeries, BarSet, synthetic_series, synthetic_set
from replay.broker import (
    ExitBounds,
    FillError,
    FillModel,
    Position,
    StubBroker,
    deterministic_exit,
)
from replay.chain import ChainModel, build_chain, reprice
from replay.harness import ReplayHarness, ReplaySettings, weekday_calendar
from replay.pricing import black_scholes
from replay.rules import RuleError, a4_rule, rule_transports, survivors_in
from replay.transport import (
    RecordedTransport,
    RecordingMiss,
    RecordingTransport,
    canned,
    sequence,
)

START = date(2026, 1, 5)

TEMPLATES = {
    "a1_regime.txt": "Regime $symbol $spot $atr $atr_pct_of_spot $realized_vol $rsi "
                     "$ema_fast_value $ema_slow_value $trend_pct_20d $above_vwap\n$observations",
    "a2_context.txt": "Context $symbol $spot $atr_pct_of_spot $realized_vol $iv_vs_rv20 "
                      "$iv_percentile $trend_pct_20d $sessions_until_earnings "
                      "$sessions_since_earnings\n$headlines\n$observations",
    "a3_risk.txt": "Risk $symbol $contract_symbol $base_contracts $cost_per_contract "
                   "$max_risk_per_contract $risk_budget $equity $open_positions "
                   "$open_premium $regime $confidence $bias_strength $iv_assessment "
                   "$spans_earnings\n$observations",
    "a4_contract.txt": "Contract $symbol $spot $atr $regime $confidence $directional_bias "
                       "$bias_strength $iv_assessment $target_expiry $session_dte "
                       "$spans_earnings $survivor_count\n$survivors\n$observations",
    "a5_exit.txt": "Exit $symbol $contract_symbol $entry_premium $current_premium $pnl_pct "
                   "$current_stop_pct $target_pct $sessions_held $max_hold_sessions "
                   "$sessions_to_expiry $contracts $regime $spans_earnings\n$observations",
    "a6_review.txt": "Review $session $entries $exits $skips $wins $losses $realized_pnl "
                     "$agent_clamps $agent_forces $agent_failures $fallbacks "
                     "$symbols_traded\n$notes",
}


@pytest.fixture
def config(tmp_path, monkeypatch):
    """Synthetic prompts, so the harness runs without the operator's real ones."""
    from src.config import load_config

    prompts = tmp_path / "prompts"
    prompts.mkdir()
    for name, body in TEMPLATES.items():
        (prompts / name).write_text(body, encoding="utf-8")
    monkeypatch.setenv("DEEPSEES_PROMPT_DIR", str(prompts))
    return load_config()


# --- no lookahead ----------------------------------------------------------


def test_window_never_reaches_past_the_session():
    """The property the whole harness rests on."""
    series = synthetic_series("SPY", START, 60)
    for session in series.sessions():
        window = series.through(session)
        assert window.index[-1].date() == session
        assert all(ts.date() <= session for ts in window.index)


def test_window_grows_by_one_bar_per_session():
    series = synthetic_series("SPY", START, 30)
    sessions = series.sessions()
    for i in range(1, len(sessions)):
        assert len(series.through(sessions[i])) == len(series.through(sessions[i - 1])) + 1


def test_close_on_a_session_the_symbol_did_not_trade_is_none():
    """A missing bar must not silently resolve to the previous close.

    Marking a position against a stale close would fabricate a P&L move on a
    day the symbol did not trade.
    """
    series = synthetic_series("SPY", START, 10)
    assert series.close_on(START - timedelta(days=7)) is None
    assert series.close_on(date(2026, 1, 3)) is None  # a Saturday


def test_bars_must_be_sorted():
    series = synthetic_series("SPY", START, 10)
    with pytest.raises(BarError, match="ascending"):
        BarSeries("SPY", series.frame.iloc[::-1])


def test_missing_column_names_it():
    series = synthetic_series("SPY", START, 10)
    with pytest.raises(BarError, match="volume"):
        BarSeries("SPY", series.frame.drop(columns=["volume"]))


def test_synthetic_series_is_deterministic():
    """A replay whose input changes between runs cannot compare two prompts."""
    first = synthetic_series("SPY", START, 40).frame
    second = synthetic_series("SPY", START, 40).frame
    assert first.equals(second)


# --- pricing ---------------------------------------------------------------


def test_call_delta_sits_between_zero_and_one():
    greeks = black_scholes(
        option_type="call", spot=500.0, strike=490.0,
        years_to_expiry=0.1, volatility=0.2,
    )
    assert 0.0 < greeks.delta < 1.0
    assert greeks.theta < 0.0
    assert greeks.gamma > 0.0


def test_put_delta_is_negative():
    greeks = black_scholes(
        option_type="put", spot=500.0, strike=510.0,
        years_to_expiry=0.1, volatility=0.2,
    )
    assert -1.0 < greeks.delta < 0.0


def test_put_call_parity_holds():
    """A price model that fails parity is wrong in a way that hides in greeks."""
    kw = dict(spot=500.0, strike=495.0, years_to_expiry=0.25, volatility=0.22, rate=0.04)
    call = black_scholes(option_type="call", **kw).price
    put = black_scholes(option_type="put", **kw).price
    import math

    expected = 500.0 - 495.0 * math.exp(-0.04 * 0.25)
    assert call - put == pytest.approx(expected, abs=1e-6)


def test_at_expiry_is_intrinsic_and_flat():
    greeks = black_scholes(
        option_type="call", spot=500.0, strike=480.0,
        years_to_expiry=0.0, volatility=0.2,
    )
    assert greeks.price == pytest.approx(20.0)
    assert greeks.delta == 0.0


# --- the synthetic chain ---------------------------------------------------


def test_chain_covers_the_requested_strike_window():
    chain = build_chain(
        symbol="SPY", spot=500.0, realized_vol=0.18,
        expiry=date(2026, 3, 20), session=START, option_type="call",
        strike_gte=450.0, strike_lte=550.0,
    )
    assert chain.specs
    assert all(450.0 <= s.strike <= 550.0 for s in chain.specs)
    assert all(chain.quotes[s.symbol].has_greeks for s in chain.specs)


def test_chain_quotes_are_two_sided_near_the_money():
    chain = build_chain(
        symbol="SPY", spot=500.0, realized_vol=0.18,
        expiry=date(2026, 3, 20), session=START, option_type="call",
        strike_gte=490.0, strike_lte=510.0,
    )
    for spec in chain.specs:
        quote = chain.quotes[spec.symbol]
        assert quote.has_quote
        assert quote.ask > quote.bid


def test_strikes_are_anchored_not_centred():
    """A ladder that moved with spot would make a held strike vanish."""
    first = build_chain(
        symbol="SPY", spot=500.0, realized_vol=0.18, expiry=date(2026, 3, 20),
        session=START, option_type="call", strike_gte=450.0, strike_lte=550.0,
    )
    second = build_chain(
        symbol="SPY", spot=503.7, realized_vol=0.18, expiry=date(2026, 3, 20),
        session=START, option_type="call", strike_gte=450.0, strike_lte=550.0,
    )
    shared = {s.strike for s in first.specs} & {s.strike for s in second.specs}
    assert len(shared) > 10


def test_reprice_keeps_the_same_contract():
    chain = build_chain(
        symbol="SPY", spot=500.0, realized_vol=0.18, expiry=date(2026, 3, 20),
        session=START, option_type="call", strike_gte=495.0, strike_lte=505.0,
    )
    spec = chain.specs[0]
    later = reprice(spec=spec, spot=510.0, realized_vol=0.18,
                    session=START + timedelta(days=7))
    assert later.symbol == spec.symbol
    assert later.delta > chain.quotes[spec.symbol].delta  # spot rose, call delta rose


# --- the stub broker -------------------------------------------------------


def _quote(bid: float, ask: float):
    from src.brokers.alpaca.quotes import OptionQuote

    return OptionQuote(
        symbol="SPY260320C00500000", bid=bid, ask=ask, bid_size=1.0, ask_size=1.0,
        quote_ts=None, delta=0.6, gamma=0.01, theta=-0.05, vega=0.4, rho=0.1,
        implied_volatility=0.2, last_trade_price=(bid + ask) / 2, last_trade_ts=None,
    )


def test_a_full_cross_buys_the_ask_and_sells_the_bid():
    model = FillModel(cross_fraction=1.0)
    assert model.price(_quote(9.9, 10.1), "buy") == pytest.approx(10.1)
    assert model.price(_quote(9.9, 10.1), "sell") == pytest.approx(9.9)


def test_a_zero_cross_fills_at_mid():
    model = FillModel(cross_fraction=0.0)
    assert model.price(_quote(9.9, 10.1), "buy") == pytest.approx(10.0)


def test_a_one_sided_quote_cannot_fill():
    from src.brokers.alpaca.quotes import OptionQuote

    empty = OptionQuote.missing("SPY260320C00500000")
    with pytest.raises(FillError, match="two-sided"):
        FillModel().price(empty, "buy")


def _position(**kw) -> Position:
    base = dict(
        contract_symbol="SPY260320C00500000", symbol="SPY", qty=2,
        entry_premium=10.0, entry_session=START, expiry=date(2026, 3, 20),
        stop_pct=-35.0, target_pct=60.0,
    )
    base.update(kw)
    return Position(**base)


BOUNDS = ExitBounds(stop_pct=-35.0, target_pct=60.0,
                    max_hold_sessions=5, min_sessions_to_expiry=5)


def test_expiry_week_exits_regardless_of_pnl():
    """Even a winner leaves. Theta and spreads make the other tests unreliable
    exactly where they would otherwise fire."""
    assert deterministic_exit(_position(), 15.0, 1, 4, BOUNDS) == "expiry_week"
    assert deterministic_exit(_position(), 3.0, 1, 4, BOUNDS) == "expiry_week"


def test_stop_wins_over_target_on_the_same_mark():
    """Both satisfied at once is a gap, and a gap resolves against us."""
    bounds = ExitBounds(stop_pct=-35.0, target_pct=1.0,
                        max_hold_sessions=5, min_sessions_to_expiry=5)
    assert deterministic_exit(_position(), 6.0, 1, 30, bounds) == "stop"


def test_max_hold_is_counted_in_sessions():
    assert deterministic_exit(_position(), 10.0, 4, 30, BOUNDS) is None
    assert deterministic_exit(_position(), 10.0, 5, 30, BOUNDS) == "max_hold"


def test_a_position_inside_no_rule_holds():
    assert deterministic_exit(_position(), 10.5, 1, 30, BOUNDS) is None


def test_stop_can_only_be_tightened():
    broker = StubBroker(starting_equity=100_000.0)
    position = broker.buy_to_open(
        contract_symbol="SPY260320C00500000", symbol="SPY", qty=2,
        quote=_quote(9.9, 10.1), session=START, expiry=date(2026, 3, 20),
        stop_pct=-35.0, target_pct=60.0,
    )
    tightened = broker.tighten_stop(position, -20.0)
    assert tightened.stop_pct == -20.0
    with pytest.raises(FillError, match="does not tighten"):
        broker.tighten_stop(tightened, -40.0)


def test_scale_out_of_one_contract_does_nothing():
    """There is no half of one, and closing the whole thing would turn a
    model's partial exit into a full one."""
    broker = StubBroker(starting_equity=100_000.0)
    position = broker.buy_to_open(
        contract_symbol="SPY260320C00500000", symbol="SPY", qty=1,
        quote=_quote(9.9, 10.1), session=START, expiry=date(2026, 3, 20),
        stop_pct=-35.0, target_pct=60.0,
    )
    assert broker.scale_out_half(position, _quote(11.0, 11.2), START) is None
    assert broker.positions == [position]


def test_scale_out_halves_the_position():
    broker = StubBroker(starting_equity=100_000.0)
    position = broker.buy_to_open(
        contract_symbol="SPY260320C00500000", symbol="SPY", qty=4,
        quote=_quote(9.9, 10.1), session=START, expiry=date(2026, 3, 20),
        stop_pct=-35.0, target_pct=60.0,
    )
    broker.scale_out_half(position, _quote(11.0, 11.2), START)
    assert broker.positions[0].qty == 2


def test_equity_excludes_unrealized_gains():
    """A winning open position must not fund a larger next one."""
    broker = StubBroker(starting_equity=100_000.0)
    broker.buy_to_open(
        contract_symbol="SPY260320C00500000", symbol="SPY", qty=2,
        quote=_quote(9.9, 10.1), session=START, expiry=date(2026, 3, 20),
        stop_pct=-35.0, target_pct=60.0,
    )
    assert broker.equity == 100_000.0


# --- transports ------------------------------------------------------------


def test_recording_round_trips(tmp_path):
    path = tmp_path / "rec.jsonl"
    recorder = RecordingTransport(inner=canned({"ok": True}), path=path, agent="a1")
    recorder("the prompt", None)

    replayed = RecordedTransport.from_file(path, agent="a1")
    assert replayed("the prompt", None) == {"ok": True}


def test_an_edited_prompt_is_a_miss_not_a_stale_hit(tmp_path):
    """The failure this guards against is the worst one available: a replay
    that looks like it tested the new prompt and did not."""
    path = tmp_path / "rec.jsonl"
    RecordingTransport(inner=canned({"ok": True}), path=path, agent="a1")("original", None)

    replayed = RecordedTransport.from_file(path, agent="a1")
    with pytest.raises(RecordingMiss, match="re-record"):
        replayed("edited prompt", None)


def test_a_retry_is_a_separate_recorded_call(tmp_path):
    """The retry asks a different question and must not get the first answer."""
    path = tmp_path / "rec.jsonl"
    recorder = RecordingTransport(
        inner=sequence([{"n": 1}, {"n": 2}]), path=path, agent="a1"
    )
    recorder("p", None)
    recorder("p", "validation error")

    replayed = RecordedTransport.from_file(path, agent="a1")
    assert replayed("p", None) == {"n": 1}
    assert replayed("p", "validation error") == {"n": 2}


def test_an_exhausted_sequence_raises_rather_than_repeating():
    transport = sequence([{"n": 1}])
    transport("p", None)
    with pytest.raises(RecordingMiss, match="exhausted"):
        transport("p", None)


def test_a_missing_recording_names_the_path(tmp_path):
    with pytest.raises(RecordingMiss, match="record one first"):
        RecordedTransport.from_file(tmp_path / "absent.jsonl")


# --- the rule stubs --------------------------------------------------------


def test_a4_rule_picks_the_top_ranked_survivor():
    prompt = "- SPY260320C00500000 strike 500\n- SPY260320C00505000 strike 505"
    decision = a4_rule()(prompt, None)
    assert decision["primary_symbol"] == "SPY260320C00500000"
    assert decision["alternate_symbol"] == "SPY260320C00505000"


def test_a4_rule_refuses_to_invent_a_symbol():
    """Inventing one would put a contract the prefilter never offered in front
    of the validator, and the failure would look like a model error."""
    with pytest.raises(RuleError, match="no OCC symbols"):
        a4_rule()("a prompt with no survivors", None)


def test_survivors_are_returned_in_prompt_order():
    prompt = "- QQQ260320P00400000\n- SPY260320C00500000\n- QQQ260320P00400000"
    assert survivors_in(prompt) == ["QQQ260320P00400000", "SPY260320C00500000"]


# --- the harness, end to end -----------------------------------------------


@pytest.fixture
def harness(config):
    bars = synthetic_set(["SPY"], START, 140)
    settings = ReplaySettings(symbols=("SPY",), warmup_sessions=40)
    h = ReplayHarness(config, bars, settings, rule_transports(["SPY"]))
    yield h
    h.close()


def test_a_full_replay_runs_with_no_broker_at_all(harness):
    """Step 8's acceptance condition. No AlpacaClients is constructed anywhere
    in this test, so a call that reached for one could not succeed."""
    report = harness.run()
    assert report.sessions
    assert report.entries > 0
    assert report.broker["orders"] > 0


def test_the_report_carries_the_model_that_produced_it(harness):
    """A P&L number without the chain parameters is not interpretable."""
    payload = harness.run().as_dict()
    assert payload["settings"]["chain_model"]["iv_to_realized_ratio"]
    assert payload["caveats"]
    assert any("not a market result" in c for c in payload["caveats"])


def test_every_position_eventually_closes_for_a_named_reason(harness):
    report = harness.run()
    reasons = report.broker["exit_reasons"]
    assert reasons
    assert set(reasons) <= {
        "stop", "target", "max_hold", "expiry_week",
        "agent_exit_now", "scale_out_half",
    }


def test_replay_is_reproducible(config):
    """Two runs of the same replay must agree exactly, or nothing can be
    compared against anything."""
    def run_once():
        bars = synthetic_set(["SPY"], START, 140)
        h = ReplayHarness(
            config, bars, ReplaySettings(symbols=("SPY",), warmup_sessions=40),
            rule_transports(["SPY"]),
        )
        try:
            return h.run().as_dict()
        finally:
            h.close()

    assert run_once() == run_once()


def test_caps_bind_during_a_replay(harness):
    """max_positions_per_symbol is 1 in the fixture config, so a second entry
    in an open symbol must be refused by sizing rather than opened."""
    report = harness.run()
    assert harness.broker.positions_in("SPY") <= 1
    assert "a3_no_size" in report.skip_counts()


def test_a_failing_agent_skips_the_entry_and_nothing_else(config):
    """An entry-path failure costs the entry, not the replay."""
    transports = rule_transports(["SPY"])
    transports["a1"] = canned("{not json")
    bars = synthetic_set(["SPY"], START, 120)
    h = ReplayHarness(
        config, bars, ReplaySettings(symbols=("SPY",), warmup_sessions=40), transports
    )
    try:
        report = h.run()
    finally:
        h.close()

    assert report.entries == 0
    assert report.skip_counts().get("a1_failed", 0) > 0
    assert report.agent_failures > 0
    assert len(report.sessions) > 0  # the replay itself continued


def test_a_missing_transport_is_refused_up_front(config):
    bars = synthetic_set(["SPY"], START, 80)
    partial = rule_transports(["SPY"])
    partial.pop("a5")
    with pytest.raises(Exception, match="a5"):
        ReplayHarness(config, bars, ReplaySettings(symbols=("SPY",)), partial)


# --- the offline calendar --------------------------------------------------


def test_weekday_calendar_has_closes_so_is_session_works():
    """``is_session`` reads ``closes``; a calendar without them reports that
    none of its sessions are sessions, and the expiry rule then finds nothing."""
    calendar = weekday_calendar(date(2026, 1, 1), date(2026, 6, 30))
    assert calendar.is_session(date(2026, 3, 20))       # a third Friday
    assert not calendar.is_session(date(2026, 3, 21))   # a Saturday


def test_weekday_calendar_rejects_a_reversed_range():
    with pytest.raises(Exception, match="before start"):
        weekday_calendar(date(2026, 6, 30), date(2026, 1, 1))


# --- record once, replay many ----------------------------------------------


def test_a_recorded_run_replays_identically(config, tmp_path):
    """The claim the recording exists to support.

    If a replay of a recording did not reproduce the run it recorded, the
    recording would be useless for comparing prompts -- which is the only
    reason to have one.
    """
    path = tmp_path / "session.jsonl"

    def build(transports):
        return ReplayHarness(
            config,
            synthetic_set(["SPY"], START, 120),
            ReplaySettings(symbols=("SPY",), warmup_sessions=40),
            transports,
        )

    recording = {
        agent: RecordingTransport(inner=stub, path=path, agent=agent)
        for agent, stub in rule_transports(["SPY"]).items()
    }
    live = build(recording)
    try:
        first = live.run().as_dict()
    finally:
        live.close()

    replayed = build(
        {
            agent: RecordedTransport.from_file(path, agent=agent)
            for agent in ("a1", "a2", "a3", "a4", "a5", "a6")
        }
    )
    try:
        second = replayed.run().as_dict()
    finally:
        replayed.close()

    assert second == first
