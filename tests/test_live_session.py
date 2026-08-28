"""The live session handlers. The module that actually sends orders.

It had no tests until an attribute typo -- ``position.contract_symbol`` on a
class whose field is ``symbol`` -- reached a live session and took out the exit
handler on every tick. The orchestrator isolated it, so the loop survived and
the deterministic exits stayed armed, but no open position was managed until it
was found. These are the tests that would have caught it before the market did.

Everything is faked. No broker, no provider, no network.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta

import pandas as pd
import pytest

from src.brokers.alpaca.calendar import ET, TradingCalendar
from src.brokers.alpaca.quotes import OptionQuote
from src.orchestrator.live import LiveSession
from src.orchestrator.session import SessionClock, SessionWindows

SESSION = date(2026, 8, 31)
CONTRACT = "PLTR261016C00170000"

TEMPLATES = {
    "a1_regime.txt": "$symbol $spot $atr $atr_pct_of_spot $realized_vol $rsi "
                     "$ema_fast_value $ema_slow_value $trend_pct_20d $above_vwap $observations",
    "a2_context.txt": "$symbol $spot $atr_pct_of_spot $realized_vol $iv_vs_rv20 "
                      "$iv_percentile $trend_pct_20d $headlines $sessions_until_earnings "
                      "$sessions_since_earnings $observations",
    "a3_risk.txt": "$symbol $contract_symbol $base_contracts $cost_per_contract "
                   "$max_risk_per_contract $risk_budget $equity $open_positions "
                   "$open_premium $regime $confidence $bias_strength $iv_assessment "
                   "$spans_earnings $observations",
    "a4_contract.txt": "$symbol $spot $atr $regime $confidence $directional_bias "
                       "$bias_strength $iv_assessment $target_expiry $session_dte "
                       "$spans_earnings $survivors $survivor_count $observations",
    "a5_exit.txt": "$symbol $contract_symbol $entry_premium $current_premium $pnl_pct "
                   "$current_stop_pct $target_pct $sessions_held $max_hold_sessions "
                   "$sessions_to_expiry $contracts $regime $spans_earnings $observations",
    "a6_review.txt": "$session $entries $exits $skips $wins $losses $realized_pnl "
                     "$agent_clamps $agent_forces $agent_failures $fallbacks "
                     "$symbols_traded $notes",
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


# --- fakes -----------------------------------------------------------------


class RecordingLog:
    def __init__(self):
        self.records = []

    def write(self, payload, action, **kw):
        self.records.append({"payload": payload, "action": action, **kw})

    def of_kind(self, kind):
        return [r for r in self.records if r["payload"].kind == kind]


class FakePosition:
    """Shaped like OptionPosition. Field names matter -- that is the point."""

    def __init__(self, symbol=CONTRACT, qty=1, entry=20.0, current=19.0):
        self.symbol = symbol
        self.underlying = "PLTR"
        self.qty = qty
        self.avg_entry_price = entry
        self.current_price = current
        self.market_value = current * qty * 100
        self.cost_basis = entry * qty * 100
        self.unrealized_pl = (current - entry) * qty * 100
        self.unrealized_plpc = 0.0
        self.side = "long"
        self.expiry = date(2026, 10, 16)
        self.strike = 170.0
        self.option_type = "call"

    def pnl_pct(self):
        return (self.current_price - self.avg_entry_price) / self.avg_entry_price * 100.0

    @property
    def premium_paid(self):
        return self.avg_entry_price * abs(self.qty) * 100


class FakeAccount:
    equity = 100_000.0
    options_buying_power = 100_000.0
    cash = 100_000.0
    multiplier = "1"


class FakeTrading:
    def __init__(self):
        self.orders = []

    def get_account(self):
        return FakeAccount()

    def get_all_positions(self):
        return []


class FakeClients:
    def __init__(self, config):
        self.config = config
        self.trading = FakeTrading()
        self.equities_feed = "iex"
        self.options_feed = "indicative"
        self.mock = False

    def assert_writable(self, action):
        return None


def calendar_for(days=90) -> TradingCalendar:
    sessions = tuple(
        SESSION - timedelta(days=30) + timedelta(days=i)
        for i in range(days)
        if (SESSION - timedelta(days=30) + timedelta(days=i)).weekday() < 5
    )
    return TradingCalendar(
        sessions=sessions,
        closes={d: datetime.combine(d, time(16, 0), tzinfo=ET) for d in sessions},
    )


def quote(bid=18.9, ask=19.1) -> OptionQuote:
    return OptionQuote(
        symbol=CONTRACT, bid=bid, ask=ask, bid_size=10.0, ask_size=10.0,
        quote_ts=datetime(2026, 8, 31, 15, 0), delta=0.6, gamma=0.01, theta=-0.05,
        vega=0.4, rho=0.1, implied_volatility=0.4, last_trade_price=19.0,
        last_trade_ts=datetime(2026, 8, 31, 15, 0),
    )


def transports(**overrides):
    base = {
        "a1": lambda p, f: {
            "symbol": "PLTR", "regime": "trending_up", "confidence": 0.8,
            "signal_profile": {"ema_fast": 9, "confirmation_bars": 1,
                               "require_vwap_alignment": False, "min_atr_multiple": 0.3,
                               "allowed_direction": "long_calls"},
            "rationale": "test"},
        "a2": lambda p, f: {
            "symbol": "PLTR", "eligible": True, "hard_blocks": [],
            "directional_bias": "bullish", "bias_strength": 0.8, "event_risk": "low",
            "iv_assessment": "fair", "notes": ""},
        "a3": lambda p, f: {"size_multiplier": 1.0, "reason": "full"},
        "a4": lambda p, f: {"structure": "single_leg", "primary_symbol": CONTRACT,
                            "expected_hold_sessions": 3, "reason": "best"},
        "a5": lambda p, f: {"action": "hold", "new_stop_pct": None, "reason": "steady"},
        "a6": lambda p, f: {"observations": []},
    }
    base.update(overrides)
    return base


@pytest.fixture
def session(config):
    log = RecordingLog()
    s = LiveSession(
        config=config, clients=FakeClients(config), calendar=calendar_for(),
        transports=transports(), decision_log=log,
        symbols=("PLTR", "MSFT"), dry_run=True,
    )
    s.recording = log
    yield s
    s.close()


def state_at(clock_time=time(10, 0)):
    windows = SessionWindows(
        market_open=time(9, 30), market_close=time(16, 0), first_entry=time(9, 45),
        last_entry=time(15, 0), flat_by=time(15, 45), skip_dates=frozenset(),
    )
    return SessionClock(windows, calendar_for()).state(
        datetime.combine(SESSION, clock_time, tzinfo=ET)
    )


# --- the typo that reached production --------------------------------------


def test_managing_a_position_uses_the_real_field_names(session, monkeypatch):
    """``OptionPosition`` has ``symbol``, not ``contract_symbol``.

    The regression this file was written for: the wrong attribute took out the
    exit handler on every tick, so no open position was managed at all.
    """
    position = FakePosition()
    monkeypatch.setattr(
        "src.orchestrator.live.reconcile",
        lambda clients: type("B", (), {
            "positions": [position], "symbols": (position.symbol,),
            "open_premium": position.premium_paid, "unparseable": (),
            "__len__": lambda self: 1,
        })(),
    )
    monkeypatch.setattr(
        "src.orchestrator.live.fetch_snapshots",
        lambda clients, symbols, **kw: {position.symbol: quote()},
    )
    session.a5_exit(state_at())
    assert not [s for s in session.skips if s[0] == "a5"]


def test_a_position_is_marked_and_held_when_inside_every_bound(session, monkeypatch, caplog):
    position = FakePosition(entry=20.0, current=19.0)      # -5%, inside the -35 stop
    monkeypatch.setattr(
        "src.orchestrator.live.reconcile",
        lambda clients: type("B", (), {
            "positions": [position], "symbols": (position.symbol,),
            "open_premium": 0.0, "unparseable": (), "__len__": lambda self: 1,
        })(),
    )
    monkeypatch.setattr("src.orchestrator.live.fetch_snapshots",
                        lambda clients, symbols, **kw: {position.symbol: quote()})
    with caplog.at_level("INFO"):
        session.a5_exit(state_at())
    assert "hold" in caplog.text


def test_an_exit_is_recorded_when_the_stop_is_breached(session, monkeypatch):
    """The deterministic exits fire on the mark regardless of Agent 5."""
    position = FakePosition(entry=20.0, current=10.0)      # -50%, past the -35 stop
    monkeypatch.setattr(
        "src.orchestrator.live.reconcile",
        lambda clients: type("B", (), {
            "positions": [position], "symbols": (position.symbol,),
            "open_premium": 0.0, "unparseable": (), "__len__": lambda self: 1,
        })(),
    )
    monkeypatch.setattr("src.orchestrator.live.fetch_snapshots",
                        lambda clients, symbols, **kw: {position.symbol: quote(9.9, 10.1)})
    session.a5_exit(state_at())
    # dry_run, so it records the intent rather than sending an order.
    assert any("would exit (stop)" in why for _, _, why in session.skips)


def test_a_position_without_a_quote_is_skipped_not_marked(session, monkeypatch):
    """Marking against a missing quote would fabricate a P&L."""
    position = FakePosition()
    monkeypatch.setattr(
        "src.orchestrator.live.reconcile",
        lambda clients: type("B", (), {
            "positions": [position], "symbols": (position.symbol,),
            "open_premium": 0.0, "unparseable": (), "__len__": lambda self: 1,
        })(),
    )
    monkeypatch.setattr(
        "src.orchestrator.live.fetch_snapshots",
        lambda clients, symbols, **kw: {position.symbol: OptionQuote.missing(position.symbol)},
    )
    session.a5_exit(state_at())
    assert any(stage == "a5" and "no two-sided quote" in why
               for stage, _, why in session.skips)


# --- skips reach the decision log ------------------------------------------


def test_every_skip_is_written_to_the_decision_log(session):
    """A skip that only reached stderr would be invisible in the artifact the
    session exists to produce."""
    session.skip("prefilter", "PLTR", "no survivors")
    [record] = session.recording.of_kind("skip")
    assert record["payload"].stage == "prefilter"
    assert record["payload"].reason == "no survivors"
    assert record["symbol"] == "PLTR"


def test_a_skip_carries_the_current_trace(session):
    session._trace = "e001-PLTR"
    session.skip("a3", "PLTR", "sized to zero")
    assert session.recording.records[-1]["trace_id"] == "e001-PLTR"


# --- session state ---------------------------------------------------------


def test_rolling_a_session_drops_the_profile(session):
    """A regime read is a judgment about one day. Carrying it into the next
    would apply it to a session it was never made for."""
    session.roll(SESSION)
    session.eligible["PLTR"] = object()
    session.profiles["PLTR"] = object()
    session.entries_this_session = 2

    session.roll(SESSION + timedelta(days=1))
    assert session.eligible == {}
    assert session.profiles == {}
    assert session.entries_this_session == 0


def test_rolling_to_the_same_session_is_a_no_op(session):
    session.roll(SESSION)
    session.eligible["PLTR"] = object()
    session.roll(SESSION)
    assert "PLTR" in session.eligible


# --- fail-closed -----------------------------------------------------------


def test_no_symbol_is_admitted_when_the_earnings_feed_is_unavailable(session, monkeypatch):
    """An unknown earnings date excludes, so a dead feed must exclude
    everything -- running the model without the check inverts the rule."""
    monkeypatch.setattr(session, "_earnings_inputs", lambda: (None, None))
    monkeypatch.setattr(session, "stats", lambda s, d: (100.0, 2.0, 0.3))
    monkeypatch.setattr(session, "bars", lambda s, d: pd.DataFrame(
        {"open": [1.0] * 30, "high": [1.0] * 30, "low": [1.0] * 30,
         "close": [1.0] * 30, "volume": [1] * 30},
        index=pd.date_range("2026-07-01", periods=30),
    ))
    session.a2_context(state_at(time(9, 0)))
    assert session.eligible == {}
    assert any("earnings feed unavailable" in why for _, _, why in session.skips)


def test_entry_is_refused_when_nothing_was_profiled(session):
    session.roll(SESSION)
    session.entry_scan(state_at())
    assert any(stage == "entry" and "no profiled symbol" in why
               for stage, _, why in session.skips)


# --- the handler surface ---------------------------------------------------


def test_every_scheduled_job_has_a_handler(session, config):
    """The orchestrator refuses to start otherwise -- this keeps the two in
    step as jobs are added."""
    from src.orchestrator.scheduler import standard_jobs

    handlers = session.handlers()
    for job in standard_jobs(config.limits):
        assert job.name in handlers, f"no handler for {job.name}"


def test_handlers_are_callable_with_a_state(session):
    for name, handler in session.handlers().items():
        assert callable(handler), name
