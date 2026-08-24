"""Prefilter: ordering, multi-label rejection, hard rejects, ranking, capping."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pytest

from src.brokers.alpaca.cache import MarketDataCache, TtlCache
from src.brokers.alpaca.calendar import TradingCalendar
from src.brokers.alpaca.contracts import ContractSpec
from src.brokers.alpaca.quotes import OptionQuote
from src.decisionlog.adapters import prefilter_payload
from src.options.prefilter import evaluate_candidates, run_prefilter
from tests.test_cache import FakeClock

def _band():
    from src.config import load_config

    limits = load_config().limits
    return limits.get_float("prefilter.delta_min"), limits.get_float("prefilter.delta_max")


DELTA_MIN, DELTA_MAX = _band()
# Fixtures are derived from the configured band, never hardcoded. The strategy
# revision moved the single-leg band from 0.30-0.60 to 0.55-0.75 and every
# hardcoded delta in this file silently started testing a different gate.
IN_BAND = round((DELTA_MIN + DELTA_MAX) / 2, 4)
JUST_OVER_BAND = round(DELTA_MAX * 1.05, 4)      # ~5% out: near-boundary
FAR_OVER_BAND = round(min(0.99, DELTA_MAX * 1.35), 4)   # well out: not near

SESSION = date(2026, 8, 24)
# Must reach past the September monthly (2026-09-18): the prefilter now
# CHOOSES the nearest monthly expiry rather than bounding a day window, and
# sessions cannot be counted across days that were never fetched.
SESSIONS = tuple(
    SESSION + timedelta(days=d)
    for d in (0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25)
)


def _pinned(limits, **overrides):
    """Limits with specific keys pinned, so tuning config cannot break tests.

    The suite must assert against fixed numbers. Reading the operator's tuned
    ``config/limits.yaml`` made every legitimate tuning decision a test
    failure -- and those values are not even in a fresh clone.
    """
    from src.config import Section

    data = limits.as_dict()
    for dotted, value in overrides.items():
        node = data
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return Section(data, limits.source)


@pytest.fixture
def calendar():
    return TradingCalendar(
        sessions=SESSIONS,
        closes={d: datetime(d.year, d.month, d.day, 16, 0) for d in SESSIONS},
    )


@pytest.fixture
def limits():
    from src.config import load_config

    # Pinned: these fixtures use a 1-session expiry and unmixed weeklies.
    return _pinned(
        load_config().limits,
        **{"prefilter.dte_min": 1, "prefilter.dte_max": 45,
           "prefilter.require_monthly_expiry": False,
           "prefilter.prefer_monthly_expiry": False},
    )


def spec(symbol="SPY260918C00100000", strike=100.0, expiry=date(2026, 9, 18),
         option_type="call", open_interest=500, tradable=True) -> ContractSpec:
    return ContractSpec(
        symbol=symbol, underlying="SPY", root="SPY", expiry=expiry, strike=strike,
        option_type=option_type, style="american", open_interest=open_interest,
        open_interest_date=None, close_price=None, close_price_date=None,
        size=100, tradable=tradable, status="active",
    )


def quote(symbol="SPY260918C00100000", bid=1.97, ask=2.03, delta=IN_BAND, gamma=0.04,
          theta=-0.20, iv=0.30) -> OptionQuote:
    """Default is a contract that passes every gate comfortably.

    Spread 0.06 on a 2.00 mid = 3%, inside the 6% cap. Fixtures that sit
    outside a threshold by accident make every test assert the wrong gate.
    """
    return OptionQuote(
        symbol=symbol, bid=bid, ask=ask, bid_size=10, ask_size=10,
        quote_ts=datetime(2026, 8, 24, 15, 0), delta=delta, gamma=gamma, theta=theta,
        vega=0.1, rho=0.01, implied_volatility=iv, last_trade_price=2.0,
        last_trade_ts=datetime(2026, 8, 24, 15, 0),
    )


def evaluate(specs, quotes, calendar, limits, spot=100.0, atr=2.0, rv=0.25):
    return evaluate_candidates(
        specs, {q.symbol: q for q in quotes}, calendar, SESSION, spot, atr, rv, limits
    )


# --- the hard reject -------------------------------------------------------


def test_missing_delta_is_a_hard_reject(calendar, limits):
    [candidate] = evaluate([spec()], [quote(delta=None)], calendar, limits)
    assert not candidate.survived
    assert "no delta" in candidate.failures
    assert "no greeks" in candidate.failures
    assert candidate.metrics is None


def test_missing_delta_is_never_defaulted_from_moneyness(calendar, limits):
    """An ATM contract with no delta must not be assumed to be 0.50."""
    [candidate] = evaluate([spec(strike=100.0)], [quote(delta=None)], calendar, limits)
    assert candidate.quote.delta is None
    assert candidate.metrics is None


def test_missing_iv_is_also_a_hard_reject(calendar, limits):
    """IV goes missing on exactly the contracts greeks do, and every survivor
    must have all six metrics populated."""
    [candidate] = evaluate([spec()], [quote(iv=None)], calendar, limits)
    assert "no iv" in candidate.failures


def test_a_contract_that_passes_every_gate_but_cannot_be_scored_is_rejected(calendar, limits):
    """Belt and braces: metric failure demotes rather than shipping partials."""
    [candidate] = evaluate([spec()], [quote(gamma=None)], calendar, limits)
    assert not candidate.survived


# --- multi-label -----------------------------------------------------------


def test_all_failing_reasons_are_recorded_not_just_the_first(calendar, limits):
    bad = quote(bid=0.01, ask=5.0, delta=FAR_OVER_BAND)
    [candidate] = evaluate([spec(open_interest=0)], [bad], calendar, limits)
    assert {"bid below floor", "open interest", "spread", "delta band"} <= set(candidate.failures)


def test_reason_counts_sum_above_rejected(calendar, limits):
    candidates = evaluate(
        [spec(symbol="A" * 3 + "260918C00100000", open_interest=0),
         spec(symbol="B" * 3 + "260918C00100000")],
        [quote(symbol="A" * 3 + "260918C00100000", bid=0.01, ask=4.0),
         quote(symbol="B" * 3 + "260918C00100000")],
        calendar, limits,
    )
    payload = prefilter_payload(candidates, thresholds={}, detail="aggregate")
    assert sum(payload.reason_counts.values()) > payload.rejected


# --- individual gates ------------------------------------------------------


def test_survivor_gets_all_six_metrics_populated(calendar, limits):
    [candidate] = evaluate([spec()], [quote()], calendar, limits)
    assert candidate.survived
    assert candidate.metrics is not None and candidate.metrics.is_finite
    values = candidate.metrics.as_dict()
    for key in ("theta_pct_per_day", "gamma_per_1pct", "iv_vs_rv",
                "spread_cost_pct_of_atr", "breakeven_distance_atr", "modeled_pnl_1atr"):
        assert key in values


def test_session_dte_gate_rejects_zero_dte(calendar, limits):
    [candidate] = evaluate(
        [spec(symbol="SPY260824C00100000", expiry=SESSION)], [quote("SPY260824C00100000")],
        calendar, limits,
    )
    assert "session dte" in candidate.failures


def test_expired_contract_is_flagged(calendar, limits):
    past = spec(symbol="SPY260821C00100000", expiry=date(2026, 8, 21))
    [candidate] = evaluate([past], [quote("SPY260821C00100000")], calendar, limits)
    assert "expired" in candidate.failures


def test_untradable_contract_is_rejected(calendar, limits):
    [candidate] = evaluate([spec(tradable=False)], [quote()], calendar, limits)
    assert "not tradable" in candidate.failures


def test_missing_snapshot_yields_no_quote(calendar, limits):
    candidates = evaluate_candidates(
        [spec()], {}, calendar, SESSION, 100.0, 2.0, 0.25, limits
    )
    assert "no quote" in candidates[0].failures


# --- boundary distance -----------------------------------------------------


def test_single_reason_reject_records_how_close_it_came(calendar, limits):
    """5% past delta_max is comfortably inside the near-boundary window."""
    [candidate] = evaluate([spec()], [quote(delta=JUST_OVER_BAND)], calendar, limits)
    assert candidate.failures == ("delta band",)
    assert candidate.boundary_distance == pytest.approx(0.05, abs=0.01)


def test_a_far_miss_is_not_near_boundary(calendar, limits):
    [candidate] = evaluate([spec()], [quote(delta=FAR_OVER_BAND)], calendar, limits)
    assert candidate.failures == ("delta band",)
    assert candidate.boundary_distance > 0.20


def test_multi_reason_rejects_have_no_boundary_distance(calendar, limits):
    [candidate] = evaluate([spec(open_interest=0)], [quote(delta=FAR_OVER_BAND)], calendar, limits)
    assert len(candidate.failures) > 1
    assert candidate.boundary_distance is None


# --- log detail reduction --------------------------------------------------


def test_boundary_detail_keeps_only_near_misses_and_kept_symbols(calendar, limits):
    near = spec(symbol="NEAR260918C00100000")
    far = spec(symbol="FARX260918C00100000")
    keep = spec(symbol="KEEP260918C00100000", open_interest=0)
    candidates = evaluate(
        [near, far, keep],
        [quote("NEAR260918C00100000", delta=JUST_OVER_BAND),
         quote("FARX260918C00100000", delta=FAR_OVER_BAND),
         quote("KEEP260918C00100000", delta=FAR_OVER_BAND)],
        calendar, limits,
    )
    payload = prefilter_payload(
        candidates, thresholds={}, detail="boundary",
        near_boundary_pct=0.20, keep_symbols=["KEEP260918C00100000"],
    )
    assert "NEAR260918C00100000" in payload.rejections     # near miss
    assert "KEEP260918C00100000" in payload.rejections     # explicitly kept
    assert "FARX260918C00100000" not in payload.rejections  # far miss, dropped
    # aggregates stay complete regardless of retention
    assert payload.rejected == 3
    assert sum(payload.reason_counts.values()) >= 3


def test_aggregate_detail_keeps_no_rows(calendar, limits):
    candidates = evaluate([spec()], [quote(delta=JUST_OVER_BAND)], calendar, limits)
    payload = prefilter_payload(candidates, thresholds={}, detail="aggregate")
    assert payload.rejections == {}
    assert payload.reason_counts


def test_full_detail_keeps_everything(calendar, limits):
    candidates = evaluate([spec()], [quote(delta=FAR_OVER_BAND)], calendar, limits)
    payload = prefilter_payload(candidates, thresholds={}, detail="full")
    assert len(payload.rejections) == 1


def test_boundary_detail_is_much_smaller_than_full(calendar, limits):
    """The whole reason for the setting: a real scan is hundreds of contracts."""
    specs, quotes = [], []
    for i in range(200):
        symbol = f"SPY2608{25 + i % 3}C{100000 + i * 1000:08d}"
        specs.append(spec(symbol=symbol, strike=100.0 + i))
        quotes.append(quote(symbol=symbol, delta=FAR_OVER_BAND))   # all far misses
    candidates = evaluate(specs, quotes, calendar, limits)
    full = prefilter_payload(candidates, thresholds={}, detail="full")
    boundary = prefilter_payload(candidates, thresholds={}, detail="boundary")
    assert len(full.rejections) > 10 * max(1, len(boundary.rejections))


# --- ranking and capping ---------------------------------------------------


class FakeClients:
    """Serves a fixed narrowed universe and records what was requested."""

    def __init__(self, config, specs, quotes):
        self.config = config
        self._specs = specs
        self._quotes = quotes
        self.contract_requests = []
        self.snapshot_symbols = []
        self.options_feed = "indicative"
        self.trading = SimpleNamespace(get_option_contracts=self._contracts)
        self.options = SimpleNamespace(get_option_snapshot=self._snapshots)

    def _contracts(self, request):
        self.contract_requests.append(request)
        return SimpleNamespace(option_contracts=[_as_api(s) for s in self._specs],
                               next_page_token=None)

    def _snapshots(self, request):
        self.snapshot_symbols.extend(request.symbol_or_symbols)
        return {q.symbol: _as_api_snapshot(q) for q in self._quotes}


def _as_api(s: ContractSpec):
    return SimpleNamespace(
        symbol=s.symbol, underlying_symbol=s.underlying, root_symbol=s.root,
        expiration_date=s.expiry, strike_price=str(s.strike), type=s.option_type,
        style=s.style, open_interest=str(s.open_interest), open_interest_date=None,
        close_price=None, close_price_date=None, size="100", tradable=s.tradable,
        status="active",
    )


def _as_api_snapshot(q: OptionQuote):
    return SimpleNamespace(
        latest_quote=SimpleNamespace(bid_price=q.bid, ask_price=q.ask, bid_size=q.bid_size,
                                     ask_size=q.ask_size, timestamp=q.quote_ts),
        greeks=None if q.delta is None else SimpleNamespace(
            delta=q.delta, gamma=q.gamma, theta=q.theta, vega=q.vega, rho=q.rho),
        implied_volatility=q.implied_volatility,
        latest_trade=SimpleNamespace(price=q.last_trade_price, timestamp=q.last_trade_ts),
    )


def build_universe(n=30):
    specs, quotes = [], []
    for i in range(n):
        symbol = f"SPY260918C{(95000 + i * 500):08d}"
        strike = (95000 + i * 500) / 1000
        specs.append(spec(symbol=symbol, strike=strike))
        # Vary the spread so ranking has something to discriminate on, while
        # keeping every one inside the 6% cap so the cap is not what is tested.
        half = 0.02 + (i % 5) * 0.004
        quotes.append(quote(symbol=symbol, bid=2.00 - half, ask=2.00 + half, delta=IN_BAND))
    return specs, quotes


class _PinnedConfig:
    """The real Config with a pinned limits Section.

    ``run_prefilter`` reads ``clients.config.limits`` directly, so pinning the
    ``limits`` fixture alone does not reach it. SESSIONS below is a hardcoded
    13-session calendar, which cannot satisfy a 21-32 session band -- so the
    band is pinned here rather than the calendar being regenerated per tuning.
    """

    def __init__(self, config, limits):
        self._config = config
        self.limits = limits

    def __getattr__(self, name):
        return getattr(self._config, name)


@pytest.fixture
def config():
    from src.config import load_config

    real = load_config()
    return _PinnedConfig(real, _pinned(
        real.limits,
        **{"prefilter.dte_min": 1, "prefilter.dte_max": 45,
           "prefilter.require_monthly_expiry": False,
           "prefilter.prefer_monthly_expiry": False},
    ))


def test_universe_is_narrowed_before_snapshots_are_requested(config, calendar):
    """The ordering change: never snapshot chain we cannot use."""
    specs, quotes = build_universe()
    clients = FakeClients(config, specs, quotes)
    run_prefilter(clients, "SPY", spot=100.0, atr=2.0, realized_vol=0.25,
                  calendar=calendar, order_session=SESSION, option_type="call")

    request = clients.contract_requests[0]
    window = config.limits.get_float("prefilter.strike_window_pct")
    assert float(request.strike_price_gte) == pytest.approx(100.0 * (1 - window))
    assert float(request.strike_price_lte) == pytest.approx(100.0 * (1 + window))
    assert request.expiration_date_gte is not None
    assert request.expiration_date_lte is not None
    # Snapshots requested only for the narrowed set.
    assert set(clients.snapshot_symbols) == {s.symbol for s in specs}


def test_the_expiry_is_a_chosen_monthly_not_a_day_window(config, calendar):
    """The rule changed: pick the nearest monthly at least N sessions out.

    A fixed calendar-day window inside a ~30-day monthly cycle misses the
    monthly about half the time -- on 2026-08-24 a 30-45 day band fell
    entirely between the September and October monthlies and requiring
    monthlies inside it produced an empty survivor set on every symbol.
    """
    from src.options.expiry import is_standard_monthly

    specs, quotes = build_universe()
    clients = FakeClients(config, specs, quotes)
    result = run_prefilter(clients, "SPY", spot=100.0, atr=2.0, realized_vol=0.25,
                           calendar=calendar, order_session=SESSION, option_type="call")
    request = clients.contract_requests[0]

    # One expiry is requested, not a range.
    assert request.expiration_date_gte == request.expiration_date_lte
    assert request.expiration_date_gte == result.target_expiry

    # It is a real monthly, and it clears the configured floor.
    assert is_standard_monthly(result.target_expiry, calendar.is_session)
    floor = config.limits.get_int("prefilter.monthly_min_sessions")
    assert result.target_session_dte >= floor
    assert result.target_session_dte == calendar.sessions_until(result.target_expiry, SESSION)

    # The realised DTE is recorded for the entry, never assumed from config.
    assert result.thresholds["target_session_dte"] == float(result.target_session_dte)


def test_survivors_are_ranked_by_pnl_to_spread_ratio(config, calendar):
    specs, quotes = build_universe()
    clients = FakeClients(config, specs, quotes)
    result = run_prefilter(clients, "SPY", spot=100.0, atr=2.0, realized_vol=0.25,
                           calendar=calendar, order_session=SESSION, option_type="call")
    ratios = [c.metrics.pnl_to_spread_ratio for c in result.survivors]
    assert ratios == sorted(ratios, reverse=True)


def test_only_twelve_reach_the_model_but_all_survivors_are_kept(config, calendar):
    specs, quotes = build_universe(n=30)
    clients = FakeClients(config, specs, quotes)
    result = run_prefilter(clients, "SPY", spot=100.0, atr=2.0, realized_vol=0.25,
                           calendar=calendar, order_session=SESSION, option_type="call")
    cap = config.limits.get_int("prefilter.max_survivors_per_symbol")
    assert cap == 12
    assert len(result.top) == cap
    assert len(result.survivors) > cap          # full set retained for the log
    assert result.top == result.survivors[:cap]


def test_coverage_is_reported_for_the_narrowed_window(config, calendar):
    specs, quotes = build_universe(n=20)
    quotes[0] = replace(quotes[0], delta=None, gamma=None, theta=None, implied_volatility=None)
    clients = FakeClients(config, specs, quotes)
    result = run_prefilter(clients, "SPY", spot=100.0, atr=2.0, realized_vol=0.25,
                           calendar=calendar, order_session=SESSION, option_type="call")
    coverage = result.narrowed_coverage
    assert coverage["window"] == "narrowed"
    assert coverage["total"] == 20
    assert coverage["missing_greeks"] == 1
    assert coverage["coverage"] == pytest.approx(0.95)


def test_near_boundary_helper_filters_by_distance(config, calendar):
    specs, quotes = build_universe(n=6)
    quotes[0] = replace(quotes[0], delta=JUST_OVER_BAND)   # just outside
    quotes[1] = replace(quotes[1], delta=FAR_OVER_BAND)    # far outside
    clients = FakeClients(config, specs, quotes)
    result = run_prefilter(clients, "SPY", spot=100.0, atr=2.0, realized_vol=0.25,
                           calendar=calendar, order_session=SESSION, option_type="call")
    near = result.near_boundary(within=0.20)
    assert quotes[0].symbol in {c.symbol for c in near}
    assert quotes[1].symbol not in {c.symbol for c in near}


def test_prefilter_uses_the_cache_for_both_tiers(config, calendar):
    specs, quotes = build_universe(n=10)
    clients = FakeClients(config, specs, quotes)
    cache = MarketDataCache(contracts=TtlCache(900, clock=FakeClock()),
                            quotes=TtlCache(8, clock=FakeClock()))
    args = dict(symbol="SPY", spot=100.0, atr=2.0, realized_vol=0.25,
                calendar=calendar, order_session=SESSION, option_type="call")
    run_prefilter(clients, cache=cache, **args)
    requests, symbols = len(clients.contract_requests), len(clients.snapshot_symbols)
    run_prefilter(clients, cache=cache, **args)
    assert len(clients.contract_requests) == requests
    assert len(clients.snapshot_symbols) == symbols
