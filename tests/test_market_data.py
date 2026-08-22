"""Contract discovery and snapshot parsing, tested against recorded responses.

Per CLAUDE.md: parsers are tested against real recorded Alpaca responses, not
hand-written JSON, because the shape surprises are the whole point. The
snapshot fixture deliberately includes contracts with no greeks -- 46 of 120 --
because that is what the feed actually returns.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.brokers.alpaca.cache import MarketDataCache, TtlCache
from src.brokers.alpaca.client import BrokerError
from src.brokers.alpaca.contracts import ContractSpec, fetch, group_by_expiry
from src.brokers.alpaca.quotes import OptionQuote, fetch_snapshots, greeks_coverage
from tests.test_cache import FakeClock

FIXTURES = Path(__file__).parent / "fixtures"
CONTRACTS = json.loads((FIXTURES / "option_contracts.json").read_text(encoding="utf-8"))["contracts"]
SNAPSHOTS = json.loads((FIXTURES / "option_snapshots.json").read_text(encoding="utf-8"))["snapshots"]


def as_api_contract(row: dict) -> SimpleNamespace:
    """Rebuild the object shape alpaca-py hands back."""
    return SimpleNamespace(
        symbol=row["symbol"],
        underlying_symbol=row["underlying_symbol"],
        root_symbol=row["root_symbol"],
        expiration_date=date.fromisoformat(row["expiration_date"]),
        strike_price=row["strike_price"],
        type=row["type"],
        style=row["style"],
        open_interest="100",
        open_interest_date=None,
        close_price=None,
        close_price_date=None,
        size="100",
        tradable=True,
        status="active",
    )


def as_api_snapshot(row: dict) -> SimpleNamespace | None:
    if not row["snapshot_present"]:
        return None
    quote = row["latest_quote"]
    greeks = row["greeks"]
    trade = row["latest_trade"]
    return SimpleNamespace(
        latest_quote=None if quote is None else SimpleNamespace(
            bid_price=quote["bid_price"], ask_price=quote["ask_price"],
            bid_size=quote["bid_size"], ask_size=quote["ask_size"],
            timestamp=datetime.fromisoformat(quote["timestamp"]) if quote["timestamp"] else None,
        ),
        greeks=None if greeks is None else SimpleNamespace(**greeks),
        implied_volatility=row["implied_volatility"],
        latest_trade=None if trade is None else SimpleNamespace(
            price=trade["price"],
            timestamp=datetime.fromisoformat(trade["timestamp"]) if trade["timestamp"] else None,
        ),
    )


# --- ContractSpec ----------------------------------------------------------


@pytest.mark.parametrize("row", CONTRACTS[:80], ids=lambda r: r["symbol"])
def test_contract_parses_and_agrees_with_its_own_symbol(row):
    spec = ContractSpec.from_api(as_api_contract(row))
    assert spec.symbol == row["symbol"]
    assert spec.root == row["root_symbol"]
    assert spec.strike == pytest.approx(float(row["strike_price"]))
    assert spec.expiry == date.fromisoformat(row["expiration_date"])
    assert spec.option_type == row["type"]


def test_contract_rejects_a_symbol_that_disagrees_with_its_fields():
    """The cross-check is the point: a symbol/field mismatch must never be
    absorbed silently, because it would size a position on the wrong strike."""
    row = dict(CONTRACTS[0])
    api = as_api_contract(row)
    api.strike_price = "999.0"
    with pytest.raises(BrokerError, match="disagrees with fields"):
        ContractSpec.from_api(api)


def test_contract_rejects_an_unparseable_symbol():
    api = as_api_contract(dict(CONTRACTS[0]))
    api.symbol = "NOT-AN-OCC-SYMBOL"
    with pytest.raises(BrokerError, match="unparseable"):
        ContractSpec.from_api(api)


def test_group_by_expiry_sorts_and_partitions():
    specs = [ContractSpec.from_api(as_api_contract(r)) for r in CONTRACTS[:60]]
    grouped = group_by_expiry(specs)
    assert sum(len(v) for v in grouped.values()) == len(specs)
    for items in grouped.values():
        assert [c.strike for c in items] == sorted(c.strike for c in items)
    assert list(grouped) == sorted(grouped)


# --- pagination ------------------------------------------------------------


class FakeTrading:
    """Serves the fixture in pages, exactly as the real endpoint does."""

    def __init__(self, rows, page_size=100, loop=False):
        self.rows = rows
        self.page_size = page_size
        self.loop = loop
        self.requests = []

    def get_option_contracts(self, request):
        self.requests.append(request)
        start = int(request.page_token or 0)
        chunk = self.rows[start : start + self.page_size]
        nxt = start + self.page_size
        token = None
        if self.loop:
            token = "0"                       # always the same -> loop
        elif nxt < len(self.rows):
            token = str(nxt)
        return SimpleNamespace(
            option_contracts=[as_api_contract(r) for r in chunk], next_page_token=token
        )


def make_clients(config, trading):
    return SimpleNamespace(config=config, trading=trading)


@pytest.fixture
def config():
    from src.config import load_config

    return load_config()


def test_fetch_follows_pagination_to_the_end(config):
    trading = FakeTrading(CONTRACTS, page_size=100)
    specs = fetch(make_clients(config, trading), "SPY",
                  expiry_gte=date(2026, 8, 24), expiry_lte=date(2026, 12, 31))
    assert len(specs) == len(CONTRACTS)
    assert len(trading.requests) == 4          # 320 rows / 100 per page


def test_every_page_carries_explicit_date_bounds(config):
    """Alpaca defaults expiration_date_lte to next weekend. Every request, not
    just the first, must pin both bounds or later pages truncate."""
    trading = FakeTrading(CONTRACTS, page_size=100)
    fetch(make_clients(config, trading), "SPY",
          expiry_gte=date(2026, 8, 24), expiry_lte=date(2026, 12, 31))
    for request in trading.requests:
        assert request.expiration_date_gte == date(2026, 8, 24)
        assert request.expiration_date_lte == date(2026, 12, 31)


@pytest.mark.parametrize("gte,lte", [(None, date(2026, 12, 31)), (date(2026, 8, 24), None)])
def test_missing_date_bounds_raise(config, gte, lte):
    with pytest.raises(ValueError, match="required"):
        fetch(make_clients(config, FakeTrading([])), "SPY", expiry_gte=gte, expiry_lte=lte)


def test_inverted_date_bounds_raise(config):
    with pytest.raises(ValueError, match="before"):
        fetch(make_clients(config, FakeTrading([])), "SPY",
              expiry_gte=date(2026, 9, 1), expiry_lte=date(2026, 8, 1))


def test_pagination_loop_is_detected(config):
    trading = FakeTrading(CONTRACTS, page_size=10, loop=True)
    with pytest.raises(BrokerError, match="pagination loop"):
        fetch(make_clients(config, trading), "SPY",
              expiry_gte=date(2026, 8, 24), expiry_lte=date(2026, 12, 31))


def test_contract_cache_prevents_a_second_fetch(config):
    trading = FakeTrading(CONTRACTS, page_size=100)
    cache = MarketDataCache(contracts=TtlCache(900, clock=FakeClock()),
                            quotes=TtlCache(8, clock=FakeClock()))
    clients = make_clients(config, trading)
    args = dict(expiry_gte=date(2026, 8, 24), expiry_lte=date(2026, 12, 31))

    first = fetch(clients, "SPY", cache=cache, **args)
    calls = len(trading.requests)
    second = fetch(clients, "SPY", cache=cache, **args)

    assert len(trading.requests) == calls        # no new requests
    assert [c.symbol for c in first] == [c.symbol for c in second]


def test_different_bounds_are_different_cache_entries(config):
    trading = FakeTrading(CONTRACTS, page_size=100)
    cache = MarketDataCache(contracts=TtlCache(900, clock=FakeClock()),
                            quotes=TtlCache(8, clock=FakeClock()))
    clients = make_clients(config, trading)
    fetch(clients, "SPY", expiry_gte=date(2026, 8, 24), expiry_lte=date(2026, 12, 31), cache=cache)
    calls = len(trading.requests)
    fetch(clients, "SPY", expiry_gte=date(2026, 8, 24), expiry_lte=date(2026, 11, 30), cache=cache)
    assert len(trading.requests) > calls


# --- OptionQuote -----------------------------------------------------------


@pytest.mark.parametrize("row", SNAPSHOTS, ids=lambda r: r["symbol"])
def test_snapshot_parses(row):
    quote = OptionQuote.from_snapshot(row["symbol"], as_api_snapshot(row))
    assert quote.symbol == row["symbol"]
    if row["greeks"] and row["greeks"]["delta"] is not None:
        assert quote.delta == pytest.approx(row["greeks"]["delta"])
        assert quote.has_greeks
    else:
        assert quote.delta is None
        assert not quote.has_greeks


def test_the_fixture_actually_contains_missing_greeks():
    """If this ever fails the fixture stopped covering the case that matters."""
    missing = [r for r in SNAPSHOTS if not r["greeks"]]
    assert missing, "fixture must include contracts with no greeks"
    assert len(missing) / len(SNAPSHOTS) > 0.1


def test_missing_greek_is_none_never_zero():
    """A delta of 0.0 is a real far-OTM value. Conflating it with 'unknown'
    would let un-scoreable contracts through the delta band test."""
    row = next(r for r in SNAPSHOTS if not r["greeks"])
    quote = OptionQuote.from_snapshot(row["symbol"], as_api_snapshot(row))
    assert quote.delta is None
    assert quote.delta != 0.0


def test_derived_quote_maths():
    quote = OptionQuote.missing("X")
    assert quote.mid is None and quote.spread is None and quote.spread_pct_of_mid is None

    real = OptionQuote(
        symbol="X", bid=1.00, ask=1.20, bid_size=1, ask_size=1, quote_ts=None,
        delta=0.5, gamma=None, theta=None, vega=None, rho=None,
        implied_volatility=None, last_trade_price=None, last_trade_ts=None,
    )
    assert real.mid == pytest.approx(1.10)
    assert real.spread == pytest.approx(0.20)
    assert real.spread_pct_of_mid == pytest.approx(0.20 / 1.10)
    assert real.has_quote and real.has_greeks


@pytest.mark.parametrize(
    "bid, ask",
    [(None, 1.0), (1.0, None), (0.0, 1.0), (1.0, 0.0), (1.5, 1.0)],   # last is crossed
)
def test_unusable_quotes_are_rejected(bid, ask):
    quote = OptionQuote(
        symbol="X", bid=bid, ask=ask, bid_size=None, ask_size=None, quote_ts=None,
        delta=None, gamma=None, theta=None, vega=None, rho=None,
        implied_volatility=None, last_trade_price=None, last_trade_ts=None,
    )
    assert not quote.has_quote


def test_greeks_coverage_reports_the_gap():
    quotes = [OptionQuote.from_snapshot(r["symbol"], as_api_snapshot(r)) for r in SNAPSHOTS]
    coverage = greeks_coverage(quotes)
    assert coverage["total"] == len(SNAPSHOTS)
    assert coverage["with_greeks"] + coverage["missing_greeks"] == coverage["total"]
    assert 0.0 < coverage["coverage"] < 1.0


# --- snapshot fetching + cache --------------------------------------------


class FakeOptions:
    def __init__(self, rows):
        self.by_symbol = {r["symbol"]: as_api_snapshot(r) for r in rows}
        self.batches = []

    def get_option_snapshot(self, request):
        symbols = request.symbol_or_symbols
        self.batches.append(list(symbols))
        return {s: self.by_symbol[s] for s in symbols if self.by_symbol.get(s) is not None}


def make_data_clients(config, options):
    return SimpleNamespace(
        config=config, options=options,
        options_feed=config.limits.get_str("broker.data_feed_options"),
    )


def test_fetch_snapshots_batches(config):
    options = FakeOptions(SNAPSHOTS)
    clients = make_data_clients(config, options)
    symbols = [r["symbol"] for r in SNAPSHOTS]
    quotes = fetch_snapshots(clients, symbols)
    assert set(quotes) == set(symbols)
    assert all(len(b) <= config.limits.get_int("broker.snapshot_batch_size") for b in options.batches)


def test_fetch_snapshots_caches_per_symbol(config):
    clock = FakeClock()
    cache = MarketDataCache(contracts=TtlCache(900, clock=clock), quotes=TtlCache(8, clock=clock))
    options = FakeOptions(SNAPSHOTS)
    clients = make_data_clients(config, options)
    symbols = [r["symbol"] for r in SNAPSHOTS]

    fetch_snapshots(clients, symbols, cache=cache)
    first_batches = len(options.batches)

    fetch_snapshots(clients, symbols, cache=cache)          # all fresh
    assert len(options.batches) == first_batches

    clock.advance(9)                                        # all stale
    fetch_snapshots(clients, symbols, cache=cache)
    assert len(options.batches) > first_batches


def test_only_expired_symbols_are_refetched(config):
    """The reason quotes cache per symbol rather than per request."""
    clock = FakeClock()
    cache = MarketDataCache(contracts=TtlCache(900, clock=clock), quotes=TtlCache(8, clock=clock))
    options = FakeOptions(SNAPSHOTS)
    clients = make_data_clients(config, options)
    symbols = [r["symbol"] for r in SNAPSHOTS]

    fetch_snapshots(clients, symbols[:60], cache=cache)
    clock.advance(4)
    options.batches.clear()
    fetch_snapshots(clients, symbols, cache=cache)          # 60 fresh, 60 new

    fetched = [s for batch in options.batches for s in batch]
    assert set(fetched) == set(symbols[60:])


def test_absent_symbol_yields_a_missing_quote(config):
    options = FakeOptions(SNAPSHOTS)
    clients = make_data_clients(config, options)
    quotes = fetch_snapshots(clients, ["SPY999999C99999000"])
    assert quotes["SPY999999C99999000"].has_quote is False
    assert quotes["SPY999999C99999000"].has_greeks is False


def test_duplicate_symbols_are_requested_once(config):
    options = FakeOptions(SNAPSHOTS)
    clients = make_data_clients(config, options)
    symbol = SNAPSHOTS[0]["symbol"]
    fetch_snapshots(clients, [symbol, symbol, symbol.lower()])
    assert [s for b in options.batches for s in b] == [symbol]


def test_empty_symbol_list_makes_no_request(config):
    options = FakeOptions(SNAPSHOTS)
    assert fetch_snapshots(make_data_clients(config, options), []) == {}
    assert options.batches == []
