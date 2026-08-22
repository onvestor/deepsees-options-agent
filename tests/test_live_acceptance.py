"""Step 2 acceptance, against the live Alpaca API. Read-only; places no orders.

Deselected by default -- see pytest.ini. Run deliberately:

    pytest -m live -v

The acceptance criterion is twofold: a round-trip test on a live symbol
passes, and a chain fetch returns contracts with populated bid/ask and delta.
Both are asserted here against whatever the API returns right now, rather than
against a fixture, because the point is to detect the day Alpaca changes shape.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.brokers.alpaca.cache import MarketDataCache
from src.brokers.alpaca.client import build_clients
from src.brokers.alpaca.contracts import fetch
from src.brokers.alpaca.quotes import fetch_snapshots, greeks_coverage
from src.config import load_config
from src.options.occ import build as build_occ, parse as parse_occ

pytestmark = pytest.mark.live

SYMBOL = "SPY"


@pytest.fixture(scope="module")
def clients():
    return build_clients(load_config())


@pytest.fixture(scope="module")
def cache():
    return MarketDataCache.from_config(load_config())


@pytest.fixture(scope="module")
def chain(clients, cache):
    """A real chain: contracts joined to live snapshots."""
    specs = fetch(
        clients,
        SYMBOL,
        expiry_gte=date.today() + timedelta(days=1),
        expiry_lte=date.today() + timedelta(days=21),
        option_type="call",
        cache=cache,
    )
    assert specs, "live chain fetch returned nothing"
    quotes = fetch_snapshots(clients, [s.symbol for s in specs], cache=cache)
    return specs, quotes


# --- OCC round trip on live symbols ----------------------------------------


def test_occ_round_trips_every_live_symbol(chain):
    """Alpaca's own symbols, rebuilt from their parsed parts, byte-identical."""
    specs, _ = chain
    for spec in specs:
        parsed = parse_occ(spec.symbol)
        assert parsed.root == spec.root
        assert build_occ(parsed.root, parsed.expiry, parsed.option_type, parsed.strike) == spec.symbol


def test_live_contracts_agree_with_their_own_symbols(chain):
    """ContractSpec.from_api already cross-checks; this proves it ran clean."""
    specs, _ = chain
    for spec in specs:
        parsed = parse_occ(spec.symbol)
        assert (parsed.strike, parsed.expiry, parsed.option_type) == (
            spec.strike, spec.expiry, spec.option_type,
        )


# --- chain fetch shape ------------------------------------------------------


def test_chain_spans_multiple_expiries_and_is_bounded(chain):
    specs, _ = chain
    expiries = {s.expiry for s in specs}
    assert len(expiries) > 1, "date bounds should span several expiries"
    assert min(expiries) >= date.today() + timedelta(days=1)
    assert max(expiries) <= date.today() + timedelta(days=21)


def test_chain_is_larger_than_one_page(chain):
    """Proves pagination actually ran rather than silently truncating."""
    specs, _ = chain
    assert len(specs) > 100


# --- the acceptance criterion ----------------------------------------------


def test_chain_returns_contracts_with_populated_bid_ask_and_delta(chain):
    """Acceptance: bid/ask AND delta populated on a real slice of the chain.

    Not asserted for every contract -- the indicative feed genuinely omits
    greeks on part of the chain, which is a documented limitation and not a
    failure. What must hold is that a usable, scoreable set exists.
    """
    specs, quotes = chain
    usable = [
        quotes[s.symbol]
        for s in specs
        if quotes[s.symbol].has_quote and quotes[s.symbol].has_greeks
    ]
    assert usable, "no contract came back with both a quote and a delta"
    assert len(usable) >= 20, f"only {len(usable)} fully-populated contracts"

    for quote in usable:
        assert quote.bid > 0 and quote.ask >= quote.bid
        assert quote.mid > 0 and quote.spread >= 0
        assert -1.0 <= quote.delta <= 1.0


def test_greeks_coverage_is_reported_not_assumed(chain):
    _, quotes = chain
    coverage = greeks_coverage(quotes.values())
    print(f"\n  greeks coverage: {coverage}")
    assert coverage["total"] > 0
    assert coverage["with_quote"] > 0


def test_no_greek_is_ever_zero_when_it_should_be_absent(chain):
    """A missing greek must be None. Zero is a real far-OTM delta."""
    _, quotes = chain
    for quote in quotes.values():
        if not quote.has_greeks:
            assert quote.delta is None
            assert quote.gamma is None or isinstance(quote.gamma, float)


# --- cache behaviour against the real API ----------------------------------


def test_second_chain_fetch_is_served_from_cache(clients, cache, chain):
    before = cache.contracts.stats.hits
    fetch(
        clients, SYMBOL,
        expiry_gte=date.today() + timedelta(days=1),
        expiry_lte=date.today() + timedelta(days=21),
        option_type="call", cache=cache,
    )
    assert cache.contracts.stats.hits == before + 1


def test_quote_cache_serves_a_repeat_within_ttl(clients, cache, chain):
    specs, _ = chain
    symbols = [s.symbol for s in specs[:50]]
    before = cache.quotes.stats.hits
    fetch_snapshots(clients, symbols, cache=cache)
    assert cache.quotes.stats.hits > before
