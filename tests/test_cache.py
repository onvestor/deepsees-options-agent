"""Two-tier cache tests. The clock is injected, so nothing here sleeps."""

from __future__ import annotations

import pytest

from src.brokers.alpaca.cache import MarketDataCache, TtlCache


class FakeClock:
    """A clock we can advance. Cache expiry tested by sleeping is untested."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_hit_before_expiry():
    clock = FakeClock()
    cache = TtlCache(ttl_seconds=10, clock=clock)
    cache.put("k", "v")
    clock.advance(9.9)
    assert cache.get("k") == (True, "v")


def test_miss_after_expiry():
    clock = FakeClock()
    cache = TtlCache(ttl_seconds=10, clock=clock)
    cache.put("k", "v")
    clock.advance(10.0)
    assert cache.get("k") == (False, None)
    assert cache.stats.expirations == 1


def test_expiry_is_exclusive_at_the_boundary():
    """At exactly ttl the entry is stale. Quotes must not live one tick too long."""
    clock = FakeClock()
    cache = TtlCache(ttl_seconds=8, clock=clock)
    cache.put("k", "v")
    clock.advance(7.999)
    assert cache.get("k")[0] is True
    clock.advance(0.001)
    assert cache.get("k")[0] is False


def test_none_is_a_cacheable_value():
    """`(hit, value)` rather than a sentinel: 'no quote for this symbol' is a
    real answer worth caching, and must not read as a miss."""
    cache = TtlCache(ttl_seconds=10, clock=FakeClock())
    cache.put("k", None)
    assert cache.get("k") == (True, None)


def test_get_or_fetch_calls_once_then_serves_cache():
    calls = []

    def fetch():
        calls.append(1)
        return "value"

    clock = FakeClock()
    cache = TtlCache(ttl_seconds=10, clock=clock)
    assert cache.get_or_fetch("k", fetch) == "value"
    assert cache.get_or_fetch("k", fetch) == "value"
    assert len(calls) == 1

    clock.advance(11)
    assert cache.get_or_fetch("k", fetch) == "value"
    assert len(calls) == 2


def test_get_many_splits_hits_from_misses():
    """The point of the per-symbol quote tier: refetch only what expired."""
    clock = FakeClock()
    cache = TtlCache(ttl_seconds=10, clock=clock)
    cache.put("a", 1)
    clock.advance(6)
    cache.put("b", 2)
    clock.advance(5)                       # a is 11s old, b is 5s old

    found, missing = cache.get_many(["a", "b", "c"])
    assert found == {"b": 2}
    assert missing == ["a", "c"]


def test_put_many_and_len():
    cache = TtlCache(ttl_seconds=10, clock=FakeClock())
    cache.put_many({"a": 1, "b": 2})
    assert len(cache) == 2


def test_max_entries_evicts_oldest():
    clock = FakeClock()
    cache = TtlCache(ttl_seconds=100, max_entries=2, clock=clock)
    cache.put("a", 1)
    clock.advance(1)
    cache.put("b", 2)
    clock.advance(1)
    cache.put("c", 3)
    assert len(cache) == 2
    assert cache.get("a")[0] is False
    assert cache.get("c")[0] is True
    assert cache.stats.evictions == 1


def test_overwriting_a_key_does_not_evict():
    cache = TtlCache(ttl_seconds=100, max_entries=2, clock=FakeClock())
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("a", 9)
    assert len(cache) == 2
    assert cache.get("a") == (True, 9)
    assert cache.stats.evictions == 0


def test_invalidate_one_and_all():
    cache = TtlCache(ttl_seconds=100, clock=FakeClock())
    cache.put_many({"a": 1, "b": 2})
    cache.invalidate("a")
    assert cache.get("a")[0] is False and cache.get("b")[0] is True
    cache.invalidate()
    assert len(cache) == 0


def test_purge_expired():
    clock = FakeClock()
    cache = TtlCache(ttl_seconds=10, clock=clock)
    cache.put("a", 1)
    clock.advance(11)
    cache.put("b", 2)
    assert cache.purge_expired() == 1
    assert len(cache) == 1


def test_age_of():
    clock = FakeClock()
    cache = TtlCache(ttl_seconds=100, clock=clock)
    cache.put("k", 1)
    clock.advance(3.5)
    assert cache.age_of("k") == pytest.approx(3.5)
    assert cache.age_of("absent") is None


def test_stats_track_hit_rate():
    cache = TtlCache(ttl_seconds=100, clock=FakeClock())
    cache.put("k", 1)
    cache.get("k")
    cache.get("nope")
    assert cache.stats.hits == 1 and cache.stats.misses == 1
    assert cache.stats.hit_rate == pytest.approx(0.5)


def test_negative_ttl_rejected():
    with pytest.raises(ValueError):
        TtlCache(ttl_seconds=-1)


# --- the two tiers ---------------------------------------------------------


def test_two_tiers_come_from_config_with_different_ttls():
    from src.config import load_config

    cache = MarketDataCache.from_config(load_config(), clock=FakeClock())
    assert cache.contracts.ttl_seconds > cache.quotes.ttl_seconds
    assert cache.quotes.ttl_seconds <= 10       # quotes stale within seconds
    assert cache.contracts.ttl_seconds >= 600   # universe measured in minutes


def limits_with(contracts_ttl: int, quotes_ttl: int):
    """A real Section over a synthetic config, rather than a patched one."""
    from pathlib import Path
    from types import SimpleNamespace

    from src.config import Section

    section = Section(
        {
            "cache": {
                "contracts_ttl_seconds": contracts_ttl,
                "quotes_ttl_seconds": quotes_ttl,
                "contracts_max_entries": 64,
                "quotes_max_entries": 8000,
            }
        },
        Path("synthetic-limits.yaml"),
    )
    return SimpleNamespace(limits=section)


@pytest.mark.parametrize("quotes_ttl", [900, 901])
def test_collapsing_the_tiers_into_one_ttl_is_refused(quotes_ttl):
    """A single TTL is simultaneously too slow for quotes and too fast for
    the universe. Refuse rather than quietly degrade."""
    with pytest.raises(ValueError, match="must not share a TTL"):
        MarketDataCache.from_config(limits_with(900, quotes_ttl))


def test_distinct_ttls_are_accepted():
    cache = MarketDataCache.from_config(limits_with(900, 8), clock=FakeClock())
    assert (cache.contracts.ttl_seconds, cache.quotes.ttl_seconds) == (900.0, 8.0)


def test_tiers_are_independent():
    clock = FakeClock()
    cache = MarketDataCache(
        contracts=TtlCache(900, "contracts", clock=clock),
        quotes=TtlCache(8, "quotes", clock=clock),
    )
    cache.contracts.put("chain", ["a", "b"])
    cache.quotes.put("SPY260825C00767000", "quote")

    clock.advance(30)                      # quotes stale, universe still fresh
    assert cache.quotes.get("SPY260825C00767000")[0] is False
    assert cache.contracts.get("chain")[0] is True

    assert set(cache.stats()) == {"contracts", "quotes"}
