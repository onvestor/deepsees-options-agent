"""Two-tier TTL cache for market data.

The two tiers exist because the two datasets age at completely different
rates, and a single TTL is wrong for both at once:

* **Contract universe** (strikes, expiries, open interest) changes on the
  order of minutes. Refetching it per scan wastes a paginated multi-request
  round trip for data that has not moved.
* **Quotes and greeks** are stale within seconds. A quote cached for a minute
  is not a quote, it is a rumour, and sizing a position against it means
  entering on a price that no longer exists.

CLAUDE.md is explicit that a single 60s cache is simultaneously too aggressive
for the universe and far too stale for entry decisions. Hence two caches, two
TTLs, configured separately.

The clock is injectable. Cache expiry that can only be tested by sleeping is
cache expiry that does not get tested.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Hashable, Iterable, TypeVar

from src.config import Config

log = logging.getLogger(__name__)

T = TypeVar("T")

__all__ = ["CacheStats", "MarketDataCache", "TtlCache"]


@dataclass
class CacheStats:
    """Hit rate is the only way to know a TTL is doing anything useful."""

    hits: int = 0
    misses: int = 0
    expirations: int = 0
    evictions: int = 0
    stores: int = 0

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "expirations": self.expirations,
            "evictions": self.evictions,
            "stores": self.stores,
            "hit_rate": round(self.hit_rate, 4),
        }


@dataclass
class _Entry(Generic[T]):
    value: T
    stored_at: float


class TtlCache(Generic[T]):
    """A time-expiring key/value cache with an injectable clock.

    Uses ``time.monotonic`` rather than wall time, so a clock adjustment
    cannot make a cached quote appear fresh.
    """

    def __init__(
        self,
        ttl_seconds: float,
        name: str = "cache",
        max_entries: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_seconds < 0:
            raise ValueError(f"ttl_seconds must be >= 0, got {ttl_seconds}")
        self.ttl_seconds = float(ttl_seconds)
        self.name = name
        self.max_entries = max_entries
        self._clock = clock
        self._entries: dict[Hashable, _Entry[T]] = {}
        self._lock = threading.Lock()
        self.stats = CacheStats()

    # -- core ---------------------------------------------------------------

    def age_of(self, key: Hashable) -> float | None:
        """Seconds since the entry was stored, or None if absent."""
        with self._lock:
            entry = self._entries.get(key)
            return None if entry is None else self._clock() - entry.stored_at

    def get(self, key: Hashable) -> tuple[bool, T | None]:
        """``(hit, value)``. Explicit rather than a sentinel, because ``None``
        is a legitimate cached value for "this symbol has no quote"."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.stats.misses += 1
                return False, None
            if self._clock() - entry.stored_at >= self.ttl_seconds:
                del self._entries[key]
                self.stats.expirations += 1
                self.stats.misses += 1
                return False, None
            self.stats.hits += 1
            return True, entry.value

    def put(self, key: Hashable, value: T) -> None:
        with self._lock:
            if (
                self.max_entries is not None
                and key not in self._entries
                and len(self._entries) >= self.max_entries
            ):
                oldest = min(self._entries, key=lambda k: self._entries[k].stored_at)
                del self._entries[oldest]
                self.stats.evictions += 1
            self._entries[key] = _Entry(value=value, stored_at=self._clock())
            self.stats.stores += 1

    def get_or_fetch(self, key: Hashable, fetch: Callable[[], T]) -> T:
        hit, value = self.get(key)
        if hit:
            return value  # type: ignore[return-value]
        fresh = fetch()
        self.put(key, fresh)
        return fresh

    # -- bulk, for the per-symbol quote tier ---------------------------------

    def get_many(self, keys: Iterable[Hashable]) -> tuple[dict[Hashable, T], list[Hashable]]:
        """Split keys into cached hits and the misses still to be fetched.

        This is what makes the quote tier worth having: a 300-symbol chain
        rescanned 4 seconds later fetches only the handful that expired,
        instead of all 300 or none.
        """
        found: dict[Hashable, T] = {}
        missing: list[Hashable] = []
        for key in keys:
            hit, value = self.get(key)
            if hit:
                found[key] = value  # type: ignore[assignment]
            else:
                missing.append(key)
        return found, missing

    def put_many(self, items: dict[Hashable, T]) -> None:
        for key, value in items.items():
            self.put(key, value)

    # -- maintenance ---------------------------------------------------------

    def invalidate(self, key: Hashable | None = None) -> None:
        """Drop one key, or everything. Used when a write makes reads stale."""
        with self._lock:
            if key is None:
                self._entries.clear()
            else:
                self._entries.pop(key, None)

    def purge_expired(self) -> int:
        with self._lock:
            now = self._clock()
            stale = [k for k, e in self._entries.items() if now - e.stored_at >= self.ttl_seconds]
            for key in stale:
                del self._entries[key]
            self.stats.expirations += len(stale)
            return len(stale)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: Hashable) -> bool:
        hit, _ = self.get(key)
        return hit

    def __repr__(self) -> str:
        return (
            f"TtlCache(name={self.name!r}, ttl={self.ttl_seconds}s, "
            f"entries={len(self._entries)}, hit_rate={self.stats.hit_rate:.0%})"
        )


@dataclass
class MarketDataCache:
    """The two tiers, held together so callers cannot accidentally share one."""

    contracts: TtlCache[Any]
    quotes: TtlCache[Any]

    @classmethod
    def from_config(
        cls, config: Config, clock: Callable[[], float] = time.monotonic
    ) -> "MarketDataCache":
        limits = config.limits
        contracts_ttl = limits.get_int("cache.contracts_ttl_seconds")
        quotes_ttl = limits.get_int("cache.quotes_ttl_seconds")

        # A quote tier at or above the contract tier means the two-tier design
        # has been silently collapsed into one. Fail loudly instead.
        if quotes_ttl >= contracts_ttl:
            raise ValueError(
                f"cache.quotes_ttl_seconds ({quotes_ttl}) must be well below "
                f"cache.contracts_ttl_seconds ({contracts_ttl}) -- quotes and the "
                "contract universe age at different rates and must not share a TTL"
            )

        return cls(
            contracts=TtlCache(
                ttl_seconds=contracts_ttl,
                name="contracts",
                max_entries=limits.get_int("cache.contracts_max_entries"),
                clock=clock,
            ),
            quotes=TtlCache(
                ttl_seconds=quotes_ttl,
                name="quotes",
                max_entries=limits.get_int("cache.quotes_max_entries"),
                clock=clock,
            ),
        )

    def invalidate_all(self) -> None:
        self.contracts.invalidate()
        self.quotes.invalidate()

    def stats(self) -> dict[str, dict[str, float]]:
        return {"contracts": self.contracts.stats.as_dict(), "quotes": self.quotes.stats.as_dict()}
