"""Option snapshots: quotes and greeks.

This is the second of the two calls a chain needs. The contracts endpoint
gives strikes and open interest; everything the prefilter actually filters on
-- bid, ask, spread, delta -- comes from here.

On the Basic plan the options feed is ``indicative``. Two consequences worth
holding onto:

* Greeks are **not** returned for every contract. A live Saturday probe found
  them on 542/836 SPY contracts and 45/51 NVDA, with the gaps not following
  any moneyness boundary, and ``implied_volatility`` missing on exactly the
  same contracts. :attr:`OptionQuote.has_greeks` is therefore a real question
  the prefilter must ask, not a formality.
* A missing greek is represented as ``None``, never as ``0.0``. A delta of
  zero is a meaningful value -- a far OTM contract -- and conflating it with
  "unknown" would let un-scoreable contracts through the delta band test.

Caching is **per symbol**, not per request. A 300-contract chain rescanned a
few seconds later then refetches only the entries that actually expired.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Sequence

from alpaca.data.requests import OptionSnapshotRequest

from src.brokers.alpaca.cache import MarketDataCache
from src.brokers.alpaca.client import AlpacaClients, with_retry

log = logging.getLogger(__name__)

__all__ = ["OptionQuote", "fetch_snapshots"]


@dataclass(frozen=True)
class OptionQuote:
    """A quote plus greeks for one contract, at one moment.

    Every greek is ``float | None``. ``None`` means the feed did not supply
    it; it never means zero.
    """

    symbol: str
    bid: float | None
    ask: float | None
    bid_size: float | None
    ask_size: float | None
    quote_ts: datetime | None
    delta: float | None
    gamma: float | None
    theta: float | None
    vega: float | None
    rho: float | None
    implied_volatility: float | None
    last_trade_price: float | None
    last_trade_ts: datetime | None

    # -- derived ------------------------------------------------------------

    @property
    def has_quote(self) -> bool:
        """A two-sided, non-crossed quote. Anything else is unusable."""
        return (
            self.bid is not None
            and self.ask is not None
            and self.bid > 0
            and self.ask > 0
            and self.ask >= self.bid
        )

    @property
    def has_greeks(self) -> bool:
        return self.delta is not None

    @property
    def mid(self) -> float | None:
        if not self.has_quote:
            return None
        return (self.bid + self.ask) / 2.0  # type: ignore[operator]

    @property
    def spread(self) -> float | None:
        if not self.has_quote:
            return None
        return self.ask - self.bid  # type: ignore[operator]

    @property
    def spread_pct_of_mid(self) -> float | None:
        mid, spread = self.mid, self.spread
        if mid is None or spread is None or mid <= 0:
            return None
        return spread / mid

    @classmethod
    def from_snapshot(cls, symbol: str, snapshot: Any) -> "OptionQuote":
        quote = getattr(snapshot, "latest_quote", None)
        greeks = getattr(snapshot, "greeks", None)
        trade = getattr(snapshot, "latest_trade", None)

        def greek(name: str) -> float | None:
            if greeks is None:
                return None
            value = getattr(greeks, name, None)
            return float(value) if value is not None else None

        return cls(
            symbol=symbol,
            bid=_number(getattr(quote, "bid_price", None)),
            ask=_number(getattr(quote, "ask_price", None)),
            bid_size=_number(getattr(quote, "bid_size", None)),
            ask_size=_number(getattr(quote, "ask_size", None)),
            quote_ts=getattr(quote, "timestamp", None),
            delta=greek("delta"),
            gamma=greek("gamma"),
            theta=greek("theta"),
            vega=greek("vega"),
            rho=greek("rho"),
            implied_volatility=_number(getattr(snapshot, "implied_volatility", None)),
            last_trade_price=_number(getattr(trade, "price", None)),
            last_trade_ts=getattr(trade, "timestamp", None),
        )

    @classmethod
    def missing(cls, symbol: str) -> "OptionQuote":
        """The snapshot endpoint returned nothing for this symbol.

        Cached like any other answer -- "no data" is a real result and
        refetching it every few seconds helps nobody.
        """
        return cls(
            symbol=symbol, bid=None, ask=None, bid_size=None, ask_size=None, quote_ts=None,
            delta=None, gamma=None, theta=None, vega=None, rho=None,
            implied_volatility=None, last_trade_price=None, last_trade_ts=None,
        )


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fetch_snapshots(
    clients: AlpacaClients,
    symbols: Sequence[str],
    cache: MarketDataCache | None = None,
) -> dict[str, OptionQuote]:
    """Quotes and greeks for ``symbols``, batched and cached per symbol."""
    unique = list(dict.fromkeys(s.strip().upper() for s in symbols if s))
    if not unique:
        return {}

    config = clients.config
    batch_size = config.limits.get_int("broker.snapshot_batch_size")
    feed = clients.options_feed

    results: dict[str, OptionQuote] = {}
    to_fetch: list[str] = unique

    if cache is not None:
        cached, missing = cache.quotes.get_many(unique)
        results.update({str(k): v for k, v in cached.items()})
        to_fetch = [str(k) for k in missing]
        if cached:
            log.debug("quotes cache: %d hit, %d to fetch", len(cached), len(to_fetch))

    fresh: dict[str, OptionQuote] = {}
    for start in range(0, len(to_fetch), batch_size):
        batch = to_fetch[start : start + batch_size]
        request = OptionSnapshotRequest(symbol_or_symbols=batch, feed=feed)
        page = with_retry(
            config,
            f"option_snapshots({len(batch)} symbols)",
            lambda r=request: clients.options.get_option_snapshot(r),
        )
        for symbol in batch:
            snapshot = page.get(symbol)
            fresh[symbol] = (
                OptionQuote.from_snapshot(symbol, snapshot)
                if snapshot is not None
                else OptionQuote.missing(symbol)
            )

    if cache is not None and fresh:
        cache.quotes.put_many(dict(fresh))
    results.update(fresh)

    if to_fetch:
        with_greeks = sum(1 for s in to_fetch if results[s].has_greeks)
        log.info(
            "snapshots: %d fetched, %d with greeks (%.0f%%), %d cached",
            len(to_fetch), with_greeks,
            100.0 * with_greeks / len(to_fetch) if to_fetch else 0.0,
            len(unique) - len(to_fetch),
        )
    return results


def greeks_coverage(quotes: Iterable[OptionQuote]) -> dict[str, Any]:
    """How much of a chain is scoreable. Reported, never silently worked around."""
    quotes = list(quotes)
    total = len(quotes)
    if not total:
        return {"total": 0, "with_greeks": 0, "with_quote": 0, "coverage": 0.0}
    with_greeks = sum(1 for q in quotes if q.has_greeks)
    return {
        "total": total,
        "with_greeks": with_greeks,
        "missing_greeks": total - with_greeks,
        "with_quote": sum(1 for q in quotes if q.has_quote),
        "coverage": round(with_greeks / total, 4),
    }
