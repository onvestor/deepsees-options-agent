"""Deterministic chain prefilter: narrow first, then score.

**Order of operations matters more than the filters do.**

The naive shape -- fetch the whole chain, snapshot everything, filter after --
buys a large quantity of missing greeks and then throws them away. Measured on
a live Friday chain: 21 days of expiries across the full strike range returned
greeks on 58% of SPY and 47% of NVDA, with the gaps concentrated in the wings
and the far expiries. Neither region can produce a survivor, because a
contract inside the delta band is near the money by construction.

So the universe is narrowed by **DTE band and a strike window around spot
before a single snapshot is requested**. The delta band still does the precise
work; the strike window just stops us paying for chain we cannot use. The
window is configurable and starts at +/-10% of spot.

Everything downstream is unchanged from the multi-label design: every filter
is evaluated against every contract and all failing reasons are recorded, so
the breakdown answers "how many contracts fail this test" rather than "which
test happened to run first".

**Missing delta is a hard reject, never a default.** A contract the feed
cannot price is a contract we cannot size, and substituting a guess is exactly
the class of silent error the fail-closed rule exists to prevent. Implied
volatility is treated the same way -- it is missing on precisely the same
contracts, and every survivor must have all six metrics populated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable, Literal, Sequence

from src.brokers.alpaca.cache import MarketDataCache
from src.brokers.alpaca.calendar import TradingCalendar
from src.brokers.alpaca.client import AlpacaClients
from src.brokers.alpaca.contracts import ContractSpec, fetch as fetch_contracts
from src.brokers.alpaca.quotes import OptionQuote, fetch_snapshots
from src.options.metrics import ContractMetrics, MetricError, compute_metrics

log = logging.getLogger(__name__)

__all__ = ["Candidate", "PrefilterResult", "run_prefilter", "REASONS"]

OptionType = Literal["call", "put"]

# Every test the prefilter applies. Named here so the log's reason vocabulary
# is a closed set a dashboard can rely on rather than whatever strings the
# code happened to emit.
REASONS = (
    "no quote",
    "crossed/empty quote",
    "bid below floor",
    "open interest",
    "volume",
    "spread",
    "no greeks",
    "no delta",
    "no iv",
    "delta band",
    "session dte",
    "expired",
    "not tradable",
    "unscoreable",
)


@dataclass(frozen=True)
class Candidate:
    """One contract, its quote, every filter verdict, and its metrics."""

    spec: ContractSpec
    quote: OptionQuote
    session_dte: int
    failures: tuple[str, ...] = ()
    metrics: ContractMetrics | None = None
    boundary_distance: float | None = field(
        default=None,
        metadata={"doc": "How close a single-reason reject was to passing, 0.0-1.0."},
    )

    @property
    def symbol(self) -> str:
        return self.spec.symbol

    @property
    def survived(self) -> bool:
        return not self.failures

    @property
    def rank_key(self) -> float:
        return self.metrics.pnl_to_spread_ratio if self.metrics else float("-inf")


@dataclass(frozen=True)
class PrefilterResult:
    """The full outcome. The survivor set is the decision; the rest is evidence."""

    symbol: str
    spot: float
    atr: float
    realized_vol: float
    expiry_gte: date
    expiry_lte: date
    strike_gte: float
    strike_lte: float
    candidates: tuple[Candidate, ...]
    survivors: tuple[Candidate, ...]
    top: tuple[Candidate, ...]
    narrowed_coverage: dict[str, Any]
    thresholds: dict[str, float]

    @property
    def reason_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for candidate in self.candidates:
            for reason in candidate.failures:
                counts[reason] = counts.get(reason, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    @property
    def sole_reason(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for candidate in self.candidates:
            if len(candidate.failures) == 1:
                counts[candidate.failures[0]] = counts.get(candidate.failures[0], 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def near_boundary(self, within: float) -> tuple[Candidate, ...]:
        """Single-reason rejects that came within ``within`` of passing.

        These are the only rejects worth recording individually: they are the
        population a threshold change would actually move. Everything else is
        adequately described by the aggregate counts.
        """
        return tuple(
            c
            for c in self.candidates
            if len(c.failures) == 1
            and c.boundary_distance is not None
            and c.boundary_distance <= within
        )


def _bounded_ratio(value: float, limit: float) -> float:
    """How far past a ceiling a value sits, as a fraction. 0.0 == exactly at it."""
    if limit <= 0:
        return 1.0
    return max(0.0, (value - limit) / limit)


def evaluate_candidates(
    specs: Sequence[ContractSpec],
    quotes: dict[str, OptionQuote],
    calendar: TradingCalendar,
    order_session: date,
    spot: float,
    atr: float,
    realized_vol: float,
    limits: Any,
) -> list[Candidate]:
    """Apply every filter to every contract, then score the survivors.

    Pure with respect to the network: takes the data, returns the verdicts.
    """
    min_oi = limits.get_int("prefilter.min_open_interest")
    min_volume = limits.get_int("prefilter.min_volume")
    min_bid = limits.get_float("prefilter.min_bid")
    max_spread_pct = limits.get_float("prefilter.max_spread_pct_of_mid")
    max_spread_abs = limits.get_float("prefilter.max_spread_abs")
    delta_min = limits.get_float("prefilter.delta_min")
    delta_max = limits.get_float("prefilter.delta_max")
    dte_min = limits.get_int("prefilter.dte_min")
    dte_max = limits.get_int("prefilter.dte_max")
    hold_hours = limits.get_float("metrics.modeled_hold_hours")
    theta_day_hours = limits.get_float("metrics.theta_day_hours")

    out: list[Candidate] = []
    for spec in specs:
        quote = quotes.get(spec.symbol) or OptionQuote.missing(spec.symbol)
        failures: list[str] = []
        distances: list[float] = []

        if not spec.tradable:
            failures.append("not tradable")

        if quote.quote_ts is None and quote.bid is None:
            failures.append("no quote")
        if not quote.has_quote:
            failures.append("crossed/empty quote")

        bid = quote.bid or 0.0
        if bid < min_bid:
            failures.append("bid below floor")
            distances.append(_bounded_ratio(min_bid - bid, min_bid))

        if spec.open_interest < min_oi:
            failures.append("open interest")
            distances.append(_bounded_ratio(min_oi - spec.open_interest, min_oi))

        volume = quote.last_trade_price is not None
        if min_volume > 0 and not volume and spec.open_interest == 0:
            failures.append("volume")

        spread = quote.spread
        spread_pct = quote.spread_pct_of_mid
        if spread is None or spread_pct is None:
            if "crossed/empty quote" not in failures:
                failures.append("spread")
        else:
            over_abs = spread > max_spread_abs
            over_pct = spread_pct > max_spread_pct
            if over_abs or over_pct:
                failures.append("spread")
                distances.append(
                    min(
                        _bounded_ratio(spread, max_spread_abs),
                        _bounded_ratio(spread_pct, max_spread_pct),
                    )
                )

        if quote.delta is None:
            # Hard reject. Never defaulted, never inferred from moneyness.
            failures.append("no greeks")
            failures.append("no delta")
        else:
            magnitude = abs(quote.delta)
            if magnitude < delta_min:
                failures.append("delta band")
                distances.append(_bounded_ratio(delta_min - magnitude, delta_min))
            elif magnitude > delta_max:
                failures.append("delta band")
                distances.append(_bounded_ratio(magnitude, delta_max))

        if quote.implied_volatility is None:
            failures.append("no iv")

        session_dte = calendar.sessions_until(spec.expiry, order_session)
        if session_dte < 0:
            failures.append("expired")
        elif session_dte < dte_min or session_dte > dte_max:
            failures.append("session dte")

        metrics: ContractMetrics | None = None
        if not failures:
            try:
                metrics = compute_metrics(
                    option_type=spec.option_type,
                    strike=spec.strike,
                    spot=spot,
                    atr=atr,
                    bid=quote.bid,          # type: ignore[arg-type]
                    ask=quote.ask,          # type: ignore[arg-type]
                    delta=quote.delta,
                    gamma=quote.gamma,
                    theta=quote.theta,
                    implied_volatility=quote.implied_volatility,
                    realized_vol=realized_vol,
                    hold_hours=hold_hours,
                    theta_day_hours=theta_day_hours,
                )
            except MetricError as exc:
                # A contract that passed every gate but still cannot be scored
                # is rejected, not shipped with a partial metric set.
                log.debug("%s unscoreable: %s", spec.symbol, exc)
                failures.append("unscoreable")

        out.append(
            Candidate(
                spec=spec,
                quote=quote,
                session_dte=session_dte,
                failures=tuple(failures),
                metrics=metrics,
                boundary_distance=min(distances) if len(failures) == 1 and distances else None,
            )
        )
    return out


def run_prefilter(
    clients: AlpacaClients,
    symbol: str,
    spot: float,
    atr: float,
    realized_vol: float,
    calendar: TradingCalendar,
    order_session: date,
    option_type: OptionType,
    cache: MarketDataCache | None = None,
) -> PrefilterResult:
    """Narrow the universe, fetch quotes for it, filter, score, rank, cap.

    The narrowing is the whole point of the ordering: expiry bounds come from
    the trading-session DTE band, strike bounds from a window around spot, and
    only what survives both is ever sent to the snapshots endpoint.
    """
    limits = clients.config.limits
    dte_min = limits.get_int("prefilter.dte_min")
    dte_max = limits.get_int("prefilter.dte_max")
    window = limits.get_float("prefilter.strike_window_pct")
    max_survivors = limits.get_int("prefilter.max_survivors_per_symbol")

    expiry_gte = calendar.session_offset(order_session, dte_min)
    expiry_lte = calendar.session_offset(order_session, dte_max)
    strike_gte = round(spot * (1.0 - window), 2)
    strike_lte = round(spot * (1.0 + window), 2)

    specs = fetch_contracts(
        clients, symbol,
        expiry_gte=expiry_gte, expiry_lte=expiry_lte,
        option_type=option_type,
        strike_gte=strike_gte, strike_lte=strike_lte,
        cache=cache,
    )
    log.info(
        "%s: narrowed universe %d contracts (%s..%s, strikes %.2f..%.2f = +/-%.0f%% of %.2f)",
        symbol, len(specs), expiry_gte, expiry_lte, strike_gte, strike_lte, window * 100, spot,
    )

    quotes = fetch_snapshots(clients, [s.symbol for s in specs], cache=cache)

    # Coverage measured INSIDE the narrowed window -- the population the
    # prefilter actually sees. Reported separately from any wide-chain figure,
    # because the two answer different questions.
    with_greeks = sum(1 for s in specs if quotes[s.symbol].has_greeks)
    narrowed_coverage = {
        "total": len(specs),
        "with_greeks": with_greeks,
        "missing_greeks": len(specs) - with_greeks,
        "coverage": round(with_greeks / len(specs), 4) if specs else 0.0,
        "with_quote": sum(1 for s in specs if quotes[s.symbol].has_quote),
        "window": "narrowed",
    }

    candidates = evaluate_candidates(
        specs, quotes, calendar, order_session, spot, atr, realized_vol, limits
    )
    survivors = tuple(sorted(
        (c for c in candidates if c.survived), key=lambda c: -c.rank_key
    ))

    # Only the top N reach a prompt. More candidates makes a model's decision
    # worse, not better -- and the full survivor set stays in the log either
    # way, so nothing is lost to analysis.
    top = survivors[:max_survivors]
    if len(survivors) > len(top):
        log.info(
            "%s: %d survivors, top %d handed on (ranked by modeled P&L / spread cost)",
            symbol, len(survivors), len(top),
        )

    return PrefilterResult(
        symbol=symbol,
        spot=spot, atr=atr, realized_vol=realized_vol,
        expiry_gte=expiry_gte, expiry_lte=expiry_lte,
        strike_gte=strike_gte, strike_lte=strike_lte,
        candidates=tuple(candidates),
        survivors=survivors,
        top=top,
        narrowed_coverage=narrowed_coverage,
        thresholds={
            "strike_window_pct": window,
            "dte_min": float(dte_min),
            "dte_max": float(dte_max),
            "delta_min": limits.get_float("prefilter.delta_min"),
            "delta_max": limits.get_float("prefilter.delta_max"),
            "min_open_interest": float(limits.get_int("prefilter.min_open_interest")),
            "min_bid": limits.get_float("prefilter.min_bid"),
            "max_spread_pct_of_mid": limits.get_float("prefilter.max_spread_pct_of_mid"),
            "max_spread_abs": limits.get_float("prefilter.max_spread_abs"),
            "max_survivors_per_symbol": float(max_survivors),
        },
    )
