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
from src.brokers.alpaca.calendar import DteError, TradingCalendar
from src.brokers.alpaca.client import AlpacaClients
from src.brokers.alpaca.contracts import ContractSpec, fetch as fetch_contracts
from src.brokers.alpaca.quotes import OptionQuote, fetch_snapshots
from src.options.expiry import (
    MONTHLY, WEEKLY, ExpiryType, classify, next_monthly_at_least,
)
from src.options.metrics import (
    ContractMetrics,
    MetricError,
    compute_metrics,
    modeled_hold_hours,
)

log = logging.getLogger(__name__)

__all__ = ["Candidate", "PrefilterResult", "ScanPlan", "plan_scan",
           "assemble", "run_prefilter", "REASONS"]

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
    "weekly expiry",
    "expired",
    "not tradable",
    "unscoreable",
    # Metric acceptance bands. Applied only when
    # prefilter.apply_metric_bands is true, and always last -- they need the
    # metrics, which are only computed for a contract that cleared everything
    # else.
    "theta too high",
    "gamma too low",
    "iv rich",
    "spread cost vs premium",
    "breakeven too far",
    "modeled pnl too low",
)


@dataclass(frozen=True)
class Candidate:
    """One contract, its quote, every filter verdict, and its metrics."""

    spec: ContractSpec
    quote: OptionQuote
    session_dte: int
    expiry_type: ExpiryType = WEEKLY
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
    def is_monthly(self) -> bool:
        return self.expiry_type == MONTHLY

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
    target_expiry: date | None = None
    target_session_dte: int | None = None

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
    delta_min = limits.get_float("prefilter.delta_min")
    delta_max = limits.get_float("prefilter.delta_max")
    dte_min = limits.get_int("prefilter.dte_min")
    dte_max = limits.get_int("prefilter.dte_max")
    require_monthly = limits.get_bool("prefilter.require_monthly_expiry")
    hold_hours = modeled_hold_hours(limits)
    theta_day_hours = limits.get_float("metrics.theta_day_hours")
    apply_bands = limits.get_bool("prefilter.apply_metric_bands")

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
        elif spread_pct > max_spread_pct:
            # Percentage only. A dollar cap scales with nothing: at 37 DTE and
            # 0.55-0.75 delta the premiums here are $15-$55, so a $0.35 cap
            # rejected 1.0-2.6% quotes -- 54 contracts on SPY, 15 on IWM, and
            # every survivor AMD had. Measured 2026-08-25: across three
            # symbols exactly one contract failed the percentage test alone,
            # so the dollar cap was the entire filter and it was the wrong one.
            failures.append("spread")
            distances.append(_bounded_ratio(spread_pct, max_spread_pct))

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

        # Expiry type is a liquidity proxy, not a preference: at matched
        # strikes a weekly can carry two orders of magnitude less open
        # interest than the monthly beside it. Classified always, enforced
        # only when configured.
        expiry_type = classify(spec.expiry, calendar.is_session)
        if require_monthly and expiry_type != MONTHLY:
            failures.append("weekly expiry")

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

        # The acceptance bands run last, on a contract that cleared every
        # structural gate and scored. Metrics are kept on the candidate even
        # when a band rejects it -- the near-boundary report is the whole
        # reason these are worth recording, and it needs the values.
        if metrics is not None and apply_bands:
            for reason, value, limit, over in (
                ("theta too high", metrics.theta_pct_per_day,
                 limits.get_float("metrics.max_theta_pct_per_day"), True),
                ("gamma too low", metrics.gamma_per_1pct,
                 limits.get_float("metrics.min_gamma_per_1pct"), False),
                ("iv rich", metrics.iv_vs_rv,
                 limits.get_float("metrics.max_iv_vs_rv_ratio"), True),
                ("spread cost vs premium", metrics.spread_pct_of_premium,
                 limits.get_float("metrics.max_spread_cost_pct_of_premium"), True),
                ("breakeven too far", metrics.breakeven_move_pct,
                 limits.get_float("metrics.max_breakeven_move_pct"), True),
                ("modeled pnl too low", metrics.pnl_to_spread_ratio,
                 limits.get_float("metrics.min_modeled_pnl_ratio"), False),
            ):
                if over and value > limit:
                    failures.append(reason)
                    distances.append(_bounded_ratio(value, limit))
                elif not over and value < limit:
                    failures.append(reason)
                    distances.append(_bounded_ratio(limit - value, limit))

        out.append(
            Candidate(
                spec=spec,
                quote=quote,
                session_dte=session_dte,
                expiry_type=expiry_type,
                failures=tuple(failures),
                metrics=metrics,
                boundary_distance=min(distances) if len(failures) == 1 and distances else None,
            )
        )
    return out


@dataclass(frozen=True)
class ScanPlan:
    """Where to look, decided before anything is fetched.

    Extracted so the replay harness selects the expiry by the same rule the
    live path does. The expiry rule is the piece most likely to drift between
    a live scan and an offline one, and a replay that scanned a different
    expiry than production would be measuring the wrong contract.
    """

    target_expiry: date
    target_session_dte: int
    expiry_gte: date
    expiry_lte: date
    strike_gte: float
    strike_lte: float


def plan_scan(
    limits: Any,
    calendar: TradingCalendar,
    order_session: date,
    symbol: str,
    spot: float,
) -> ScanPlan:
    """Choose the expiry and the strike window. Raises if the guard band fails.

    The expiry is chosen, not bounded. A fixed calendar-day window inside a
    ~30-day monthly cycle misses the monthly about half the time; anchoring on
    the nearest monthly at least ``monthly_min_sessions`` out always lands on
    the liquid contract. DTE therefore varies per trade and is recorded on the
    result rather than assumed from config.
    """
    dte_min = limits.get_int("prefilter.dte_min")
    dte_max = limits.get_int("prefilter.dte_max")
    window = limits.get_float("prefilter.strike_window_pct")

    target_expiry = next_monthly_at_least(
        order_session,
        limits.get_int("prefilter.monthly_min_sessions"),
        lambda d: calendar.sessions_until(d, order_session),
        calendar.is_session,
    )
    target_dte = calendar.sessions_until(target_expiry, order_session)
    if target_dte < dte_min or target_dte > dte_max:
        raise DteError(
            f"{symbol}: nearest monthly {target_expiry} is {target_dte} sessions out, "
            f"outside the guard band [{dte_min}, {dte_max}]. Refusing to scan."
        )
    return ScanPlan(
        target_expiry=target_expiry,
        target_session_dte=target_dte,
        expiry_gte=target_expiry,
        expiry_lte=target_expiry,
        strike_gte=round(spot * (1.0 - window), 2),
        strike_lte=round(spot * (1.0 + window), 2),
    )


def assemble(
    symbol: str,
    specs: Sequence[ContractSpec],
    quotes: dict[str, OptionQuote],
    calendar: TradingCalendar,
    order_session: date,
    spot: float,
    atr: float,
    realized_vol: float,
    limits: Any,
    plan: ScanPlan,
) -> PrefilterResult:
    """Filter, score, rank and cap. Everything after the data arrives.

    Split from :func:`run_prefilter` so the offline replay harness runs the
    *same* ranking, coverage accounting and threshold record as a live session.
    Duplicating any of it would let replay and production disagree silently,
    which is the one thing a replay harness must not do.
    """
    max_survivors = limits.get_int("prefilter.max_survivors_per_symbol")
    prefer_monthly = limits.get_bool("prefilter.prefer_monthly_expiry")

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
    # Monthly first when preferred, then modelled P&L per unit of spread cost
    # within each group. A monthly never loses to a weekly on rank alone,
    # because the weekly's edge is usually a quote that will not survive the
    # round trip.
    survivors = tuple(sorted(
        (c for c in candidates if c.survived),
        key=lambda c: (-(c.is_monthly and prefer_monthly), -c.rank_key),
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
        expiry_gte=plan.expiry_gte, expiry_lte=plan.expiry_lte,
        strike_gte=plan.strike_gte, strike_lte=plan.strike_lte,
        candidates=tuple(candidates),
        survivors=survivors,
        top=top,
        narrowed_coverage=narrowed_coverage,
        target_expiry=plan.target_expiry,
        target_session_dte=plan.target_session_dte,
        thresholds={
            "strike_window_pct": limits.get_float("prefilter.strike_window_pct"),
            "dte_min": float(limits.get_int("prefilter.dte_min")),
            "dte_max": float(limits.get_int("prefilter.dte_max")),
            "monthly_min_sessions": float(limits.get_int("prefilter.monthly_min_sessions")),
            "target_session_dte": float(plan.target_session_dte),
            "delta_min": limits.get_float("prefilter.delta_min"),
            "delta_max": limits.get_float("prefilter.delta_max"),
            "min_open_interest": float(limits.get_int("prefilter.min_open_interest")),
            "min_bid": limits.get_float("prefilter.min_bid"),
            "max_spread_pct_of_mid": limits.get_float("prefilter.max_spread_pct_of_mid"),
            "max_survivors_per_symbol": float(max_survivors),
            "require_monthly_expiry": float(limits.get_bool("prefilter.require_monthly_expiry")),
            "prefer_monthly_expiry": float(prefer_monthly),
            "apply_metric_bands": float(limits.get_bool("prefilter.apply_metric_bands")),
            "max_theta_pct_per_day": limits.get_float("metrics.max_theta_pct_per_day"),
            "min_gamma_per_1pct": limits.get_float("metrics.min_gamma_per_1pct"),
            "max_iv_vs_rv_ratio": limits.get_float("metrics.max_iv_vs_rv_ratio"),
            "max_spread_cost_pct_of_premium": limits.get_float(
                "metrics.max_spread_cost_pct_of_premium"
            ),
            "max_breakeven_move_pct": limits.get_float("metrics.max_breakeven_move_pct"),
            "min_modeled_pnl_ratio": limits.get_float("metrics.min_modeled_pnl_ratio"),
        },
    )


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
    the chosen monthly, strike bounds from a window around spot, and only what
    survives both is ever sent to the snapshots endpoint.
    """
    limits = clients.config.limits
    plan = plan_scan(limits, calendar, order_session, symbol, spot)

    specs = fetch_contracts(
        clients, symbol,
        expiry_gte=plan.expiry_gte, expiry_lte=plan.expiry_lte,
        option_type=option_type,
        strike_gte=plan.strike_gte, strike_lte=plan.strike_lte,
        cache=cache,
    )
    log.info(
        "%s: target monthly expiry %s at %d sessions",
        symbol, plan.target_expiry, plan.target_session_dte,
    )
    log.info(
        "%s: narrowed universe %d contracts (%s..%s, strikes %.2f..%.2f "
        "= +/-%.0f%% of %.2f)",
        symbol, len(specs), plan.expiry_gte, plan.expiry_lte,
        plan.strike_gte, plan.strike_lte,
        limits.get_float("prefilter.strike_window_pct") * 100, spot,
    )

    quotes = fetch_snapshots(clients, [s.symbol for s in specs], cache=cache)

    return assemble(
        symbol, specs, quotes, calendar, order_session, spot, atr, realized_vol,
        limits, plan,
    )
