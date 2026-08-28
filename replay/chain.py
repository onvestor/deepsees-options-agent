"""A synthetic option chain, shaped like the one Alpaca returns.

The prefilter takes ``ContractSpec`` and ``OptionQuote`` and does not care
where they came from. That is the seam this module fills: it builds both from
an underlying price and a volatility, so :func:`src.options.prefilter.assemble`
runs offline against exactly the code path a live session uses.

**The model decides what survives, and that has to stay visible.** Open
interest, spread width and implied volatility are all generated here from a
handful of parameters. Every prefilter gate the survivors clear -- the open
interest floor, the spread ceiling, the minimum bid -- they clear because
:class:`ChainModel` said so. So a replay tells you whether the pipeline
narrows, chooses, sizes and exits correctly. It does not tell you whether the
market would have offered those contracts, and a survivor count from replay is
not evidence about a real chain.

**Parameters live here, not in ``config/``.** ``config/`` holds values that
change *trading* behaviour, and a rule that everything configurable belongs
there would put simulation knobs next to live risk limits, where changing one
by accident is a real loss. These change what a simulation reports, so they are
an explicit argument, defaulted in code, and echoed into every replay report so
a number can always be traced to the model that produced it.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from typing import Any, Sequence

from src.brokers.alpaca.contracts import ContractSpec
from src.brokers.alpaca.quotes import OptionQuote
from src.options.occ import build as build_occ
from replay.pricing import OptionType, black_scholes, years_between


@dataclass(frozen=True)
class ChainModel:
    """How the synthetic chain prices and quotes itself.

    Defaults are deliberately unexciting: a chain that is liquid near the money
    and thins in the wings, with spreads that widen the further out you go.
    They are a starting point for making the pipeline run, not a calibration.
    """

    # --- volatility ---
    iv_to_realized_ratio: float = 1.15
    """Implied over realized. Above 1.0 because long premium usually pays for
    more movement than arrives -- the same asymmetry ``iv_vs_rv20`` measures."""

    iv_skew: float = 0.35
    """Smile steepness. IV rises with the log-distance of the strike from spot."""

    min_volatility: float = 0.08
    """Floor. A realized vol of zero in a flat synthetic series would otherwise
    produce a division by zero rather than an option."""

    # --- the strike ladder ---
    strike_step: float | None = None
    """Spacing between strikes. ``None`` derives it from spot; see
    :func:`strike_step_for`."""

    # --- liquidity ---
    atm_open_interest: int = 6_000
    oi_width: float = 0.06
    """Log-moneyness at which open interest falls to 1/e of its at-the-money
    value. Wider means a chain that stays liquid further out."""

    # --- quotes ---
    base_spread_pct: float = 0.010
    spread_widening: float = 0.45
    """Added spread per unit of absolute log-moneyness."""

    min_tick: float = 0.01

    # --- rates ---
    rate: float = 0.04
    dividend_yield: float = 0.0

    def as_report(self) -> dict[str, Any]:
        """The parameters, for the replay report. A P&L number without these
        is not interpretable."""
        return asdict(self)


def strike_step_for(spot: float) -> float:
    """A plausible strike ladder for a given price level.

    Real ladders are per-symbol and set by the exchange; this is an
    approximation that keeps the number of strikes inside the prefilter's
    window in a sane range at any price. It matters less than it looks: the
    prefilter selects on delta, and a finer ladder mostly changes how many
    contracts get evaluated.
    """
    if spot < 25.0:
        return 1.0
    if spot < 100.0:
        return 2.5
    if spot < 400.0:
        return 5.0
    return 10.0


@dataclass(frozen=True)
class SyntheticChain:
    """One session's chain for one symbol and one option type."""

    specs: tuple[ContractSpec, ...]
    quotes: dict[str, OptionQuote]

    def quote_for(self, symbol: str) -> OptionQuote | None:
        return self.quotes.get(symbol)


def _implied_vol(model: ChainModel, realized_vol: float, log_moneyness: float) -> float:
    base = max(realized_vol, 0.0) * model.iv_to_realized_ratio
    smile = 1.0 + model.iv_skew * abs(log_moneyness)
    return max(model.min_volatility, base * smile)


def _open_interest(model: ChainModel, log_moneyness: float) -> int:
    if model.oi_width <= 0:
        return model.atm_open_interest
    decay = math.exp(-((log_moneyness / model.oi_width) ** 2))
    return int(round(model.atm_open_interest * decay))


def _round_to_tick(value: float, tick: float) -> float:
    return round(round(value / tick) * tick, 2)


def _strikes(spot: float, low: float, high: float, step: float) -> list[float]:
    """The ladder inside a window, anchored on a round multiple of the step.

    Anchored rather than centred on spot, because a real ladder does not move
    with the price -- and a strike set that shifted every session would make
    a position's own strike vanish from the chain the next day.
    """
    first = math.ceil(low / step) * step
    out: list[float] = []
    strike = first
    while strike <= high + 1e-9:
        if strike > 0:
            out.append(round(strike, 2))
        strike += step
    return out


def build_chain(
    *,
    symbol: str,
    spot: float,
    realized_vol: float,
    expiry: date,
    session: date,
    option_type: OptionType,
    strike_gte: float,
    strike_lte: float,
    model: ChainModel | None = None,
    quote_ts: datetime | None = None,
) -> SyntheticChain:
    """Build the contracts and quotes the prefilter would have been handed.

    ``strike_gte``/``strike_lte`` are the prefilter's own window, so this
    returns what the contracts endpoint would return for that request -- the
    narrowing stays in the prefilter rather than being duplicated here.
    """
    model = model or ChainModel()
    step = model.strike_step or strike_step_for(spot)
    stamp = quote_ts or datetime.combine(
        session, datetime.min.time(), tzinfo=timezone.utc
    )
    days = (expiry - session).days
    t = years_between(days)

    specs: list[ContractSpec] = []
    quotes: dict[str, OptionQuote] = {}

    for strike in _strikes(spot, strike_gte, strike_lte, step):
        occ = build_occ(symbol, expiry, option_type, strike)
        log_moneyness = math.log(strike / spot)
        vol = _implied_vol(model, realized_vol, log_moneyness)
        greeks = black_scholes(
            option_type=option_type,
            spot=spot,
            strike=strike,
            years_to_expiry=t,
            volatility=vol,
            rate=model.rate,
            dividend_yield=model.dividend_yield,
        )

        spread_pct = model.base_spread_pct + model.spread_widening * abs(log_moneyness)
        half = max(model.min_tick / 2.0, greeks.price * spread_pct / 2.0)
        bid = _round_to_tick(max(0.0, greeks.price - half), model.min_tick)
        ask = _round_to_tick(greeks.price + half, model.min_tick)
        if ask <= bid:
            # A contract too cheap to quote two-sidedly. Left with a zero bid so
            # the prefilter rejects it on its own min_bid rule rather than being
            # filtered out here -- the replay should exercise that gate, not
            # hide it.
            bid = 0.0
            ask = max(ask, model.min_tick)

        oi = _open_interest(model, log_moneyness)
        specs.append(
            ContractSpec(
                symbol=occ,
                underlying=symbol,
                root=symbol,
                expiry=expiry,
                strike=strike,
                option_type=option_type,
                style="american",
                open_interest=oi,
                open_interest_date=session,
                close_price=round(greeks.price, 2),
                close_price_date=session,
                size=100,
                tradable=True,
                status="active",
            )
        )
        quotes[occ] = OptionQuote(
            symbol=occ,
            bid=bid,
            ask=ask,
            bid_size=10.0,
            ask_size=10.0,
            quote_ts=stamp,
            delta=greeks.delta,
            gamma=greeks.gamma,
            theta=greeks.theta,
            vega=greeks.vega,
            rho=greeks.rho,
            implied_volatility=vol,
            # Stands in for "this contract traded today". The prefilter's volume
            # gate reads it, so a chain with no last trade would fail every
            # contract that also had zero open interest.
            last_trade_price=round(greeks.price, 2) if oi > 0 else None,
            last_trade_ts=stamp if oi > 0 else None,
        )

    return SyntheticChain(specs=tuple(specs), quotes=quotes)


def reprice(
    *,
    spec: ContractSpec,
    spot: float,
    realized_vol: float,
    session: date,
    model: ChainModel | None = None,
    quote_ts: datetime | None = None,
) -> OptionQuote:
    """Re-quote one known contract on a later session.

    Marking an open position needs the same contract repriced, not a fresh
    chain: the position is in a specific strike and expiry, and rebuilding the
    ladder would silently mark it against whatever the new window happened to
    contain.
    """
    model = model or ChainModel()
    stamp = quote_ts or datetime.combine(
        session, datetime.min.time(), tzinfo=timezone.utc
    )
    log_moneyness = math.log(spec.strike / spot)
    vol = _implied_vol(model, realized_vol, log_moneyness)
    greeks = black_scholes(
        option_type=spec.option_type,
        spot=spot,
        strike=spec.strike,
        years_to_expiry=years_between((spec.expiry - session).days),
        volatility=vol,
        rate=model.rate,
        dividend_yield=model.dividend_yield,
    )
    spread_pct = model.base_spread_pct + model.spread_widening * abs(log_moneyness)
    half = max(model.min_tick / 2.0, greeks.price * spread_pct / 2.0)
    bid = _round_to_tick(max(0.0, greeks.price - half), model.min_tick)
    ask = _round_to_tick(greeks.price + half, model.min_tick)
    if ask <= bid:
        bid, ask = 0.0, max(ask, model.min_tick)

    return OptionQuote(
        symbol=spec.symbol,
        bid=bid,
        ask=ask,
        bid_size=10.0,
        ask_size=10.0,
        quote_ts=stamp,
        delta=greeks.delta,
        gamma=greeks.gamma,
        theta=greeks.theta,
        vega=greeks.vega,
        rho=greeks.rho,
        implied_volatility=vol,
        last_trade_price=round(greeks.price, 2),
        last_trade_ts=stamp,
    )


def specs_by_symbol(chain: SyntheticChain) -> dict[str, ContractSpec]:
    return {spec.symbol: spec for spec in chain.specs}


def in_window(
    specs: Sequence[ContractSpec], strike_gte: float, strike_lte: float
) -> tuple[ContractSpec, ...]:
    """What the contracts endpoint would have returned for a strike window."""
    return tuple(s for s in specs if strike_gte <= s.strike <= strike_lte)
