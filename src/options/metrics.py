"""The six computed metrics, as pure functions.

**No I/O, no network.** Every input arrives as a number; the module never
fetches anything. Enforced by ``tests/test_signals_purity.py``, which covers
this file as well as ``src/signals/``.

The six metrics exist to turn a chain into numbers a model can reason about
without being handed prices to compute with. Each one answers a question that
is easy to get wrong by eye:

1. ``theta_pct_per_day``      -- what fraction of the premium decays per day?
2. ``gamma_per_1pct``         -- how much delta is gained per 1% underlying move?
3. ``iv_vs_rv``               -- is implied vol rich or cheap against realized?
4. ``spread_cost_pct_of_atr`` -- what fraction of one ATR of option movement
                                 does crossing the spread cost?
5. ``breakeven_distance_atr`` -- how many ATRs must the underlying travel to
                                 break even?
6. ``modeled_pnl_1atr``       -- modeled P&L on a 1-ATR move over the hold,
                                 net of decay.

Metric 4 is the one that most often kills an otherwise attractive contract:
a $0.07 spread looks trivial until you notice the contract only moves $0.21 on
a full-ATR underlying move, at which point entry and exit together consume
two-thirds of the edge.

Conventions:

* All prices are **per share**, not per contract. The 100x multiplier cancels
  in every ratio here, and introducing it would only create a units bug.
* ``theta`` and ``delta`` are taken with whatever sign the feed supplies;
  magnitudes are taken explicitly where magnitude is what matters, so a put's
  negative delta does not silently flip a comparison.
* Nothing here defaults a missing input. A metric that cannot be computed
  raises, and the prefilter rejects the contract. Fail closed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Literal, Sequence

__all__ = [
    "ContractMetrics",
    "MetricError",
    "compute_metrics",
    "realized_volatility",
    "TRADING_DAYS_PER_YEAR",
]

OptionType = Literal["call", "put"]

TRADING_DAYS_PER_YEAR = 252


class MetricError(ValueError):
    """An input needed for a metric is missing or degenerate."""


def realized_volatility(
    closes: Sequence[float],
    window: int,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised close-to-close realised volatility over ``window`` returns.

    Uses log returns and the sample standard deviation. Returns a fraction
    (0.28 == 28% annualised), matching the units Alpaca reports IV in, so the
    ratio in :func:`compute_metrics` is dimensionless.

    Raises rather than returning NaN on insufficient history -- a realised vol
    computed from four bars is not a number worth comparing IV against.
    """
    if window < 2:
        raise MetricError(f"window must be >= 2, got {window}")
    prices = [float(c) for c in closes if c is not None and float(c) > 0]
    if len(prices) < window + 1:
        raise MetricError(
            f"need {window + 1} positive closes for a {window}-period realised vol, "
            f"got {len(prices)}"
        )

    recent = prices[-(window + 1) :]
    returns = [math.log(recent[i + 1] / recent[i]) for i in range(len(recent) - 1)]
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return math.sqrt(variance) * math.sqrt(periods_per_year)


@dataclass(frozen=True)
class ContractMetrics:
    """The six metrics plus the inputs that produced them.

    Inputs are echoed so the decision log explains itself: a record that says
    ``modeled_pnl_1atr = 0.11`` without the ATR and delta behind it cannot be
    audited months later.
    """

    # --- the six ---
    theta_pct_per_day: float
    gamma_per_1pct: float
    iv_vs_rv: float
    spread_cost_pct_of_atr: float
    breakeven_distance_atr: float
    modeled_pnl_1atr: float

    # --- ranking key ---
    pnl_to_spread_ratio: float

    # --- inputs, echoed ---
    spot: float
    atr: float
    strike: float
    premium: float
    spread: float
    delta: float
    gamma: float
    theta: float
    implied_volatility: float
    realized_volatility: float
    hold_hours: float

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}

    @property
    def is_finite(self) -> bool:
        """Acceptance requires every metric populated with no NaNs."""
        return all(math.isfinite(v) for v in self.as_dict().values())


def compute_metrics(
    *,
    option_type: OptionType,
    strike: float,
    spot: float,
    atr: float,
    bid: float,
    ask: float,
    delta: float | None,
    gamma: float | None,
    theta: float | None,
    implied_volatility: float | None,
    realized_vol: float,
    hold_hours: float,
    theta_day_hours: float,
) -> ContractMetrics:
    """Compute all six metrics for one contract.

    Every argument is required. ``delta``, ``gamma``, ``theta`` and
    ``implied_volatility`` are typed optional only because that is how they
    arrive from the feed -- passing ``None`` raises. The prefilter rejects such
    contracts before reaching here; this is the second line of that defence.
    """
    for name, value in (
        ("delta", delta), ("gamma", gamma), ("theta", theta),
        ("implied_volatility", implied_volatility),
    ):
        if value is None:
            raise MetricError(f"{name} is missing -- contract is not scoreable")

    if atr <= 0:
        raise MetricError(f"atr must be positive, got {atr}")
    if spot <= 0:
        raise MetricError(f"spot must be positive, got {spot}")
    if realized_vol <= 0:
        raise MetricError(f"realized_vol must be positive, got {realized_vol}")
    if bid <= 0 or ask < bid:
        raise MetricError(f"unusable quote: bid={bid} ask={ask}")
    if theta_day_hours <= 0 or hold_hours < 0:
        raise MetricError(f"bad hold window: hold_hours={hold_hours} day={theta_day_hours}")

    premium = (bid + ask) / 2.0
    spread = ask - bid
    if premium <= 0:
        raise MetricError(f"premium must be positive, got {premium}")

    abs_delta = abs(float(delta))
    if abs_delta <= 0:
        raise MetricError("delta is zero -- contract cannot respond to the underlying")

    # 1. Decay as a fraction of what you paid, per day.
    theta_pct_per_day = abs(float(theta)) / premium

    # 2. Delta gained per 1% underlying move. Gamma is per $1, so scale by 1%
    #    of spot to get something comparable across a $70 and a $700 name.
    gamma_per_1pct = float(gamma) * (spot * 0.01)

    # 3. Rich or cheap. Both sides annualised fractions, so dimensionless.
    iv_vs_rv = float(implied_volatility) / realized_vol

    # 4. One ATR of underlying moves the option roughly delta x ATR. The
    #    spread is what you pay to get in. Crossing it twice -- in and out --
    #    is why an intraday vertical often loses to a single leg.
    atr_implied_option_move = abs_delta * atr
    spread_cost_pct_of_atr = spread / atr_implied_option_move

    # 5. Breakeven measured in ATRs: how far must it actually travel?
    if option_type == "call":
        breakeven = strike + ask
        distance = breakeven - spot
    elif option_type == "put":
        breakeven = strike - ask
        distance = spot - breakeven
    else:
        raise MetricError(f"unknown option_type {option_type!r}")
    breakeven_distance_atr = distance / atr

    # 6. Modeled P&L on a favourable 1-ATR move held for `hold_hours`.
    #    Second-order in the underlying (gamma helps a long option), minus
    #    decay over the hold. Entry cost is the half-spread paid at the ask.
    directional = abs_delta * atr + 0.5 * float(gamma) * (atr ** 2)
    decay = abs(float(theta)) * (hold_hours / theta_day_hours)
    modeled_pnl_1atr = directional - decay

    # Ranking key: edge earned per unit of spread paid. Ranking on modeled
    # P&L alone would favour expensive contracts whose spread eats the gain.
    pnl_to_spread_ratio = modeled_pnl_1atr / spread if spread > 0 else math.inf

    return ContractMetrics(
        theta_pct_per_day=theta_pct_per_day,
        gamma_per_1pct=gamma_per_1pct,
        iv_vs_rv=iv_vs_rv,
        spread_cost_pct_of_atr=spread_cost_pct_of_atr,
        breakeven_distance_atr=breakeven_distance_atr,
        modeled_pnl_1atr=modeled_pnl_1atr,
        pnl_to_spread_ratio=pnl_to_spread_ratio,
        spot=float(spot),
        atr=float(atr),
        strike=float(strike),
        premium=premium,
        spread=spread,
        delta=float(delta),
        gamma=float(gamma),
        theta=float(theta),
        implied_volatility=float(implied_volatility),
        realized_volatility=float(realized_vol),
        hold_hours=float(hold_hours),
    )
