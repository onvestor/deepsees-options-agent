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
    "CONTRACT_MULTIPLIER",
    "ContractMetrics",
    "MetricError",
    "VerticalMetrics",
    "compute_metrics",
    "compute_vertical_metrics",
    "modeled_hold_hours",
    "realized_volatility",
    "TRADING_DAYS_PER_YEAR",
]

OptionType = Literal["call", "put"]

TRADING_DAYS_PER_YEAR = 252

# One US equity option contract controls 100 shares. This is a contract
# specification, not a tunable, so it lives here rather than in limits.yaml.
#
# Per-share and per-contract figures are kept explicitly separate throughout:
# every ratio is computed per share (where the multiplier cancels), and every
# dollar figure that sizing or the risk layer will consume is per contract
# (where it must not). Mixing the two by a factor of 100 is the single easiest
# way to size a position 100x wrong, so both forms are named and tested.
CONTRACT_MULTIPLIER = 100


class MetricError(ValueError):
    """An input needed for a metric is missing or degenerate."""


def modeled_hold_hours(limits) -> float:
    """The modelled hold, in hours, derived from the hold in sessions.

    There is deliberately no ``metrics.modeled_hold_hours`` key. It existed
    alongside ``modeled_hold_sessions`` and the two disagreed: 4.0 hours beside
    a 3-session hold, a leftover from the intraday design that survived the
    swing revision. The decay term is ``theta * hold_hours / theta_day_hours``,
    so the mismatch understated theta by a factor of eighteen in every modelled
    P&L the prefilter ranks on.

    One value, one place. ``theta_day_hours`` is the calendar basis Alpaca
    quotes theta on, so sessions convert through it rather than through market
    hours -- theta accrues over the weekend and a hold measured in trading
    sessions spans those weekends.
    """
    sessions = limits.get_int("metrics.modeled_hold_sessions")
    day_hours = limits.get_float("metrics.theta_day_hours")
    if sessions < 1:
        raise MetricError(
            f"metrics.modeled_hold_sessions must be >= 1, got {sessions}"
        )
    if day_hours <= 0:
        raise MetricError(
            f"metrics.theta_day_hours must be positive, got {day_hours}"
        )
    return float(sessions) * day_hours


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

    # --- derived, for the acceptance bands ---------------------------------

    @property
    def spread_pct_of_premium(self) -> float:
        """Quoted spread as a fraction of what the contract costs.

        Distinct from ``spread_cost_pct_of_atr``, which measures the spread
        against the option move one ATR of underlying produces. Those two
        disagree whenever delta is far from 1: a cheap far-OTM contract can
        look tight against its premium and ruinous against its ATR move.

        **It is, however, identical to the prefilter's own
        ``spread_pct_of_mid``** -- premium here is the mid, so the two are the
        same quantity under different names. That makes
        ``metrics.max_spread_cost_pct_of_premium`` a redundant backstop rather
        than an independent test: whichever of the two thresholds is tighter
        does all the work, and the structural gate runs first. Recorded rather
        than quietly dropped, because a band that cannot fire is exactly the
        thing this codebase already had six of.
        """
        return self.spread / self.premium if self.premium > 0 else math.inf

    @property
    def breakeven_move_pct(self) -> float:
        """Underlying move to breakeven, as a signed fraction of spot.

        Signed on purpose. A contract already past its breakeven has a
        negative distance, and that should clear a ceiling on how far the
        underlying still has to travel -- taking the absolute value would
        reject the best case in the set.
        """
        if self.spot <= 0:
            return math.inf
        return (self.breakeven_distance_atr * self.atr) / self.spot

    # --- per-contract dollars, for sizing and the risk layer ---------------

    @property
    def premium_per_contract(self) -> float:
        """What one contract costs at the mid, in dollars."""
        return self.premium * CONTRACT_MULTIPLIER

    @property
    def cost_per_contract(self) -> float:
        """What one contract costs paying the ask -- the number sizing spends."""
        return (self.premium + self.spread / 2.0) * CONTRACT_MULTIPLIER

    @property
    def max_risk(self) -> float:
        """A long option's maximum loss is the premium paid, in dollars.

        Bounded, unlike its gain -- which is why single legs have no
        reward-to-risk and are ranked on modeled P&L per unit of spread
        instead. See :class:`VerticalMetrics` for the bounded case.
        """
        return self.cost_per_contract


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


# ---------------------------------------------------------------------------
# Debit verticals
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerticalMetrics:
    """A debit vertical's risk, reward, and how much of it a hold can capture.

    A vertical differs from a single leg in the way that matters most to
    sizing: **both sides are bounded**. Max risk is the net debit and max gain
    is the width less that debit, both known at entry. That makes
    reward-to-risk a real number, rather than the unbounded ratio a long call
    has, and it is why verticals rank differently.

    ``pct_of_max_capturable_at_hold`` is the metric that decides whether a
    vertical earns its four bid-ask crossings at all. A debit spread converges
    toward max value only near expiry; over a short hold on a long-dated
    contract the short leg's decay offsets the long leg's gain and the spread
    barely moves. A 3:1 reward-to-risk that captures 6% of max gain over the
    hold is worse than a single leg, and this number is what exposes that.
    """

    # --- structure ---
    option_type: OptionType
    long_strike: float
    short_strike: float
    width: float
    net_debit: float                      # per share

    # --- the four headline figures, per contract in dollars ---
    max_risk: float
    max_gain: float
    reward_to_risk: float
    pct_of_max_capturable_at_hold: float

    # --- per share, where the multiplier cancels ---
    max_risk_per_share: float
    max_gain_per_share: float
    breakeven: float
    breakeven_distance_atr: float
    modeled_pnl_1atr: float
    modeled_pnl_per_contract: float

    # --- ranking ---
    rank_score: float

    # --- position greeks and friction, echoed ---
    net_delta: float
    net_gamma: float
    net_theta: float
    entry_spread_cost: float
    round_trip_spread_cost: float

    # --- inputs, echoed ---
    spot: float
    atr: float
    hold_hours: float
    multiplier: int

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def is_finite(self) -> bool:
        return all(
            math.isfinite(v)
            for v in self.as_dict().values()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        )


def compute_vertical_metrics(
    *,
    option_type: OptionType,
    long_strike: float,
    short_strike: float,
    spot: float,
    atr: float,
    long_bid: float,
    long_ask: float,
    short_bid: float,
    short_ask: float,
    long_delta: float | None,
    long_gamma: float | None,
    long_theta: float | None,
    short_delta: float | None,
    short_gamma: float | None,
    short_theta: float | None,
    hold_hours: float,
    theta_day_hours: float,
    breakeven_discount_k: float,
    multiplier: int = CONTRACT_MULTIPLIER,
) -> VerticalMetrics:
    """Compute a debit vertical's bounded risk/reward and its ranking score.

    Entry is modelled honestly: the long leg is bought at the **ask** and the
    short leg sold at the **bid**, so ``net_debit`` already contains both
    crossings. Exit crosses both again, which ``round_trip_spread_cost``
    records -- four crossings in total, the cost that most often makes a
    short-hold vertical lose to a single leg.
    """
    for name, value in (
        ("long_delta", long_delta), ("long_gamma", long_gamma), ("long_theta", long_theta),
        ("short_delta", short_delta), ("short_gamma", short_gamma), ("short_theta", short_theta),
    ):
        if value is None:
            raise MetricError(f"{name} is missing -- vertical is not scoreable")

    if atr <= 0:
        raise MetricError(f"atr must be positive, got {atr}")
    if spot <= 0:
        raise MetricError(f"spot must be positive, got {spot}")
    if multiplier <= 0:
        raise MetricError(f"multiplier must be positive, got {multiplier}")
    if hold_hours < 0 or theta_day_hours <= 0:
        raise MetricError(f"bad hold window: hold_hours={hold_hours} day={theta_day_hours}")
    for label, bid, ask in (("long", long_bid, long_ask), ("short", short_bid, short_ask)):
        if bid <= 0 or ask < bid:
            raise MetricError(f"unusable {label} quote: bid={bid} ask={ask}")

    # A debit call spread buys the lower strike; a debit put spread buys the
    # higher. Getting this backwards builds a *credit* spread, which is out of
    # scope entirely -- Level 3 covers debit structures only.
    if option_type == "call":
        if short_strike <= long_strike:
            raise MetricError(
                f"debit call vertical needs short_strike > long_strike, "
                f"got {short_strike} <= {long_strike}"
            )
    elif option_type == "put":
        if short_strike >= long_strike:
            raise MetricError(
                f"debit put vertical needs short_strike < long_strike, "
                f"got {short_strike} >= {long_strike}"
            )
    else:
        raise MetricError(f"unknown option_type {option_type!r}")

    width = abs(short_strike - long_strike)
    net_debit = long_ask - short_bid
    if net_debit <= 0:
        raise MetricError(
            f"net debit must be positive, got {net_debit} -- that is a credit spread, "
            "which is out of scope"
        )
    if net_debit >= width:
        raise MetricError(f"net debit {net_debit} >= width {width} -- no achievable gain")

    max_risk_per_share = net_debit
    max_gain_per_share = width - net_debit
    max_risk = max_risk_per_share * multiplier
    max_gain = max_gain_per_share * multiplier
    reward_to_risk = max_gain_per_share / max_risk_per_share

    if option_type == "call":
        breakeven = long_strike + net_debit
        distance = breakeven - spot
    else:
        breakeven = long_strike - net_debit
        distance = spot - breakeven
    breakeven_distance_atr = distance / atr

    # Position greeks. Long a spread is long the near leg and short the far
    # one, so every position greek is the difference between them.
    net_delta = abs(float(long_delta)) - abs(float(short_delta))
    net_gamma = float(long_gamma) - float(short_gamma)
    net_theta = float(long_theta) - float(short_theta)
    if net_delta <= 0:
        raise MetricError(
            f"net delta {net_delta} is not positive -- the short leg is not further OTM"
        )

    directional = net_delta * atr + 0.5 * net_gamma * (atr ** 2)
    decay = net_theta * (hold_hours / theta_day_hours)   # signed; normally negative
    modeled = directional + decay

    # A spread cannot be worth more than its width. Capping matters here in a
    # way it does not for a single leg: an uncapped delta+gamma estimate on a
    # narrow spread happily projects a gain the structure cannot pay.
    capped = min(modeled, max_gain_per_share)
    pct_of_max_capturable_at_hold = capped / max_gain_per_share

    entry_spread_cost = (long_ask - long_bid) / 2.0 + (short_ask - short_bid) / 2.0
    round_trip_spread_cost = entry_spread_cost * 2.0

    # Reward-to-risk, discounted by how far the underlying must travel to break
    # even. A 4:1 spread needing three ATRs of movement is not a better trade
    # than a 2:1 needing half of one, and ranking on raw R:R systematically
    # prefers the former -- the wide, cheap, far-OTM spread shows the most
    # attractive ratio and almost never pays.
    #
    # The decay is EXPONENTIAL, not a 1/(1+kd) divisor. Reward-to-risk grows
    # without bound as a spread moves OTM (a 0.45 debit on a 10-wide spread is
    # 21:1) while a linear discount only ever divides by a small factor, so the
    # junk spread still wins. Measured on exactly that pair: linear ranked the
    # 21:1 / 5.2-ATR spread at 3.41 against 0.61 for a 1.27:1 / 1.1-ATR spread
    # -- 5.6x the wrong way round. Exponential gives 0.11 against 0.42.
    #
    # Probability of travelling d ATRs falls off roughly exponentially, so this
    # is also the shape the discount ought to have.
    discount = math.exp(-breakeven_discount_k * max(0.0, breakeven_distance_atr))
    rank_score = reward_to_risk * discount

    return VerticalMetrics(
        option_type=option_type,
        long_strike=float(long_strike),
        short_strike=float(short_strike),
        width=width,
        net_debit=net_debit,
        max_risk=max_risk,
        max_gain=max_gain,
        reward_to_risk=reward_to_risk,
        pct_of_max_capturable_at_hold=pct_of_max_capturable_at_hold,
        max_risk_per_share=max_risk_per_share,
        max_gain_per_share=max_gain_per_share,
        breakeven=breakeven,
        breakeven_distance_atr=breakeven_distance_atr,
        modeled_pnl_1atr=capped,
        modeled_pnl_per_contract=capped * multiplier,
        rank_score=rank_score,
        net_delta=net_delta,
        net_gamma=net_gamma,
        net_theta=net_theta,
        entry_spread_cost=entry_spread_cost,
        round_trip_spread_cost=round_trip_spread_cost,
        spot=float(spot),
        atr=float(atr),
        hold_hours=float(hold_hours),
        multiplier=int(multiplier),
    )
