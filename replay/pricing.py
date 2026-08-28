"""Black-Scholes prices and greeks. Pure functions, no I/O, no config.

**This is a model of an option, not a recording of one.** Alpaca has historical
option data only from Feb 2024 and this project runs on the indicative feed, so
an offline replay cannot be handed the chain that actually existed on a past
session. The alternative to modelling it is not replaying at all.

What that costs is stated plainly here rather than buried: replay measures the
*pipeline* -- whether the prefilter narrows, whether the agents choose, whether
the risk layer sizes and the exits fire -- against a chain whose liquidity and
pricing were chosen by :mod:`replay.chain`. It does not measure whether the
strategy makes money, and a replay P&L figure is a property of the model's
parameters at least as much as of the decisions. Every report the harness emits
carries the model parameters with it for exactly that reason.

Conventions match what the Alpaca snapshots endpoint returns, so a candidate
built here is interchangeable with a live one:

* ``delta`` signed -- positive for calls, negative for puts
* ``theta`` per calendar day, negative
* ``vega`` per one percentage point of implied volatility
* ``rho`` per one percentage point of rate
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

OptionType = Literal["call", "put"]

DAYS_PER_YEAR = 365.0


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


@dataclass(frozen=True)
class Greeks:
    """One contract's theoretical value and sensitivities."""

    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


def black_scholes(
    *,
    option_type: OptionType,
    spot: float,
    strike: float,
    years_to_expiry: float,
    volatility: float,
    rate: float = 0.04,
    dividend_yield: float = 0.0,
) -> Greeks:
    """Price and greeks for a European option.

    American exercise is not modelled. For the long calls and puts this system
    trades, the early-exercise premium is small enough that modelling it would
    add error of its own -- and a replay that mispriced the wings by a few
    cents would still select the same contracts, because selection runs on
    delta and spread rather than on absolute price.

    At or past expiry the option is worth its intrinsic value and every greek
    is zero. That is returned rather than raised: a replay reaching expiry is
    an ordinary event the harness handles, and the "never hold into expiry
    week" rule is enforced upstream where it belongs.
    """
    if spot <= 0 or strike <= 0:
        raise ValueError(f"spot and strike must be positive, got {spot} and {strike}")
    if volatility <= 0:
        raise ValueError(f"volatility must be positive, got {volatility}")

    if years_to_expiry <= 0:
        intrinsic = max(0.0, spot - strike) if option_type == "call" else max(0.0, strike - spot)
        return Greeks(price=intrinsic, delta=0.0, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)

    t = years_to_expiry
    sqrt_t = math.sqrt(t)
    sigma_sqrt_t = volatility * sqrt_t

    d1 = (
        math.log(spot / strike) + (rate - dividend_yield + 0.5 * volatility**2) * t
    ) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t

    discount = math.exp(-rate * t)
    carry = math.exp(-dividend_yield * t)
    pdf_d1 = _norm_pdf(d1)

    # Shared across both types.
    gamma = carry * pdf_d1 / (spot * sigma_sqrt_t)
    vega = spot * carry * pdf_d1 * sqrt_t / 100.0

    if option_type == "call":
        price = spot * carry * _norm_cdf(d1) - strike * discount * _norm_cdf(d2)
        delta = carry * _norm_cdf(d1)
        theta_year = (
            -spot * carry * pdf_d1 * volatility / (2.0 * sqrt_t)
            - rate * strike * discount * _norm_cdf(d2)
            + dividend_yield * spot * carry * _norm_cdf(d1)
        )
        rho = strike * t * discount * _norm_cdf(d2) / 100.0
    elif option_type == "put":
        price = strike * discount * _norm_cdf(-d2) - spot * carry * _norm_cdf(-d1)
        delta = -carry * _norm_cdf(-d1)
        theta_year = (
            -spot * carry * pdf_d1 * volatility / (2.0 * sqrt_t)
            + rate * strike * discount * _norm_cdf(-d2)
            - dividend_yield * spot * carry * _norm_cdf(-d1)
        )
        rho = -strike * t * discount * _norm_cdf(-d2) / 100.0
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    return Greeks(
        price=max(price, 0.0),
        delta=delta,
        gamma=gamma,
        theta=theta_year / DAYS_PER_YEAR,
        vega=vega,
        rho=rho,
    )


def years_between(start_days: int) -> float:
    """Calendar days to a year fraction, the unit Black-Scholes wants.

    Calendar days, not sessions: theta accrues over weekends. The rest of this
    system counts in sessions because that is how holds and expiries are
    reasoned about, and the two must not be confused at this boundary.
    """
    return max(0.0, start_days) / DAYS_PER_YEAR
