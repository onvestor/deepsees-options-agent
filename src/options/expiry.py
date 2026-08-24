"""Standard-monthly vs weekly expiry classification. PURE -- no I/O.

The distinction is not cosmetic. On the same underlying and the same strikes,
a standard monthly and a nearby weekly are different instruments in every way
the prefilter cares about: the monthly is where institutional open interest
accumulates and where market makers quote tightest, and the weekly beside it
can be an order of magnitude thinner.

Measured on AMD, 2026-08-24, at matched ITM strikes:

    Sep 25 (weekly)   open interest      2 - 47      spread 3.5% - 8.2%
    Oct 16 (monthly)  open interest    703 - 6000    spread 1.7% - 2.4%

A liquidity floor applied blind to expiry type therefore reads as "this
underlying has no tradable contracts" when what it actually found was the
wrong expiry. That is the failure this module exists to prevent.

**Definition.** A standard monthly option expires the third Friday of its
month. When that Friday is a market holiday -- Good Friday is the case that
actually occurs -- the contract is listed against the preceding trading day
instead. Callers holding a trading calendar may pass ``is_session`` to
resolve that; without it the rule is the plain third Friday, which is correct
on every month of a normal year.
"""
from __future__ import annotations

import calendar as _calendar
from datetime import date, timedelta
from typing import Callable, Literal

ExpiryType = Literal["monthly", "weekly"]

MONTHLY: ExpiryType = "monthly"
WEEKLY: ExpiryType = "weekly"

_FRIDAY = 4


def third_friday(year: int, month: int) -> date:
    """The third Friday of the given month -- the standard OCC expiry date."""
    if not 1 <= month <= 12:
        raise ValueError(f"month must be 1-12, got {month}")
    first_weekday, _days = _calendar.monthrange(year, month)
    # weekday() is Mon=0..Sun=6; monthrange gives the weekday of day 1.
    offset = (_FRIDAY - first_weekday) % 7
    return date(year, month, 1 + offset + 14)


def standard_monthly_expiry(
    year: int, month: int, is_session: Callable[[date], bool] | None = None
) -> date:
    """The listed monthly expiry, backing off a holiday third Friday.

    Without ``is_session`` this is the plain third Friday. With one, a third
    Friday that is not a trading session walks backwards to the session before
    it, which is how the exchanges list the contract.
    """
    expiry = third_friday(year, month)
    if is_session is None:
        return expiry
    walked = expiry
    for _ in range(7):
        if is_session(walked):
            return walked
        walked -= timedelta(days=1)
    # Seven consecutive non-sessions is not a market condition -- it is a
    # calendar that does not reach this month. Fall back to the plain third
    # Friday rather than failing: expiry type is a liquidity heuristic, and a
    # calendar-free answer is right on every month of a normal year. Nothing
    # downstream sizes a position off this.
    return expiry


def is_standard_monthly(
    expiry: date, is_session: Callable[[date], bool] | None = None
) -> bool:
    """True if ``expiry`` is the standard monthly for its own month."""
    return expiry == standard_monthly_expiry(expiry.year, expiry.month, is_session)


def classify(
    expiry: date, is_session: Callable[[date], bool] | None = None
) -> ExpiryType:
    """``"monthly"`` or ``"weekly"``. Everything that is not the standard
    monthly for its month is a weekly, including quarterlies and dailies --
    the prefilter's question is only ever "is this the deep expiry"."""
    return MONTHLY if is_standard_monthly(expiry, is_session) else WEEKLY


def next_monthly_at_least(
    order_session: date,
    min_sessions: int,
    sessions_until: Callable[[date], int],
    is_session: Callable[[date], bool] | None = None,
    max_months: int = 14,
) -> date:
    """The nearest standard monthly expiry at least ``min_sessions`` out.

    This replaces a fixed calendar-day band. A 15-day-wide window inside a
    ~30-day monthly cycle misses the monthly roughly half the time -- on
    2026-08-24 a 30-45 day band fell entirely between the September monthly
    (25 days out) and the October one (53 days out), and requiring monthlies
    inside it produced an empty survivor set on every symbol.

    Anchoring on the expiry instead of the window always lands on the liquid
    contract and can never empty. The cost is that DTE varies per trade --
    roughly ``min_sessions`` to ``min_sessions + 21`` -- so the realised DTE
    is a property of the entry and must be logged with it, not assumed from
    config.

    ``sessions_until`` must be able to reach far enough forward; a calendar
    that stops short raises rather than silently returning a nearer expiry.
    """
    if min_sessions < 0:
        raise ValueError(f"min_sessions must be >= 0, got {min_sessions}")

    year, month = order_session.year, order_session.month
    for _ in range(max_months):
        expiry = standard_monthly_expiry(year, month, is_session)
        if expiry >= order_session and sessions_until(expiry) >= min_sessions:
            return expiry
        month += 1
        if month > 12:
            year, month = year + 1, 1

    raise ValueError(
        f"no standard monthly expiry at least {min_sessions} sessions after "
        f"{order_session} within {max_months} months -- the trading calendar "
        "does not reach far enough to choose one"
    )
