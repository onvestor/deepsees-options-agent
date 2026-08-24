"""Expiry classification. Pure functions, direct unit tests."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.options.expiry import (
    MONTHLY,
    WEEKLY,
    classify,
    is_standard_monthly,
    standard_monthly_expiry,
    third_friday,
)


@pytest.mark.parametrize(
    "year,month,expected",
    [
        (2026, 8, date(2026, 8, 21)),
        (2026, 9, date(2026, 9, 18)),
        (2026, 10, date(2026, 10, 16)),
        (2026, 11, date(2026, 11, 20)),
        (2027, 1, date(2027, 1, 15)),
        (2027, 5, date(2027, 5, 21)),
        # A month starting ON a Friday: the third Friday is the 15th, not the 22nd.
        (2027, 10, date(2027, 10, 15)),
    ],
)
def test_third_friday_is_the_third_friday(year, month, expected):
    got = third_friday(year, month)
    assert got == expected
    assert got.weekday() == 4
    # Structural check: exactly two Fridays precede it in the same month.
    earlier = [d for d in range(1, got.day) if date(year, month, d).weekday() == 4]
    assert len(earlier) == 2


def test_the_amd_case_that_prompted_this():
    """Sep 25 read as thin, Oct 16 read as deep. They are different animals."""
    assert classify(date(2026, 9, 25)) == WEEKLY
    assert classify(date(2026, 10, 16)) == MONTHLY


def test_month_boundaries_do_not_leak():
    """A date is classified against its OWN month, never a neighbouring one."""
    assert is_standard_monthly(date(2026, 10, 16))
    # The September monthly is not the October monthly.
    assert not is_standard_monthly(date(2026, 9, 16))


@pytest.mark.parametrize("month", range(1, 13))
def test_every_month_has_exactly_one_monthly(month):
    friday = third_friday(2026, month)
    days = [date(2026, month, 1) + timedelta(days=i) for i in range(28)]
    monthlies = [d for d in days if d.month == month and is_standard_monthly(d)]
    assert monthlies == [friday]


def test_holiday_third_friday_walks_back_to_the_session_before():
    """Good Friday is the real case: the monthly lists on the Thursday."""
    friday = third_friday(2026, 4)
    thursday = friday - timedelta(days=1)
    closed = lambda d: d != friday  # noqa: E731

    assert standard_monthly_expiry(2026, 4, closed) == thursday
    assert is_standard_monthly(thursday, closed)
    assert not is_standard_monthly(friday, closed)


def test_an_uninformative_calendar_degrades_to_the_plain_rule():
    """A calendar that does not reach the month must not fail the classifier.

    Expiry type is a liquidity heuristic; nothing sizes a position off it. A
    narrow calendar answering "not a session" to everything should yield the
    plain third Friday rather than raising or walking forever.
    """
    friday = third_friday(2026, 10)
    assert standard_monthly_expiry(2026, 10, lambda d: False) == friday
    assert is_standard_monthly(friday, lambda d: False)


def test_month_must_be_valid():
    with pytest.raises(ValueError, match="month must be 1-12"):
        third_friday(2026, 13)
