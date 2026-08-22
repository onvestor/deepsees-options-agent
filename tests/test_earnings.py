"""Earnings exclusion. Every test here is about failing closed.

The happy path is one test. The rest are the ways this exclusion could
silently stop excluding, which is the only way it can hurt us: an exclusion
that quietly passes everything looks identical to one that is working.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import pytest

from src.brokers.alpaca.calendar import TradingCalendar
from src.earnings.calendar import (
    EarningsCalendar,
    EarningsEntry,
    EarningsError,
    evaluate_exclusion,
    spans_earnings,
)

NOW = datetime(2026, 8, 24, 13, 0, tzinfo=timezone.utc)
SESSION = date(2026, 8, 24)
SESSIONS = tuple(
    SESSION + timedelta(days=d)
    for d in (0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25)
)
CAL = TradingCalendar(
    sessions=SESSIONS,
    closes={d: datetime(d.year, d.month, d.day, 16, 0) for d in SESSIONS},
)


def entry(days_ahead=30, confirmed=True, age_hours=1.0, no_date=False):
    return EarningsEntry(
        symbol="NVDA",
        date=None if no_date else (SESSION + timedelta(days=days_ahead)).isoformat(),
        confirmed=confirmed,
        fetched_at=(NOW - timedelta(hours=age_hours)).isoformat(),
    )


def verdict(e, **kwargs):
    defaults = dict(
        symbol="NVDA", entry=e, order_session=SESSION, trading_calendar=CAL, now=NOW,
        max_hold_sessions=5, buffer_sessions=2, max_cache_age_hours=26.0,
    )
    return evaluate_exclusion(**{**defaults, **kwargs})


# --- the one happy path ----------------------------------------------------


def test_a_symbol_reporting_well_beyond_the_window_is_tradeable():
    result = verdict(entry(days_ahead=25))
    assert not result.excluded
    assert result.sessions_until > 7
    assert "clear of the window" in result.reason


# --- every way it fails closed ---------------------------------------------


def test_no_entry_at_all_excludes():
    result = verdict(None)
    assert result.excluded
    assert "no earnings data" in result.reason


def test_entry_with_no_date_excludes():
    result = verdict(entry(no_date=True))
    assert result.excluded
    assert "no earnings date known" in result.reason


def test_stale_data_excludes():
    """A date fetched last week is not evidence about this week."""
    result = verdict(entry(days_ahead=30, age_hours=200.0))
    assert result.excluded
    assert "old" in result.reason
    assert result.age_hours == pytest.approx(200.0)


def test_data_just_inside_the_age_limit_is_accepted():
    assert not verdict(entry(days_ahead=30, age_hours=25.9)).excluded


def test_data_just_past_the_age_limit_excludes():
    assert verdict(entry(days_ahead=30, age_hours=26.1)).excluded


def test_earnings_inside_the_hold_window_excludes():
    """5 max hold + 2 buffer = any symbol reporting within 7 sessions."""
    result = verdict(entry(days_ahead=3))
    assert result.excluded
    assert "within the 7-session hold window" in result.reason


def test_earnings_exactly_at_the_horizon_excludes():
    """The boundary is inclusive -- reporting on the last session we could
    still be holding is not clear of the window."""
    horizon_date = CAL.sessions[7]
    days = (horizon_date - SESSION).days
    result = verdict(entry(days_ahead=days))
    assert result.sessions_until == 7
    assert result.excluded


def test_earnings_one_session_past_the_horizon_is_clear():
    horizon_date = CAL.sessions[8]
    days = (horizon_date - SESSION).days
    result = verdict(entry(days_ahead=days))
    assert result.sessions_until == 8
    assert not result.excluded


def test_a_past_earnings_date_excludes_because_the_data_is_not_current():
    """Stale-by-content rather than stale-by-clock: a fresh fetch returning a
    date already behind us means the provider is wrong, not that we are clear."""
    result = verdict(entry(days_ahead=-10))
    assert result.excluded
    assert "in the past" in result.reason


def test_unconfirmed_can_be_treated_as_unknown():
    lenient = verdict(entry(days_ahead=30, confirmed=False))
    strict = verdict(entry(days_ahead=30, confirmed=False), require_confirmed=True)
    assert not lenient.excluded
    assert strict.excluded
    assert "estimate, not confirmed" in strict.reason


def test_the_window_is_measured_in_trading_sessions_not_calendar_days():
    """9 calendar days spanning a weekend is only 7 trading sessions.

    A naive calendar-day test against a 7-session horizon would clear this
    symbol and trade straight into the print; counting sessions excludes it.
    """
    result = verdict(entry(days_ahead=9))
    assert result.sessions_until == 7 < 9
    assert result.excluded


# --- fetching --------------------------------------------------------------


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, by_symbol, status=200):
        self.by_symbol = by_symbol
        self.status = status
        self.urls = []

    def get(self, url, timeout=None):
        self.urls.append(url)
        symbol = url.split("symbol=")[1].split("&")[0]
        return FakeResponse(self.by_symbol.get(symbol, []), self.status)


def make_calendar(tmp_path, session, api_key="test-key"):
    return EarningsCalendar(
        api_key=api_key,
        cache_path=tmp_path / "earnings.json",
        base_url="https://example.invalid",
        path="/stable/earnings-calendar",
        timeout=5.0,
        lookahead_days=120,
        clock=lambda: NOW,
        session=session,
    )


def test_refresh_stores_date_confirmed_and_fetched_at(tmp_path):
    session = FakeSession({"NVDA": [{"symbol": "NVDA", "date": "2026-11-19", "time": "amc"}]})
    calendar = make_calendar(tmp_path, session)
    result = calendar.refresh(["NVDA"])

    stored = result["NVDA"]
    assert stored.date == "2026-11-19"
    assert stored.confirmed is True
    assert stored.fetched_at == NOW.isoformat()
    assert stored.source == "fmp"


def test_refresh_picks_the_earliest_upcoming_date(tmp_path):
    session = FakeSession({"NVDA": [
        {"symbol": "NVDA", "date": "2027-02-18"},
        {"symbol": "NVDA", "date": "2026-11-19"},
        {"symbol": "NVDA", "date": "2026-05-01"},        # already past
    ]})
    assert make_calendar(tmp_path, session).refresh(["NVDA"])["NVDA"].date == "2026-11-19"


def test_a_row_with_no_confirmation_signal_is_unconfirmed(tmp_path):
    session = FakeSession({"NVDA": [{"symbol": "NVDA", "date": "2026-11-19"}]})
    assert make_calendar(tmp_path, session).refresh(["NVDA"])["NVDA"].confirmed is False


def test_empty_provider_response_yields_no_date_which_excludes(tmp_path):
    calendar = make_calendar(tmp_path, FakeSession({"NVDA": []}))
    stored = calendar.refresh(["NVDA"])["NVDA"]
    assert stored.date is None
    assert verdict(stored).excluded


def test_a_failed_fetch_raises_rather_than_returning_a_partial_map(tmp_path):
    """Returning what succeeded would empty the exclusion list for the rest --
    the precise failure this module exists to prevent."""
    calendar = make_calendar(tmp_path, FakeSession({"SPY": []}, status=503))
    with pytest.raises(EarningsError, match="refresh failed"):
        calendar.refresh(["SPY", "NVDA"])


def test_missing_api_key_raises_rather_than_running_empty(tmp_path):
    calendar = make_calendar(tmp_path, FakeSession({}), api_key=None)
    with pytest.raises(EarningsError, match="FMP_API_KEY"):
        calendar.refresh(["NVDA"])


def test_malformed_rows_are_skipped_not_guessed(tmp_path):
    session = FakeSession({"NVDA": [
        "not a dict",
        {"symbol": "NVDA", "date": "not-a-date"},
        {"symbol": "NVDA"},
        {"symbol": "OTHER", "date": "2026-09-01"},       # wrong symbol
        {"symbol": "NVDA", "date": "2026-11-19"},
    ]})
    assert make_calendar(tmp_path, session).refresh(["NVDA"])["NVDA"].date == "2026-11-19"


def test_non_list_payload_raises(tmp_path):
    calendar = make_calendar(tmp_path, FakeSession({"NVDA": {"error": "nope"}}))
    with pytest.raises(EarningsError):
        calendar.refresh(["NVDA"])


# --- cache round trip ------------------------------------------------------


def test_cache_persists_and_reloads(tmp_path):
    session = FakeSession({"NVDA": [{"symbol": "NVDA", "date": "2026-11-19", "time": "amc"}]})
    make_calendar(tmp_path, session).refresh(["NVDA"])

    reloaded = make_calendar(tmp_path, FakeSession({}))
    stored = reloaded.get("nvda")
    assert stored is not None
    assert stored.date == "2026-11-19"
    assert stored.confirmed is True


def test_a_corrupt_cache_reads_as_empty_which_excludes(tmp_path):
    (tmp_path / "earnings.json").write_text("{not json", encoding="utf-8")
    calendar = make_calendar(tmp_path, FakeSession({}))
    assert len(calendar) == 0
    assert verdict(calendar.get("NVDA")).excluded


def test_malformed_cache_entries_are_discarded(tmp_path):
    (tmp_path / "earnings.json").write_text(
        json.dumps({"entries": {"NVDA": {"unexpected": "shape"}}}), encoding="utf-8"
    )
    assert len(make_calendar(tmp_path, FakeSession({}))) == 0


def test_cache_file_records_every_required_field(tmp_path):
    session = FakeSession({"NVDA": [{"symbol": "NVDA", "date": "2026-11-19", "time": "amc"}]})
    calendar = make_calendar(tmp_path, session)
    calendar.refresh(["NVDA"])
    row = json.loads(calendar.cache_path.read_text(encoding="utf-8"))["entries"]["NVDA"]
    assert set(row) == {"symbol", "date", "confirmed", "fetched_at", "source"}


# --- contract-span test ----------------------------------------------------


def test_spans_earnings():
    assert spans_earnings(date(2026, 12, 18), date(2026, 11, 19)) is True
    assert spans_earnings(date(2026, 10, 16), date(2026, 11, 19)) is False


def test_spans_earnings_is_unknown_when_the_date_is_unknown():
    """None, not False. The caller decides, and the prefilter excludes."""
    assert spans_earnings(date(2026, 12, 18), None) is None
