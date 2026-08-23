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
    assert_universe_resolves,
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


def entry(days_ahead=30, confirmed=True, age_hours=1.0, no_date=False, days_since=90):
    """``days_since`` is the *previous* print, well clear of any buffer unless a
    test moves it."""
    return EarningsEntry(
        symbol="NVDA",
        date=None if no_date else (SESSION + timedelta(days=days_ahead)).isoformat(),
        confirmed=confirmed,
        fetched_at=(NOW - timedelta(hours=age_hours)).isoformat(),
        previous_date=None if days_since is None
        else (SESSION - timedelta(days=days_since)).isoformat(),
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
    assert "marks it an estimate" in strict.reason


def test_no_confirmation_signal_is_reported_as_silence_not_as_an_estimate():
    """`confirmed=None` and `confirmed=False` both exclude under
    require_confirmed, but for different reasons, and the reason is the whole
    point: one means the provider called it an estimate, the other means the
    provider said nothing. Collapsing them hides a dead field."""
    strict = verdict(entry(days_ahead=30, confirmed=None), require_confirmed=True)
    assert strict.excluded
    assert "no confirmation signal" in strict.reason
    assert not verdict(entry(days_ahead=30, confirmed=None)).excluded


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
        path="/stable/earnings",
        timeout=5.0,
        max_rows=5,
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


def test_a_row_with_no_confirmation_signal_is_null_not_false(tmp_path):
    """The live /stable/earnings payload carries epsEstimated and lastUpdated
    and no scheduling field at all. Reporting False there would claim the
    provider called it an estimate, which it never did."""
    session = FakeSession({"NVDA": [
        {"symbol": "NVDA", "date": "2026-11-19", "epsActual": None,
         "epsEstimated": 2.09, "lastUpdated": "2026-08-23"},
    ]})
    assert make_calendar(tmp_path, session).refresh(["NVDA"])["NVDA"].confirmed is None


def test_an_explicit_time_field_still_reads_as_confirmed(tmp_path):
    session = FakeSession({"NVDA": [{"symbol": "NVDA", "date": "2026-11-19", "time": "amc"}]})
    assert make_calendar(tmp_path, session).refresh(["NVDA"])["NVDA"].confirmed is True


def test_an_unrecognised_time_field_reads_as_unconfirmed_not_null(tmp_path):
    session = FakeSession({"NVDA": [{"symbol": "NVDA", "date": "2026-11-19", "time": "--"}]})
    assert make_calendar(tmp_path, session).refresh(["NVDA"])["NVDA"].confirmed is False


def test_a_response_for_other_symbols_raises_rather_than_filtering_to_nothing(tmp_path):
    """The bug this rewrite exists to remove. /stable/earnings-calendar ignores
    `symbol` and returns a capped market-wide slice; filtering it client-side
    yields None, which fails closed and therefore looks fine -- right up until
    a later row for the right symbol survives the cap and reads as *clear*."""
    session = FakeSession({"NVDA": [
        {"symbol": "FDX", "date": "2026-09-17"},
        {"symbol": "ADBE", "date": "2026-09-10"},
    ]})
    with pytest.raises(EarningsError, match="not symbol-scoped"):
        make_calendar(tmp_path, session).refresh(["NVDA"])


def test_the_request_is_symbol_scoped_and_carries_no_date_bounds(tmp_path):
    session = FakeSession({"NVDA": [{"symbol": "NVDA", "date": "2026-11-19"}]})
    make_calendar(tmp_path, session).refresh(["NVDA"])
    url = session.urls[0]
    assert "symbol=NVDA" in url and "limit=5" in url
    assert "from=" not in url and "to=" not in url


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
    assert set(row) == {
        "symbol", "date", "confirmed", "fetched_at", "source", "previous_date",
    }


# --- contract-span test ----------------------------------------------------


def test_spans_earnings():
    assert spans_earnings(date(2026, 12, 18), date(2026, 11, 19)) is True
    assert spans_earnings(date(2026, 10, 16), date(2026, 11, 19)) is False


def test_spans_earnings_is_unknown_when_the_date_is_unknown():
    """None, not False. The caller decides, and the prefilter excludes."""
    assert spans_earnings(date(2026, 12, 18), None) is None


# --- the declared no-earnings class ----------------------------------------


def test_a_declared_no_earnings_instrument_is_tradeable_despite_having_no_date():
    """SPY has no print and never will. Without the declaration, rule 1 blocks
    the most liquid name in the universe forever, behind an exclusion that
    looks perfectly healthy."""
    result = verdict(entry(no_date=True), symbol="SPY", no_earnings=True)
    assert not result.excluded
    assert result.no_earnings_class
    assert "declared in universe.yaml" in result.reason


def test_a_no_earnings_instrument_with_no_entry_at_all_is_still_clear():
    """The claim is about the instrument and comes from config, so it does not
    depend on the provider having been reached."""
    assert not verdict(None, symbol="SPY", no_earnings=True).excluded


def test_a_no_earnings_instrument_is_not_exempt_from_staleness_by_accident():
    """Staleness is skipped for the class deliberately -- but only because the
    claim is config-borne. A stale entry must not become a back door for a
    symbol that is NOT in the class."""
    assert verdict(entry(days_ahead=30, age_hours=200.0), no_earnings=False).excluded


def test_a_declared_no_earnings_symbol_that_returns_a_date_excludes_loudly():
    """A wrong declaration is more dangerous than a missing one: it reads as
    deliberate. Trust the provider over the config and say so."""
    result = verdict(entry(days_ahead=3), symbol="SPY", no_earnings=True)
    assert result.excluded
    assert "declared no-earnings but the provider returned" in result.reason
    assert result.no_earnings_class


# --- the startup assertion -------------------------------------------------


def populated(tmp_path, by_symbol):
    calendar = make_calendar(tmp_path, FakeSession(by_symbol))
    calendar.refresh(list(by_symbol))
    return calendar


def test_universe_resolves_when_every_symbol_has_a_date_or_a_declaration(tmp_path):
    calendar = populated(tmp_path, {
        "NVDA": [{"symbol": "NVDA", "date": "2026-11-19"}],
        "SPY": [],
    })
    resolved = assert_universe_resolves(
        ["NVDA", "SPY"], calendar, {"SPY"}, NOW, 26.0
    )
    assert resolved == {"NVDA": "earnings_date", "SPY": "no_earnings_class"}


def test_a_symbol_resolving_to_silence_halts_the_session(tmp_path):
    """The assertion that would have caught the endpoint bug on day one. Every
    other rule makes a dead feed *safe*; only this one makes it *visible*."""
    calendar = populated(tmp_path, {"NVDA": [], "AAPL": [{"symbol": "AAPL", "date": "2026-10-29"}]})
    with pytest.raises(EarningsError, match="NVDA .fetched, no date returned"):
        assert_universe_resolves(["NVDA", "AAPL"], calendar, set(), NOW, 26.0)


def test_a_never_fetched_symbol_halts_the_session(tmp_path):
    calendar = populated(tmp_path, {"NVDA": [{"symbol": "NVDA", "date": "2026-11-19"}]})
    with pytest.raises(EarningsError, match="AAPL .no entry"):
        assert_universe_resolves(["NVDA", "AAPL"], calendar, set(), NOW, 26.0)


def test_a_stale_symbol_halts_the_session(tmp_path):
    calendar = populated(tmp_path, {"NVDA": [{"symbol": "NVDA", "date": "2026-11-19"}]})
    later = NOW + timedelta(hours=30)
    with pytest.raises(EarningsError, match="30.0h old"):
        assert_universe_resolves(["NVDA"], calendar, set(), later, 26.0)


def test_a_contradicted_declaration_halts_the_session(tmp_path):
    """Declaring a real company print-free must not be a way to skip the check."""
    calendar = populated(tmp_path, {"NVDA": [{"symbol": "NVDA", "date": "2026-11-19"}]})
    with pytest.raises(EarningsError, match="declared no-earnings, got 2026-11-19"):
        assert_universe_resolves(["NVDA"], calendar, {"NVDA"}, NOW, 26.0)


def test_the_halt_names_every_unresolved_symbol_not_just_the_first(tmp_path):
    calendar = populated(tmp_path, {"NVDA": [], "AAPL": []})
    with pytest.raises(EarningsError) as exc:
        assert_universe_resolves(["NVDA", "AAPL"], calendar, set(), NOW, 26.0)
    assert "NVDA" in str(exc.value) and "AAPL" in str(exc.value)
    assert "2 of 2" in str(exc.value)


# --- the post-print IV crush buffer ----------------------------------------
#
# The forward window cannot see a print that has already happened: the moment
# it passes, the next date jumps a quarter out and the symbol reads clear on
# the exact session its IV is collapsing. Every test here is that hole.


# A calendar reaching back far enough to place a previous print. CAL above
# starts at SESSION, which is fine forwards and useless backwards.
def weekday_calendar(back_days, forward_days):
    days = tuple(
        d for d in (SESSION + timedelta(days=n)
                    for n in range(-back_days, forward_days + 1))
        if d.weekday() < 5
    )
    return TradingCalendar(
        sessions=days,
        closes={d: datetime(d.year, d.month, d.day, 16, 0) for d in days},
    )


BACK_CAL = weekday_calendar(120, 120)
BACK_SESSIONS = tuple(d for d in BACK_CAL.sessions if d <= SESSION)


def prev_session(n):
    """The date exactly ``n`` trading sessions before the order session."""
    return BACK_SESSIONS[-1 - n]


def post(sessions_back, buffer=2, calendar=BACK_CAL, **kwargs):
    e = EarningsEntry(
        symbol="NVDA",
        date=(SESSION + timedelta(days=60)).isoformat(),
        confirmed=True,
        fetched_at=(NOW - timedelta(hours=1)).isoformat(),
        previous_date=None if sessions_back is None else prev_session(sessions_back).isoformat(),
    )
    return verdict(e, trading_calendar=calendar,
                   post_print_buffer_sessions=buffer, **kwargs)


def test_the_session_after_a_print_is_excluded_though_the_forward_window_is_clear():
    """NVDA reports Wednesday; Thursday's next date is a quarter out and reads
    clear on every forward check. It is the worst session of the quarter to buy
    premium on that name."""
    result = post(sessions_back=1)
    assert result.excluded
    assert result.sessions_since_last == 1
    assert "post-print IV crush buffer" in result.reason


def test_the_last_session_inside_the_buffer_is_excluded():
    result = post(sessions_back=2)
    assert result.sessions_since_last == 2
    assert result.excluded


def test_the_first_session_past_the_buffer_is_clear():
    result = post(sessions_back=3)
    assert result.sessions_since_last == 3
    assert not result.excluded


def test_a_print_a_full_quarter_back_is_clear():
    assert not post(sessions_back=60).excluded


def test_an_unknown_previous_date_excludes():
    """Not knowing when a name last reported is not evidence that it did not
    report yesterday -- the same fail-closed logic as the forward side."""
    result = post(sessions_back=None)
    assert result.excluded
    assert "no previous earnings date known" in result.reason


def test_the_buffer_is_off_when_configured_to_zero():
    """0 disables the check, and the unknown-previous requirement with it."""
    assert not post(sessions_back=None, buffer=0).excluded
    assert not post(sessions_back=1, buffer=0).excluded


def test_the_buffer_counts_trading_sessions_not_calendar_days():
    """A Friday print read on the following Monday is 1 session ago but 3
    calendar days. Counting days would clear it a full buffer early."""
    friday = prev_session(1)
    assert friday.weekday() == 4 and (SESSION - friday).days == 3
    result = post(sessions_back=1, buffer=1)
    assert result.sessions_since_last == 1
    assert result.excluded


def test_a_print_predating_the_calendar_window_is_bounded_not_counted_from_the_edge():
    """The guard this needs. Counting from the window edge would report a
    months-old print as days old, and a false exclusion looks exactly like a
    real one. Here the window holds 10 sessions before the order session, which
    is enough to place any earlier print outside a 2-session buffer."""
    short = weekday_calendar(14, 120)          # 10 sessions back, ample forward
    e = EarningsEntry(
        symbol="NVDA", date=(SESSION + timedelta(days=60)).isoformat(), confirmed=True,
        fetched_at=(NOW - timedelta(hours=1)).isoformat(),
        previous_date=(SESSION - timedelta(days=90)).isoformat(),   # outside `short`
    )
    result = verdict(e, trading_calendar=short, post_print_buffer_sessions=2)
    assert not result.excluded


def test_a_calendar_too_short_to_place_the_print_fails_closed():
    """When the window cannot even bound the print outside the buffer, we do
    not guess in the permissive direction."""
    tiny = TradingCalendar(
        sessions=SESSIONS[:1],
        closes={SESSIONS[0]: datetime(2026, 8, 24, 16, 0)},
    )
    e = EarningsEntry(
        symbol="NVDA", date=(SESSION + timedelta(days=60)).isoformat(), confirmed=True,
        fetched_at=(NOW - timedelta(hours=1)).isoformat(),
        previous_date=(SESSION - timedelta(days=90)).isoformat(),
    )
    result = verdict(e, trading_calendar=tiny, post_print_buffer_sessions=2)
    assert result.excluded
    assert "predates the" in result.reason and "widen back_days" in result.reason


def test_refresh_records_the_previous_print_alongside_the_next(tmp_path):
    """The clock is 2026-08-24, so 2026-08-20 is past and 2026-11-19 upcoming."""
    session = FakeSession({"NVDA": [
        {"symbol": "NVDA", "date": "2026-11-19"},
        {"symbol": "NVDA", "date": "2026-08-20"},
        {"symbol": "NVDA", "date": "2026-05-20"},
    ]})
    stored = make_calendar(tmp_path, session).refresh(["NVDA"])["NVDA"]
    assert stored.date == "2026-11-19"
    assert stored.previous_date == "2026-08-20"      # the latest past, not the earliest


def test_a_symbol_with_no_previous_print_halts_the_session_when_the_buffer_is_on(tmp_path):
    """The buffer is only as good as previous_date. A provider that quietly
    stopped returning past quarters would disarm it while every forward-looking
    check kept reporting healthy."""
    calendar = populated(tmp_path, {"NVDA": [{"symbol": "NVDA", "date": "2026-11-19"}]})
    with pytest.raises(EarningsError, match="post-print buffer is blind"):
        assert_universe_resolves(["NVDA"], calendar, set(), NOW, 26.0,
                                 post_print_buffer_sessions=2)
    # ...and is silent about it when the buffer is off.
    assert assert_universe_resolves(["NVDA"], calendar, set(), NOW, 26.0)
