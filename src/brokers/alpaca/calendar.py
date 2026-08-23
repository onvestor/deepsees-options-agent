"""Trading-session calendar and session-based DTE.

Calendar days are the wrong unit for an intraday options system. A contract
scanned on Saturday with a Monday expiry is 2 calendar days out and **zero
trading sessions** out -- it expires during the very session the order would
be placed in. Sizing, theta and the time-stop all assume there is a session
left to manage the position in.

Two rules follow, and this module exists to enforce both:

1. DTE is measured in **trading sessions**, from Alpaca's own calendar.
2. DTE is measured from the **session the order will belong to**, not from
   whenever the scan happened to run. Those differ every time the scanner runs
   outside market hours, which is most of the time.
"""

from __future__ import annotations

import logging
from bisect import bisect_right
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from alpaca.trading.requests import GetCalendarRequest

from src.brokers.alpaca.client import AlpacaClients, with_retry

log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

__all__ = ["ET", "TradingCalendar", "DteError", "now_et"]


class DteError(RuntimeError):
    """A contract's session-DTE is outside the configured band."""


def now_et() -> datetime:
    """Current time in ET. All times are ET internally, per CLAUDE.md."""
    return datetime.now(tz=ET)


@dataclass(frozen=True)
class TradingCalendar:
    """An ordered tuple of real trading-session dates from Alpaca.

    Holidays and early closes are Alpaca's answer, not ours -- deriving them
    from weekday arithmetic is how a system ends up trying to trade on
    Thanksgiving.
    """

    sessions: tuple[date, ...]
    closes: dict[date, datetime]

    @classmethod
    def fetch(cls, clients: AlpacaClients, start: date, end: date) -> "TradingCalendar":
        request = GetCalendarRequest(start=start, end=end)
        days = with_retry(
            clients.config, "get_calendar", lambda: clients.trading.get_calendar(request)
        )
        sessions = tuple(sorted(d.date for d in days))
        closes = {d.date: d.close for d in days}
        log.debug("calendar: %d sessions %s..%s", len(sessions), start, end)
        return cls(sessions=sessions, closes=closes)

    @classmethod
    def around(
        cls,
        clients: AlpacaClients,
        anchor: date,
        forward_days: int,
        back_days: int = 7,
    ) -> "TradingCalendar":
        """Window wide enough to cover the DTE band plus holiday slack.

        ``back_days`` defaults to a week, which is all a forward-looking DTE
        question needs. Anything measuring backwards -- the post-print earnings
        buffer counts sessions since the last report -- must widen it: sessions
        cannot be counted across days that were never fetched.
        """
        return cls.fetch(
            clients, anchor - timedelta(days=back_days), anchor + timedelta(days=forward_days)
        )

    def is_session(self, day: date) -> bool:
        return day in self.closes

    def order_session(self, at: datetime | None = None) -> date:
        """The session an order placed at ``at`` would belong to.

        If ``at`` falls on a trading day before its close, that day. Otherwise
        the next session -- which is the case every evening and all weekend.
        """
        moment = at or now_et()
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=ET)
        today = moment.astimezone(ET).date()

        close = self.closes.get(today)
        if close is not None:
            close_et = close if close.tzinfo else close.replace(tzinfo=ET)
            if moment.astimezone(ET) < close_et.astimezone(ET):
                return today

        for session in self.sessions:
            if session > today:
                return session
        raise DteError(
            f"calendar does not extend past {today}; widen the fetch window"
        )

    def sessions_until(self, expiry: date, from_session: date) -> int:
        """Trading sessions from ``from_session`` to ``expiry``, exclusive of the
        start and inclusive of the end.

        Same-day expiry is **0** -- a 0DTE contract, which is exactly the case
        a floor of 1 is meant to exclude. Returns -1 for an expiry already in
        the past.
        """
        if expiry < from_session:
            return -1
        if expiry == from_session:
            return 0
        start = bisect_right(self.sessions, from_session)
        end = bisect_right(self.sessions, expiry)
        if end == 0 or self.sessions[end - 1] != expiry:
            # Expiry is not itself a session (rare, but do not silently round).
            log.warning("expiry %s is not a trading session in the calendar", expiry)
        return max(0, end - start)

    def session_offset(self, from_session: date, sessions: int) -> date:
        """The date ``sessions`` trading sessions after ``from_session``.

        Used to translate a session-based DTE band into the calendar-date
        bounds the contracts endpoint actually accepts.
        """
        if sessions <= 0:
            return from_session
        start = bisect_right(self.sessions, from_session)
        index = start + sessions - 1
        if index >= len(self.sessions):
            raise DteError(
                f"calendar does not extend {sessions} sessions past {from_session}"
            )
        return self.sessions[index]


def assert_dte_within_band(
    calendar: TradingCalendar,
    expiry: date,
    order_session: date,
    dte_min: int,
    dte_max: int,
    symbol: str,
) -> int:
    """Fail closed if session-DTE is outside the band. Called at order time.

    The scan-time query already bounds expiries, but the scan may have run
    hours or days earlier. This is the check that actually protects the entry,
    and it is why the 0DTE contract selected on Saturday is rejected on Monday
    rather than bought.
    """
    dte = calendar.sessions_until(expiry, order_session)
    if dte < dte_min or dte > dte_max:
        raise DteError(
            f"{symbol} expires {expiry}, which is {dte} trading session(s) from the "
            f"{order_session} session -- outside the configured band "
            f"[{dte_min}, {dte_max}]. Refusing to open."
        )
    return dte
