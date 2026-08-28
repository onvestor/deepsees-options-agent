"""The market-hours state machine. Pure over an injected clock.

Every window this system has is a clock time in ``config/limits.yaml`` under
``session:``, and every one of them is a policy choice rather than an
observation: the market opens at 09:30, but *we* do not enter before
``first_entry_et`` because the opening auction imbalance is not a signal.

**No wall-clock reads inside.** ``now`` is always passed in. A state machine
that read the clock itself could not be tested for the boundary minutes, which
are the only minutes that matter -- the session before the first entry window,
the minute after the last, the moment the halt flips. Every method here is a
function of its arguments.

**A non-session day is not a closed market, and the difference matters.** The
market being shut on a Sunday and the operator having listed a date in
``session.skip_dates`` produce the same "do not trade", but for reasons that
should never be conflated in a log: one is the calendar and one is a decision.
:class:`SessionPhase` keeps them apart.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import Enum
from typing import Any, Iterable

from src.brokers.alpaca.calendar import ET, TradingCalendar

log = logging.getLogger(__name__)


class SessionPhase(str, Enum):
    """Where the clock is, in the terms this system acts on.

    Ordered roughly by the day, but never compared by order -- the actions
    permitted in each phase are explicit properties, because "later than
    ENTRY_WINDOW" is not a safe way to ask "may I still exit?".
    """

    NOT_A_SESSION = "not_a_session"
    """The market is shut: weekend, holiday, or outside the calendar."""

    SKIPPED = "skipped"
    """A trading day the operator listed in ``session.skip_dates``."""

    PRE_MARKET = "pre_market"
    """Before the open. Agents 1 and 2 run here; nothing trades."""

    WARMUP = "warmup"
    """Open, but before ``first_entry_et``. Positions are managed; no entries."""

    ENTRY_WINDOW = "entry_window"
    """The only phase in which a new position may be opened."""

    MANAGE_ONLY = "manage_only"
    """After ``last_entry_et``. Exits and stops still run."""

    AFTER_CLOSE = "after_close"
    """The session is over. Agent 6 runs here."""


@dataclass(frozen=True)
class SessionWindows:
    """The clock times, read once from config."""

    market_open: time
    market_close: time
    first_entry: time
    last_entry: time
    flat_by: time
    skip_dates: frozenset[date]

    @classmethod
    def from_limits(cls, limits: Any) -> "SessionWindows":
        windows = cls(
            market_open=_as_time(limits.get_str("session.market_open_et")),
            market_close=_as_time(limits.get_str("session.market_close_et")),
            first_entry=_as_time(limits.get_str("session.first_entry_et")),
            last_entry=_as_time(limits.get_str("session.last_entry_et")),
            flat_by=_as_time(limits.get_str("session.flat_by_et")),
            skip_dates=frozenset(_as_dates(limits.get_list("session.skip_dates"))),
        )
        windows.validate()
        return windows

    def validate(self) -> None:
        """Refuse an ordering that cannot be satisfied.

        A ``last_entry`` after ``flat_by`` would open a position the same loop
        is about to be told to close. Caught at construction because the
        alternative is discovering it at 15:44 on a live session.
        """
        ordered = (
            ("market_open", self.market_open),
            ("first_entry", self.first_entry),
            ("last_entry", self.last_entry),
            ("flat_by", self.flat_by),
            ("market_close", self.market_close),
        )
        for (a_name, a), (b_name, b) in zip(ordered, ordered[1:]):
            if a > b:
                raise ValueError(
                    f"session.{a_name}_et ({a}) must not be after "
                    f"session.{b_name}_et ({b})"
                )


def _as_time(text: str) -> time:
    try:
        hour, minute = text.split(":")
        return time(int(hour), int(minute))
    except Exception as exc:
        raise ValueError(f"expected a 'HH:MM' clock time, got {text!r}") from exc


def _as_dates(values: Iterable[Any]) -> list[date]:
    out: list[date] = []
    for value in values or ():
        if isinstance(value, date):
            out.append(value)
        else:
            out.append(date.fromisoformat(str(value)))
    return out


@dataclass(frozen=True)
class SessionState:
    """The phase, plus the facts that decided it."""

    phase: SessionPhase
    session: date
    now: datetime
    windows: SessionWindows

    # -- what is permitted, asked explicitly rather than by phase ordering ---

    @property
    def is_trading_day(self) -> bool:
        return self.phase not in (SessionPhase.NOT_A_SESSION, SessionPhase.SKIPPED)

    @property
    def may_open(self) -> bool:
        """Only inside the entry window. Nothing else opens a position."""
        return self.phase is SessionPhase.ENTRY_WINDOW

    @property
    def may_manage(self) -> bool:
        """Exits run from the open until the close.

        Deliberately wider than ``may_open``. An open position is risk whether
        or not the entry window is still up, and the phase that stops new
        entries must not stop the loop that manages what is already on.
        """
        return self.phase in (
            SessionPhase.WARMUP,
            SessionPhase.ENTRY_WINDOW,
            SessionPhase.MANAGE_ONLY,
        )

    @property
    def may_run_premarket_agents(self) -> bool:
        return self.phase is SessionPhase.PRE_MARKET

    @property
    def may_review(self) -> bool:
        return self.phase is SessionPhase.AFTER_CLOSE

    @property
    def past_flat_by(self) -> bool:
        """Past the hard time-stop.

        Note this is *not* an instruction to go flat: the swing design holds
        overnight by design and has no intraday flat rule. It exists because
        expiry-week positions must not be left to auto-exercise, and the exit
        layer consults it for those.
        """
        return self.is_trading_day and _et(self.now).time() >= self.windows.flat_by

    def describe(self) -> str:
        return f"{self.session} {self.phase.value} at {_et(self.now):%H:%M %Z}"


def _et(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=ET)
    return moment.astimezone(ET)


class SessionClock:
    """Answers "what phase is it" for any moment. Construct once per process."""

    def __init__(self, windows: SessionWindows, calendar: TradingCalendar) -> None:
        self.windows = windows
        self.calendar = calendar

    @classmethod
    def from_config(cls, config: Any, calendar: TradingCalendar) -> "SessionClock":
        return cls(SessionWindows.from_limits(config.limits), calendar)

    def state(self, now: datetime) -> SessionState:
        """The phase at ``now``, and the session it belongs to."""
        moment = _et(now)
        today = moment.date()
        w = self.windows

        if not self.calendar.is_session(today):
            return SessionState(SessionPhase.NOT_A_SESSION, today, moment, w)
        if today in w.skip_dates:
            # A decision, not the calendar. Kept distinct so a quiet day in the
            # log can be attributed to one or the other.
            return SessionState(SessionPhase.SKIPPED, today, moment, w)

        clock = moment.time()
        if clock < w.market_open:
            phase = SessionPhase.PRE_MARKET
        elif clock < w.first_entry:
            phase = SessionPhase.WARMUP
        elif clock < w.last_entry:
            phase = SessionPhase.ENTRY_WINDOW
        elif clock < w.market_close:
            phase = SessionPhase.MANAGE_ONLY
        else:
            phase = SessionPhase.AFTER_CLOSE
        return SessionState(phase, today, moment, w)

    def next_session(self, after: date) -> date | None:
        for session in self.calendar.sessions:
            if session > after and session not in self.windows.skip_dates:
                return session
        return None

    def session_bounds(self, session: date) -> tuple[datetime, datetime]:
        """Open and close as aware datetimes, for scheduling within a session."""
        return (
            datetime.combine(session, self.windows.market_open, tzinfo=ET),
            datetime.combine(session, self.windows.market_close, tzinfo=ET),
        )
