"""Cadence per agent. Deterministic, and driven by an injected clock.

Two shapes of job, and they are not interchangeable:

* **Clock jobs** fire once per session at a wall-clock time --
  ``agents.a2.run_at_et`` at 09:05, ``agents.a1.run_at_et`` at 09:15. Once per
  session is the point. The swing revision turned Agent 1 from a 30-minute
  poll into a daily judgment, and expressing that as a clock time rather than
  a very long interval is what stops it drifting back into a polling cadence.

* **Interval jobs** fire every N seconds while their phase permits --
  Agent 5 at ``agents.a5.cadence_seconds``, the entry manager on
  ``execution.entry_reprice_cadence_seconds``. These two must never share a
  key: exit management and entry repricing have different urgency, and tuning
  one must not move the other.

**A missed firing is not made up.** If the process was down at 09:15, Agent 1
does not run at 09:41 as though nothing happened -- it runs, once, at the next
tick inside its phase, and the record says it was late. Replaying a backlog of
missed intervals would fire Agent 5 twenty times in a second on restart, each
against the same stale mark.

**Nothing here calls an agent.** The scheduler decides *which* jobs are due and
the caller runs them. That keeps it a pure function of (clock, state, history),
which is the only way the boundary minutes are testable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any, Callable, Iterable

from src.orchestrator.session import SessionPhase, SessionState, _as_time, _et

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Job:
    """One scheduled unit of work.

    ``phases`` is the whitelist. A job outside its phase is not due, no matter
    how long it has been since it last ran -- which is what stops an entry scan
    firing at 15:59 because the process was asleep through the entry window.
    """

    name: str
    phases: frozenset[SessionPhase]
    at: time | None = None
    """Clock job: fire once per session at or after this time."""

    every: timedelta | None = None
    """Interval job: fire this often while in phase."""

    def __post_init__(self) -> None:
        if (self.at is None) == (self.every is None):
            raise ValueError(
                f"job {self.name!r} must be exactly one of a clock job (at=) "
                "or an interval job (every=)"
            )
        if self.every is not None and self.every <= timedelta(0):
            raise ValueError(f"job {self.name!r} interval must be positive")

    @property
    def is_clock_job(self) -> bool:
        return self.at is not None


@dataclass
class JobHistory:
    """When each job last ran, and on which session.

    Sessions are tracked per job rather than globally because a clock job's
    "once" is once per *session*, and the process may span several.
    """

    last_run: dict[str, datetime] = field(default_factory=dict)
    ran_on: dict[str, date] = field(default_factory=dict)

    def record(self, name: str, moment: datetime, session: date) -> None:
        self.last_run[name] = moment
        self.ran_on[name] = session

    def has_run_this_session(self, name: str, session: date) -> bool:
        return self.ran_on.get(name) == session

    def since(self, name: str, moment: datetime) -> timedelta | None:
        last = self.last_run.get(name)
        return None if last is None else moment - last


@dataclass(frozen=True)
class DueJob:
    job: Job
    late_by: timedelta = timedelta(0)
    """How far past its intended firing this is. Recorded, never made up."""

    @property
    def is_late(self) -> bool:
        return self.late_by > timedelta(0)


class Scheduler:
    """Decides what is due. Runs nothing."""

    def __init__(self, jobs: Iterable[Job], history: JobHistory | None = None) -> None:
        self.jobs = tuple(jobs)
        names = [j.name for j in self.jobs]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(f"duplicate job names: {sorted(duplicates)}")
        self.history = history or JobHistory()

    def due(self, state: SessionState, now: datetime | None = None) -> list[DueJob]:
        """Every job due at ``now``, in declaration order.

        Declaration order is the execution order, and it is load-bearing:
        Agent 2 produces the eligible set Agent 1 profiles, so a2 is declared
        before a1 and its ``run_at_et`` is earlier. Sorting by time would agree
        today and silently disagree the day someone retunes one of them.
        """
        moment = _et(now or state.now)
        out: list[DueJob] = []
        for job in self.jobs:
            if state.phase not in job.phases:
                continue
            late = self._due_by(job, state, moment)
            if late is not None:
                out.append(DueJob(job, late))
        return out

    def _due_by(
        self, job: Job, state: SessionState, moment: datetime
    ) -> timedelta | None:
        if job.is_clock_job:
            if self.history.has_run_this_session(job.name, state.session):
                return None
            target = datetime.combine(state.session, job.at, tzinfo=moment.tzinfo)
            if moment < target:
                return None
            return moment - target

        elapsed = self.history.since(job.name, moment)
        if elapsed is None:
            # Never run. Due immediately on entering its phase, but not "late"
            # -- there was no intended earlier firing to have missed.
            return timedelta(0)
        if elapsed < job.every:
            return None
        # Late by however far past the interval we are. Not multiplied out into
        # a backlog: one firing, with the lateness recorded.
        return elapsed - job.every

    def record(self, job: Job | DueJob, state: SessionState, now: datetime | None = None) -> None:
        target = job.job if isinstance(job, DueJob) else job
        self.history.record(target.name, _et(now or state.now), state.session)

    def reset_session(self) -> None:
        """Drop clock-job history. For a new session, never mid-session."""
        self.history = JobHistory()


# --- the standard job set --------------------------------------------------


PREMARKET = frozenset({SessionPhase.PRE_MARKET})
ENTRY = frozenset({SessionPhase.ENTRY_WINDOW})
MANAGING = frozenset(
    {SessionPhase.WARMUP, SessionPhase.ENTRY_WINDOW, SessionPhase.MANAGE_ONLY}
)
REVIEW = frozenset({SessionPhase.AFTER_CLOSE})


def standard_jobs(limits: Any) -> tuple[Job, ...]:
    """The cadence this system actually runs, read from config.

    Order is the contract. Agent 2 screens, Agent 1 profiles what survived, the
    entry scan acts on both, exits run throughout, and the review runs after
    the close.

    The entry manager is declared but only when
    ``execution.entry_reprice_cadence_seconds`` is set. That key is
    deliberately unset -- the entry order manager is specified and unbuilt --
    so reading it raises ``ConfigError`` naming the key rather than letting the
    orchestrator start with a repricing loop it has no implementation for.
    """
    jobs = [
        Job("a2_context", PREMARKET, at=_as_time(limits.get_str("agents.a2.run_at_et"))),
        Job("a1_regime", PREMARKET, at=_as_time(limits.get_str("agents.a1.run_at_et"))),
        Job("entry_scan", ENTRY,
            every=timedelta(minutes=limits.get_int("caps.min_minutes_between_entries"))),
        Job("a5_exit", MANAGING,
            every=timedelta(seconds=limits.get_int("agents.a5.cadence_seconds"))),
        Job("reconcile", MANAGING, every=timedelta(minutes=5)),
        Job("a6_review", REVIEW, at=_as_time(limits.get_str("session.market_close_et"))),
    ]
    return tuple(jobs)


def entry_manager_job(limits: Any) -> Job:
    """The entry repricing loop, on its own clock.

    Separate from :func:`standard_jobs` because the manager does not exist yet
    and its cadence key is unset. Calling this raises ``ConfigError`` naming
    the key, which is the intended failure -- see "Entry order management" in
    CLAUDE.md.
    """
    return Job(
        "entry_manager",
        ENTRY,
        every=timedelta(
            seconds=limits.get_int("execution.entry_reprice_cadence_seconds")
        ),
    )
