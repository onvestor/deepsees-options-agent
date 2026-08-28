"""The session runner: cadence, halts, and one tick at a time.

This is the loop an autonomous session actually runs. It owns three things and
deliberately nothing else:

1. **When** work happens -- delegated to :mod:`~src.orchestrator.session` and
   :mod:`~src.orchestrator.scheduler`.
2. **Whether** new entries are permitted -- the deterministic kill switches,
   plus the agent failure rate the runner already tracks.
3. **That a failure in one job does not take down the loop.**

**What it does not own is what each job does.** The handlers are supplied by
the caller. That keeps the runner testable against stubs, keeps the same
cadence usable by a live session and a dry run, and -- the reason that matters
here -- means the loop is finished and provable before the order builder it
will eventually call exists.

**The halt asymmetry is the safety property.** A fired kill switch stops *new
entries* and nothing else. Exits, reconciliation and the review keep running,
because a halt is a statement about opening risk, not about the risk already
open. A loop that stopped managing positions when it stopped opening them would
turn a bad session into an unmanaged one, which is strictly worse.

**A halt is sticky for the session.** It is re-evaluated every tick, but once
fired it stays fired until the session rolls, per
``killswitch.halt_resets_next_session``. Letting a halt clear because equity
ticked back over the threshold would have the system resume trading in exactly
the conditions that stopped it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Callable, Mapping

from src.orchestrator.scheduler import DueJob, Job, Scheduler, standard_jobs
from src.orchestrator.session import SessionClock, SessionPhase, SessionState, _et
from src.risk.killswitch import (
    KillSwitchLimits,
    KillSwitchState,
    SwitchVerdict,
    evaluate_kill_switches,
    fired_switches,
    is_halted,
)

log = logging.getLogger(__name__)

Handler = Callable[[SessionState], Any]

# Jobs that open risk. These are the only ones a halt stops.
ENTRY_JOBS = frozenset({"entry_scan", "entry_manager"})


class HandlerMissing(RuntimeError):
    """A scheduled job with nothing wired to it, named."""


@dataclass
class TickResult:
    """What one tick did. One of these per loop iteration, for the log."""

    at: datetime
    state: SessionState
    ran: tuple[str, ...] = ()
    skipped_halted: tuple[str, ...] = ()
    failed: tuple[tuple[str, str], ...] = ()
    late: tuple[tuple[str, float], ...] = ()
    halted: bool = False
    halt_reasons: tuple[str, ...] = ()

    @property
    def did_work(self) -> bool:
        return bool(self.ran or self.failed)

    def as_dict(self) -> dict[str, Any]:
        return {
            "at": self.at.isoformat(),
            "session": self.state.session.isoformat(),
            "phase": self.state.phase.value,
            "ran": list(self.ran),
            "skipped_halted": list(self.skipped_halted),
            "failed": [{"job": j, "error": e} for j, e in self.failed],
            "late_seconds": {j: s for j, s in self.late},
            "halted": self.halted,
            "halt_reasons": list(self.halt_reasons),
        }


@dataclass
class HaltState:
    """Whether new entries are permitted, and why not.

    Sticky within a session by design. ``roll`` is the only way out, and it is
    called on a session change rather than on the numbers improving.
    """

    halted: bool = False
    reasons: tuple[str, ...] = ()
    session: date | None = None
    verdicts: tuple[SwitchVerdict, ...] = ()

    def apply(self, verdicts: tuple[SwitchVerdict, ...], session: date) -> None:
        self.verdicts = verdicts
        if is_halted(verdicts):
            new = fired_switches(verdicts)
            if not self.halted:
                log.warning("kill switch fired -- new entries halted: %s", ", ".join(new))
            self.halted = True
            self.reasons = tuple(sorted(set(self.reasons) | set(new)))
        self.session = session

    def roll(self, session: date, resets: bool) -> None:
        if self.session == session:
            return
        if resets:
            if self.halted:
                log.info("new session %s -- halt cleared", session)
            self.halted, self.reasons = False, ()
        self.session = session


class SessionRunner:
    """Drives one process through however many sessions it is asked to.

    ``handlers`` maps a job name to a callable taking the
    :class:`~src.orchestrator.session.SessionState`. Every job produced by the
    scheduler must have one; a missing handler raises at construction rather
    than at 09:15 on a live session.
    """

    def __init__(
        self,
        config: Any,
        clock: SessionClock,
        handlers: Mapping[str, Handler],
        scheduler: Scheduler | None = None,
        decision_log: Any | None = None,
        equity_reader: Callable[[], float] | None = None,
        agent_failure_reader: Callable[[], Any] | None = None,
    ) -> None:
        self.config = config
        self.limits = config.limits
        self.clock = clock
        self.scheduler = scheduler or Scheduler(standard_jobs(config.limits))
        self.handlers = dict(handlers)
        self.log = decision_log
        self.equity_reader = equity_reader
        self.agent_failure_reader = agent_failure_reader

        missing = [j.name for j in self.scheduler.jobs if j.name not in self.handlers]
        if missing:
            raise HandlerMissing(
                f"no handler wired for scheduled job(s) {missing}. Every job the "
                "scheduler can produce needs one -- discovering this mid-session "
                "would mean a cadence that silently does nothing."
            )

        self.kill_limits = KillSwitchLimits.from_limits(config.limits)
        self.halt = HaltState()
        self._session_open_equity: dict[date, float] = {}
        self._session_peak: dict[date, float] = {}
        self.consecutive_losses = 0
        self.broker_error_streak = 0
        self.ticks: list[TickResult] = []

    # -- the halt -----------------------------------------------------------

    def evaluate_halt(self, state: SessionState) -> TickResult | None:
        """Re-evaluate the kill switches for this session."""
        self.halt.roll(state.session, self.kill_limits.halt_resets_next_session)
        if self.equity_reader is None:
            return None

        equity = float(self.equity_reader())
        opening = self._session_open_equity.setdefault(state.session, equity)
        peak = max(self._session_peak.get(state.session, equity), equity)
        self._session_peak[state.session] = peak

        verdicts = evaluate_kill_switches(
            KillSwitchState(
                start_of_day_equity=opening,
                current_equity=equity,
                session_peak_equity=peak,
                consecutive_losing_trades=self.consecutive_losses,
                broker_error_streak=self.broker_error_streak,
            ),
            self.kill_limits,
        )
        self.halt.apply(verdicts, state.session)

        # The agent failure rate is a separate input on the same footing: a
        # model failing repeatedly is a broken input to every decision
        # downstream, so it halts entries the way a loss streak does.
        if self.agent_failure_reader is not None:
            snapshot = self.agent_failure_reader()
            if getattr(snapshot, "halts_new_entries", False):
                if not self.halt.halted:
                    log.warning("agent failure rate halts new entries: %s",
                                getattr(snapshot, "reason", ""))
                self.halt.halted = True
                self.halt.reasons = tuple(
                    sorted(set(self.halt.reasons) | {"agent_failure_rate"})
                )
        return None

    # -- one tick -----------------------------------------------------------

    def tick(self, now: datetime) -> TickResult:
        """Evaluate the clock, run whatever is due, and record it."""
        state = self.clock.state(now)
        result = TickResult(at=_et(now), state=state)

        if not state.is_trading_day:
            self.ticks.append(result)
            return result

        self.evaluate_halt(state)
        result.halted = self.halt.halted
        result.halt_reasons = self.halt.reasons

        for due in self.scheduler.due(state, now):
            name = due.job.name
            if self.halt.halted and name in ENTRY_JOBS:
                # The asymmetry: opening stops, managing does not.
                result.skipped_halted = result.skipped_halted + (name,)
                # Recorded as run so the cadence does not build a backlog of
                # entry scans to fire the instant a halt clears.
                self.scheduler.record(due, state, now)
                continue

            if due.is_late:
                result.late = result.late + ((name, due.late_by.total_seconds()),)
                log.info("job %s is %.0fs late", name, due.late_by.total_seconds())

            try:
                self.handlers[name](state)
            except Exception as exc:  # noqa: BLE001 -- a job failure is data
                # One job failing must not take down the loop. An entry-path
                # failure has already been turned into "no trade" by the agent
                # layer; anything reaching here is unexpected and is recorded.
                log.exception("job %s failed", name)
                result.failed = result.failed + ((name, f"{type(exc).__name__}: {exc}"),)
            else:
                result.ran = result.ran + (name,)
            finally:
                # Recorded either way. A job that throws every time must not
                # spin the loop by staying permanently due.
                self.scheduler.record(due, state, now)

        self.ticks.append(result)
        return result

    # -- driving ------------------------------------------------------------

    def run_range(
        self, start: datetime, end: datetime, step: timedelta
    ) -> list[TickResult]:
        """Tick from ``start`` to ``end``. Deterministic; no sleeping.

        This is how a session is replayed or dry-run. A live process uses the
        same :meth:`tick` against a real clock, so the two share every decision
        path and differ only in what advances time.
        """
        if step <= timedelta(0):
            raise ValueError(f"step must be positive, got {step}")
        out: list[TickResult] = []
        moment = _et(start)
        finish = _et(end)
        while moment <= finish:
            out.append(self.tick(moment))
            moment += step
        return out

    # -- reporting ----------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        ran: dict[str, int] = {}
        failed: dict[str, int] = {}
        for tick in self.ticks:
            for name in tick.ran:
                ran[name] = ran.get(name, 0) + 1
            for name, _ in tick.failed:
                failed[name] = failed.get(name, 0) + 1
        sessions = sorted({t.state.session for t in self.ticks if t.state.is_trading_day})
        return {
            "ticks": len(self.ticks),
            "sessions": [s.isoformat() for s in sessions],
            "jobs_run": dict(sorted(ran.items(), key=lambda kv: -kv[1])),
            "jobs_failed": dict(sorted(failed.items(), key=lambda kv: -kv[1])),
            "halted": self.halt.halted,
            "halt_reasons": list(self.halt.reasons),
            "entry_scans_skipped_by_halt": sum(
                1 for t in self.ticks if t.skipped_halted
            ),
            "late_firings": sum(len(t.late) for t in self.ticks),
        }
