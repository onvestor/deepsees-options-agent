"""Step 9 -- the session state machine, the cadence, and the halt.

Every test drives an injected clock. Nothing here reads the wall clock, which
is the whole reason the boundary minutes are testable: the minute before the
entry window opens, the minute after it shuts, and the tick on which a halt
flips are the only minutes where these rules can be wrong.

The property with the most weight is the **halt asymmetry**: a fired kill
switch stops new entries and nothing else. A loop that also stopped managing
open positions would convert a bad session into an unmanaged one.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta

import pytest

from src.brokers.alpaca.calendar import ET, TradingCalendar
from src.orchestrator.runner import HandlerMissing, SessionRunner
from src.orchestrator.scheduler import (
    ENTRY,
    MANAGING,
    PREMARKET,
    Job,
    JobHistory,
    Scheduler,
    standard_jobs,
)
from src.orchestrator.session import (
    SessionClock,
    SessionPhase,
    SessionWindows,
)

SESSION = date(2026, 8, 28)          # a Friday
WEEKEND = date(2026, 8, 29)


def _calendar() -> TradingCalendar:
    days = tuple(
        date(2026, 8, 24) + timedelta(days=i)
        for i in range(14)
        if (date(2026, 8, 24) + timedelta(days=i)).weekday() < 5
    )
    return TradingCalendar(
        sessions=days,
        closes={d: datetime.combine(d, time(16, 0), tzinfo=ET) for d in days},
    )


def _windows(**kw) -> SessionWindows:
    base = dict(
        market_open=time(9, 30), market_close=time(16, 0),
        first_entry=time(9, 45), last_entry=time(15, 0),
        flat_by=time(15, 45), skip_dates=frozenset(),
    )
    base.update(kw)
    return SessionWindows(**base)


@pytest.fixture
def clock():
    return SessionClock(_windows(), _calendar())


def at(hour: int, minute: int, day: date = SESSION) -> datetime:
    return datetime.combine(day, time(hour, minute), tzinfo=ET)


# --- the phases ------------------------------------------------------------


@pytest.mark.parametrize(
    "hour,minute,expected",
    [
        (8, 0, SessionPhase.PRE_MARKET),
        (9, 29, SessionPhase.PRE_MARKET),
        (9, 30, SessionPhase.WARMUP),
        (9, 44, SessionPhase.WARMUP),
        (9, 45, SessionPhase.ENTRY_WINDOW),
        (14, 59, SessionPhase.ENTRY_WINDOW),
        (15, 0, SessionPhase.MANAGE_ONLY),
        (15, 59, SessionPhase.MANAGE_ONLY),
        (16, 0, SessionPhase.AFTER_CLOSE),
    ],
)
def test_phase_boundaries(clock, hour, minute, expected):
    assert clock.state(at(hour, minute)).phase is expected


def test_a_weekend_is_not_a_session(clock):
    assert clock.state(at(11, 0, WEEKEND)).phase is SessionPhase.NOT_A_SESSION


def test_a_skipped_day_is_distinguishable_from_a_closed_market():
    """Same "do not trade", different reasons. Conflating them in the log makes
    a quiet week unattributable."""
    clock = SessionClock(_windows(skip_dates=frozenset({SESSION})), _calendar())
    assert clock.state(at(11, 0)).phase is SessionPhase.SKIPPED
    assert clock.state(at(11, 0, WEEKEND)).phase is SessionPhase.NOT_A_SESSION


def test_neither_permits_any_action(clock):
    skipping = SessionClock(_windows(skip_dates=frozenset({SESSION})), _calendar())
    for state in (clock.state(at(11, 0, WEEKEND)), skipping.state(at(11, 0))):
        assert not state.is_trading_day
        assert not state.may_open
        assert not state.may_manage


# --- what each phase permits ----------------------------------------------


def test_entries_are_confined_to_the_entry_window(clock):
    assert clock.state(at(9, 45)).may_open
    assert not clock.state(at(9, 44)).may_open
    assert not clock.state(at(15, 0)).may_open
    assert not clock.state(at(8, 0)).may_open


def test_management_outlives_the_entry_window(clock):
    """The phase that stops new entries must not stop the exit loop."""
    late = clock.state(at(15, 30))
    assert not late.may_open
    assert late.may_manage


def test_premarket_agents_only_run_premarket(clock):
    assert clock.state(at(9, 15)).may_run_premarket_agents
    assert not clock.state(at(10, 15)).may_run_premarket_agents


def test_review_only_runs_after_the_close(clock):
    assert clock.state(at(16, 30)).may_review
    assert not clock.state(at(15, 59)).may_review


def test_past_flat_by_is_not_an_instruction_to_go_flat(clock):
    """There is no intraday flat rule -- the swing design holds overnight."""
    state = clock.state(at(15, 46))
    assert state.past_flat_by
    assert state.may_manage


# --- window validation -----------------------------------------------------


def test_an_entry_window_after_the_flat_time_is_refused():
    """Otherwise the loop opens a position it is about to be told to close."""
    with pytest.raises(ValueError, match="last_entry"):
        _windows(last_entry=time(15, 50), flat_by=time(15, 45)).validate()


def test_a_first_entry_before_the_open_is_refused():
    with pytest.raises(ValueError, match="market_open"):
        _windows(market_open=time(9, 30), first_entry=time(9, 0)).validate()


# --- the scheduler ---------------------------------------------------------


def test_a_clock_job_fires_once_per_session(clock):
    job = Job("a1", PREMARKET, at=time(9, 15))
    sched = Scheduler([job])

    assert not sched.due(clock.state(at(9, 14)))
    state = clock.state(at(9, 16))
    assert [d.job.name for d in sched.due(state)] == ["a1"]

    sched.record(job, state, at(9, 16))
    assert not sched.due(clock.state(at(9, 20)))


def test_a_clock_job_fires_again_on_the_next_session(clock):
    job = Job("a1", PREMARKET, at=time(9, 15))
    sched = Scheduler([job])
    first = clock.state(at(9, 16))
    sched.record(job, first, at(9, 16))

    monday = date(2026, 8, 31)
    assert [d.job.name for d in sched.due(clock.state(at(9, 16, monday)))] == ["a1"]


def test_a_missed_clock_job_runs_late_rather_than_not_at_all(clock):
    """The process was down at 09:15. It should still profile, and the record
    should say it was late."""
    sched = Scheduler([Job("a1", PREMARKET, at=time(9, 15))])
    [due] = sched.due(clock.state(at(9, 28)))
    assert due.is_late
    assert due.late_by == timedelta(minutes=13)


def test_an_interval_job_does_not_replay_a_backlog(clock):
    """Firing once per missed interval would run Agent 5 twenty times in a
    second on restart, each against the same stale mark."""
    job = Job("a5", MANAGING, every=timedelta(minutes=30))
    sched = Scheduler([job])
    state = clock.state(at(10, 0))
    sched.record(job, state, at(10, 0))

    due = sched.due(clock.state(at(13, 0)))
    assert len(due) == 1
    assert due[0].late_by == timedelta(hours=2, minutes=30)


def test_an_interval_job_is_due_immediately_but_not_late_on_first_run(clock):
    sched = Scheduler([Job("a5", MANAGING, every=timedelta(minutes=30))])
    [due] = sched.due(clock.state(at(10, 0)))
    assert not due.is_late


def test_a_job_outside_its_phase_is_never_due(clock):
    """However long it has been. This is what stops an entry scan firing at
    15:59 because the process slept through the window."""
    sched = Scheduler([Job("entry", ENTRY, every=timedelta(minutes=1))])
    assert not sched.due(clock.state(at(15, 59)))
    assert sched.due(clock.state(at(10, 0)))


def test_a_job_must_be_exactly_one_shape():
    with pytest.raises(ValueError, match="exactly one"):
        Job("bad", ENTRY)
    with pytest.raises(ValueError, match="exactly one"):
        Job("bad", ENTRY, at=time(9, 0), every=timedelta(minutes=1))


def test_duplicate_job_names_are_refused():
    with pytest.raises(ValueError, match="duplicate"):
        Scheduler([Job("a", ENTRY, every=timedelta(minutes=1)),
                   Job("a", ENTRY, every=timedelta(minutes=2))])


def test_standard_jobs_screen_before_profiling():
    """a2 produces the eligible set a1 profiles. Declaration order is the
    execution order, so this ordering is the contract."""
    from src.config import load_config

    names = [j.name for j in standard_jobs(load_config().limits)]
    assert names.index("a2_context") < names.index("a1_regime")


def test_the_entry_manager_has_its_own_cadence_key():
    """It must not borrow agents.a5.cadence_seconds.

    Entry repricing and exit management have different urgency, and a shared
    key means tuning one silently moves the other. The fixture config sets both
    to distinct synthetic values precisely so this is checkable.
    """
    from src.config import load_config
    from src.orchestrator.scheduler import entry_manager_job
    from tests.test_prefilter import _pinned

    limits = _pinned(
        load_config().limits,
        **{"execution.entry_reprice_cadence_seconds": 11,
           "agents.a5.cadence_seconds": 2200},
    )
    assert entry_manager_job(limits).every == timedelta(seconds=11)


def test_an_unset_entry_cadence_names_the_key():
    """The operator's config leaves `execution:` empty on purpose, so the entry
    manager cannot start half-configured."""
    from src.config import ConfigError, Section, load_config

    data = load_config().limits.as_dict()
    data["execution"] = {}
    from src.orchestrator.scheduler import entry_manager_job

    with pytest.raises(ConfigError, match="entry_reprice_cadence_seconds"):
        entry_manager_job(Section(data, "test"))


# --- the runner ------------------------------------------------------------


class Recorder:
    def __init__(self, fail: bool = False):
        self.calls = 0
        self.fail = fail

    def __call__(self, state):
        self.calls += 1
        if self.fail:
            raise RuntimeError("handler blew up")


def _runner(clock, jobs, equity=None, **handlers):
    from src.config import load_config

    return SessionRunner(
        load_config(), clock, handlers,
        scheduler=Scheduler(jobs),
        equity_reader=equity,
    )


def test_a_missing_handler_is_refused_at_construction(clock):
    """Discovering this at 09:15 on a live session is the alternative."""
    with pytest.raises(HandlerMissing, match="entry_scan"):
        _runner(clock, [Job("entry_scan", ENTRY, every=timedelta(minutes=1))])


def test_a_tick_runs_what_is_due(clock):
    scan = Recorder()
    runner = _runner(clock, [Job("entry_scan", ENTRY, every=timedelta(minutes=5))],
                     entry_scan=scan)
    result = runner.tick(at(10, 0))
    assert result.ran == ("entry_scan",)
    assert scan.calls == 1


def test_nothing_runs_on_a_non_session_day(clock):
    scan = Recorder()
    runner = _runner(clock, [Job("entry_scan", ENTRY, every=timedelta(minutes=5))],
                     entry_scan=scan)
    result = runner.tick(at(10, 0, WEEKEND))
    assert result.ran == ()
    assert scan.calls == 0


def test_a_failing_job_does_not_stop_the_loop(clock):
    bad, good = Recorder(fail=True), Recorder()
    runner = _runner(
        clock,
        [Job("entry_scan", ENTRY, every=timedelta(minutes=5)),
         Job("a5_exit", MANAGING, every=timedelta(minutes=5))],
        entry_scan=bad, a5_exit=good,
    )
    result = runner.tick(at(10, 0))
    assert result.failed and result.failed[0][0] == "entry_scan"
    assert "a5_exit" in result.ran
    assert good.calls == 1


def test_a_permanently_failing_job_does_not_spin_the_loop(clock):
    """It is recorded as having run, so it waits its interval like any other."""
    bad = Recorder(fail=True)
    runner = _runner(clock, [Job("entry_scan", ENTRY, every=timedelta(minutes=30))],
                     entry_scan=bad)
    runner.tick(at(10, 0))
    runner.tick(at(10, 1))
    assert bad.calls == 1


# --- the halt asymmetry ----------------------------------------------------


def _halting_equity():
    """Equity far enough down to fire the daily-loss switch on the second read."""
    values = iter([100_000.0, 50_000.0] + [50_000.0] * 50)
    return lambda: next(values)


def test_a_halt_stops_entries_and_nothing_else(clock):
    entry, exit_job, review = Recorder(), Recorder(), Recorder()
    runner = _runner(
        clock,
        [Job("entry_scan", ENTRY, every=timedelta(minutes=1)),
         Job("a5_exit", MANAGING, every=timedelta(minutes=1)),
         Job("reconcile", MANAGING, every=timedelta(minutes=1))],
        equity=_halting_equity(),
        entry_scan=entry, a5_exit=exit_job, reconcile=review,
    )
    runner.tick(at(10, 0))          # healthy
    assert entry.calls == 1

    result = runner.tick(at(10, 5))  # equity collapses
    assert result.halted
    assert "entry_scan" in result.skipped_halted
    assert "a5_exit" in result.ran          # management continues
    assert "reconcile" in result.ran
    assert entry.calls == 1                 # no new entry


def test_a_halt_is_sticky_within_the_session(clock):
    """Letting it clear because equity ticked back up would resume trading in
    exactly the conditions that stopped it."""
    values = iter([100_000.0, 50_000.0, 99_000.0] + [99_000.0] * 50)
    entry = Recorder()
    runner = _runner(
        clock, [Job("entry_scan", ENTRY, every=timedelta(minutes=1))],
        equity=lambda: next(values), entry_scan=entry,
    )
    runner.tick(at(10, 0))
    assert runner.tick(at(10, 5)).halted
    assert runner.tick(at(10, 10)).halted    # equity recovered; still halted


def test_a_halt_clears_on_the_next_session(clock):
    values = iter([100_000.0, 50_000.0] + [100_000.0] * 50)
    runner = _runner(
        clock, [Job("entry_scan", ENTRY, every=timedelta(minutes=1))],
        equity=lambda: next(values), entry_scan=Recorder(),
    )
    runner.tick(at(10, 0))
    assert runner.tick(at(10, 5)).halted
    assert not runner.tick(at(10, 0, date(2026, 8, 31))).halted


def test_a_halted_entry_scan_does_not_build_a_backlog(clock):
    """Otherwise every suppressed scan fires the instant a halt clears."""
    entry = Recorder()
    runner = _runner(
        clock, [Job("entry_scan", ENTRY, every=timedelta(minutes=30))],
        equity=_halting_equity(), entry_scan=entry,
    )
    runner.tick(at(10, 0))
    runner.tick(at(10, 5))
    result = runner.tick(at(10, 6))
    assert result.skipped_halted == ()     # not due again yet


def test_the_agent_failure_rate_halts_entries_too(clock):
    """A model failing repeatedly is a broken input to every decision
    downstream, so it halts entries the way a loss streak does."""
    from src.config import load_config

    class Snapshot:
        halts_new_entries = True
        reason = "4/5 failed (80%)"

    entry, exits = Recorder(), Recorder()
    runner = SessionRunner(
        load_config(), clock, {"entry_scan": entry, "a5_exit": exits},
        scheduler=Scheduler([Job("entry_scan", ENTRY, every=timedelta(minutes=1)),
                             Job("a5_exit", MANAGING, every=timedelta(minutes=1))]),
        equity_reader=lambda: 100_000.0,
        agent_failure_reader=lambda: Snapshot(),
    )
    result = runner.tick(at(10, 0))
    assert result.halted
    assert "agent_failure_rate" in result.halt_reasons
    assert "a5_exit" in result.ran


# --- driving a whole session ----------------------------------------------


def test_a_full_session_runs_the_expected_cadence(clock):
    from src.config import load_config

    handlers = {name: Recorder() for name in
                ("a2_context", "a1_regime", "entry_scan", "a5_exit",
                 "reconcile", "a6_review")}
    runner = SessionRunner(
        load_config(), clock, handlers,
        scheduler=Scheduler(standard_jobs(load_config().limits)),
        equity_reader=lambda: 100_000.0,
    )
    runner.run_range(at(8, 0), at(17, 0), timedelta(minutes=1))
    summary = runner.summary()

    # Premarket agents: exactly once each.
    assert summary["jobs_run"]["a2_context"] == 1
    assert summary["jobs_run"]["a1_regime"] == 1
    assert summary["jobs_run"]["a6_review"] == 1
    # Exits run through the managing phases, entries only inside the window.
    assert summary["jobs_run"]["a5_exit"] > summary["jobs_run"]["entry_scan"]
    assert not summary["jobs_failed"]
    assert not summary["halted"]


def test_the_run_is_deterministic(clock):
    from src.config import load_config

    def once():
        handlers = {name: Recorder() for name in
                    ("a2_context", "a1_regime", "entry_scan", "a5_exit",
                     "reconcile", "a6_review")}
        runner = SessionRunner(
            load_config(), clock, handlers,
            scheduler=Scheduler(standard_jobs(load_config().limits)),
            equity_reader=lambda: 100_000.0,
        )
        runner.run_range(at(8, 0), at(17, 0), timedelta(minutes=1))
        return runner.summary()

    assert once() == once()


def test_a_zero_step_is_refused(clock):
    runner = _runner(clock, [Job("entry_scan", ENTRY, every=timedelta(minutes=1))],
                     entry_scan=Recorder())
    with pytest.raises(ValueError, match="step must be positive"):
        runner.run_range(at(9, 0), at(10, 0), timedelta(0))


# --- the kill switch measures P&L, not committed capital -------------------


def test_the_kill_switch_reads_equity_not_buying_power():
    """Measured live on 28 Aug 2026, and it halted the session.

    ``options_buying_power`` falls by the premium of every position opened, so
    feeding it to the kill switch makes each legitimate entry look like a loss.
    Two open positions showed as a 4,505 "loss" against a real equity change of
    -170 and tripped the daily-loss and drawdown halts after two trades.

    Asserted against the wiring in cli.run_session rather than against a
    running session, because the failure is a one-line reader choice and this
    is the line.
    """
    from pathlib import Path

    source = Path(__file__).resolve().parents[1].joinpath(
        "cli", "run_session.py"
    ).read_text(encoding="utf-8")
    live_block = source.split("if live:", 1)[1].split("else:", 1)[0]

    assert "get_account().equity" in live_block, (
        "the live equity_reader must read account.equity"
    )
    assert "sizing_capital(" not in live_block, (
        "sizing_capital is buying power -- correct for sizing, wrong for P&L"
    )


def test_a_loss_measured_against_opening_equity_fires_the_switch(clock):
    """The switch itself is fine; only its input was wrong. A genuine 5% drop
    still halts."""
    values = iter([100_000.0, 95_000.0] + [95_000.0] * 50)
    runner = _runner(
        clock, [Job("entry_scan", ENTRY, every=timedelta(minutes=1))],
        equity=lambda: next(values), entry_scan=Recorder(),
    )
    runner.tick(at(10, 0))
    result = runner.tick(at(10, 5))
    assert result.halted
    assert "daily_loss_halt_pct" in result.halt_reasons
