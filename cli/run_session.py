"""Drive an autonomous session through the orchestrator.

    python -m cli.run_session --dry-run --date 2026-08-31
    python -m cli.run_session --dry-run --from 09:00 --to 16:30 --step 5m
    python -m cli.run_session --live                      # see the caveat below

**`--dry-run` is complete and runs today.** It drives the real cadence -- the
real session state machine, the real scheduler, the real kill switches -- over
simulated time, with the handlers wired to the replay pipeline. That exercises
every decision path an autonomous session takes except the one below.

**`--live` places real orders.** It drives
:class:`~src.orchestrator.live.LiveSession`, whose handlers run the real
agents, the real prefilter, and the real order path. Every decision -- fills
and skips alike -- is appended to the decision log, which is the artifact the
session is run to produce.

``--until`` bounds the run. Without it the process ticks until the market
closes and Agent 6 has run.

**`--catch-up` lets the pre-market agents run late.** Agent 1 and Agent 2 are
whitelisted to the pre-market phase, so a process started after the open never
produces an eligible set and therefore never trades -- correct for an ordinary
restart, useless for a session begun mid-morning. With this flag they run once
on the first tick regardless of phase. The regime read is still made once per
session and still locked; what changes is that it may be made late, and the
log records that it was.

The cadence itself is shared between the two modes. Whatever `--dry-run`
proves about *when* things happen holds for `--live` unchanged.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

from src.brokers.alpaca.calendar import ET, TradingCalendar
from src.config import ConfigError, load_config
from src.orchestrator.runner import SessionRunner
from src.orchestrator.scheduler import Scheduler, standard_jobs
from src.orchestrator.session import SessionClock

log = logging.getLogger("session")

AGENTS = ("a1", "a2", "a3", "a4", "a5", "a6")


class ExecutionUnavailable(RuntimeError):
    """The live order path does not exist yet, named at the point of use."""


def _duration(text: str) -> timedelta:
    match = re.fullmatch(r"(\d+)(s|m|h)", text.strip().lower())
    if not match:
        raise argparse.ArgumentTypeError(f"expected e.g. 30s, 5m, 1h; got {text!r}")
    n, unit = int(match.group(1)), match.group(2)
    return {"s": timedelta(seconds=n), "m": timedelta(minutes=n), "h": timedelta(hours=n)}[unit]


def _clock_time(text: str) -> time:
    hour, minute = text.split(":")
    return time(int(hour), int(minute))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m cli.run_session")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True,
                      help="simulated time, replay pipeline, no broker (default)")
    mode.add_argument("--live", action="store_true",
                      help="real clock and real account reads; cannot place orders yet")
    p.add_argument("--date", type=lambda s: date.fromisoformat(s), default=None,
                   help="session to drive in dry-run (default: today)")
    p.add_argument("--from", dest="start", type=_clock_time, default=time(8, 0))
    p.add_argument("--to", dest="end", type=_clock_time, default=time(17, 0))
    p.add_argument("--step", type=_duration, default=timedelta(minutes=1))
    p.add_argument("--symbols", default=None, help="comma-separated; default universe")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--until", type=_clock_time, default=None,
                   help="live: stop at this ET time (default: after the close)")
    p.add_argument("--tick", type=_duration, default=timedelta(seconds=30),
                   help="live: wall-clock seconds between ticks")
    p.add_argument("--catch-up", action="store_true",
                   help="live: let the pre-market agents run late on the first tick")
    p.add_argument("--paper-dry", action="store_true",
                   help="live reads and real agent calls, but place no orders")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def build_calendar(config, anchor: date, live: bool) -> TradingCalendar:
    """Alpaca's calendar when we can reach it, weekday arithmetic when we cannot.

    Holidays are Alpaca's answer, not ours. A dry run offline falls back to the
    replay harness's weekday calendar and says so, because deriving sessions
    from weekday arithmetic is how a system ends up trying to trade on
    Thanksgiving -- fine for rehearsing a cadence, not for a live session.
    """
    if live:
        from src.brokers.alpaca.client import build_clients

        clients = build_clients(config)
        return TradingCalendar.fetch(
            clients, anchor - timedelta(days=10), anchor + timedelta(days=120)
        )
    try:
        from src.brokers.alpaca.client import build_clients

        clients = build_clients(config)
        return TradingCalendar.fetch(
            clients, anchor - timedelta(days=10), anchor + timedelta(days=120)
        )
    except Exception as exc:  # noqa: BLE001 -- offline is an expected dry-run state
        from replay.harness import weekday_calendar

        log.warning("calendar unavailable (%s); using weekday arithmetic. "
                    "Holidays are NOT modelled.", exc)
        return weekday_calendar(anchor - timedelta(days=10), anchor + timedelta(days=120))


def dry_run_handlers(config, symbols: tuple[str, ...]) -> tuple[dict, object]:
    """Wire the cadence to the replay pipeline.

    The handlers are thin: the orchestrator decides *when*, and the pipeline
    that already exists decides *what*. Nothing here reimplements a decision.
    """
    from datetime import date as _date

    from replay.bars import synthetic_set
    from replay.harness import ReplayHarness, ReplaySettings
    from replay.rules import rule_transports

    bars = synthetic_set(symbols, _date(2026, 1, 5), 200)
    harness = ReplayHarness(
        config, bars, ReplaySettings(symbols=symbols, warmup_sessions=40),
        rule_transports(list(symbols)),
    )

    counters: dict[str, int] = {}

    def counting(name: str):
        def handler(state):
            counters[name] = counters.get(name, 0) + 1
            log.info("%s  %s", state.describe(), name)
        return handler

    handlers = {job.name: counting(job.name) for job in standard_jobs(config.limits)}
    return handlers, harness


def live_session(config, calendar, symbols, decision_log, dry_run=False):
    """The real handlers, wired to the real order path.

    Transports are the Anthropic ones. There is no rule-stub fallback here on
    purpose: a live session that quietly substituted canned model answers would
    place real orders on decisions no model made.
    """
    from src.agents.transport import AnthropicTransport
    from src.brokers.alpaca.client import build_clients
    from src.orchestrator.live import LiveSession

    transport = AnthropicTransport(config)
    return LiveSession(
        config=config,
        clients=build_clients(config),
        calendar=calendar,
        transports={a: transport for a in AGENTS},
        decision_log=decision_log,
        symbols=tuple(symbols),
        dry_run=dry_run,
    )


def _catch_up(runner, session, live) -> None:
    """Let the pre-market agents run once, late.

    Their phase whitelist is correct for an ordinary restart -- a regime read
    made at 14:00 and traded on immediately is exactly what the pre-market
    cadence exists to prevent. But a process *started* mid-session would then
    never produce an eligible set and never trade, so this is the explicit,
    flagged escape: run them now, once, and let the log show they were late.
    """
    from src.orchestrator.session import SessionPhase

    fake = type(session)(SessionPhase.PRE_MARKET, session.session, session.now,
                         session.windows)
    for name in ("a2_context", "a1_regime"):
        job = next(j for j in runner.scheduler.jobs if j.name == name)
        if runner.scheduler.history.has_run_this_session(name, session.session):
            continue
        log.warning("catch-up: running %s late (phase is %s, not pre-market)",
                    name, session.phase.value)
        try:
            runner.handlers[name](fake)
        except Exception:  # noqa: BLE001
            log.exception("catch-up %s failed", name)
        runner.scheduler.record(job, session, session.now)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    live = bool(args.live)

    try:
        config = load_config()
    except ConfigError as exc:
        raise SystemExit(f"config: {exc}")

    symbols = tuple(
        s.strip().upper() for s in (args.symbols or "").split(",") if s.strip()
    ) or tuple(config.universe.symbols)

    anchor = args.date or date.today()
    calendar = build_calendar(config, anchor, live)
    clock = SessionClock.from_config(config, calendar)

    decision_log = None
    session = None
    harness = None

    if live:
        from src.brokers.alpaca.client import build_clients
        from src.decisionlog.decision_log import DecisionLog

        decision_log = DecisionLog.from_config(config)
        session = live_session(config, calendar, symbols, decision_log,
                               dry_run=args.paper_dry)
        handlers = session.handlers()
        clients = session.clients

        # EQUITY, not sizing capital. The kill switches measure session P&L,
        # and options_buying_power falls by the premium of every position
        # opened -- so reading it here makes each legitimate entry look like a
        # loss. Measured live on 28 Aug: two open positions showed as a 4,505
        # "loss" against a real equity change of -170, tripping the daily-loss
        # and drawdown halts after two trades. sizing_capital is the right
        # number for sizing and the wrong one for P&L.
        equity_reader = lambda: float(  # noqa: E731
            clients.trading.get_account().equity
        )
        failure_reader = session.runner.failure_snapshot
    else:
        handlers, harness = dry_run_handlers(config, symbols)
        equity_reader = lambda: harness.broker.equity  # noqa: E731
        failure_reader = harness.runner.failure_snapshot

    runner = SessionRunner(
        config, clock, handlers,
        scheduler=Scheduler(standard_jobs(config.limits)),
        equity_reader=equity_reader,
        decision_log=decision_log,
        agent_failure_reader=failure_reader,
    )

    # Kill-switch verdicts are the orchestrator's, but the log is the session's.
    if session is not None:
        original = runner.evaluate_halt

        def evaluate_and_log(state):
            result = original(state)
            session.log_kill_switches(runner.halt.verdicts)
            return result

        runner.evaluate_halt = evaluate_and_log

    try:
        if live:
            return _run_live(runner, clock, args, session)

        start = datetime.combine(anchor, args.start, tzinfo=ET)
        end = datetime.combine(anchor, args.end, tzinfo=ET)
        runner.run_range(start, end, args.step)
        payload = {"summary": runner.summary()}
    finally:
        if harness is not None:
            harness.close()

    text = json.dumps(payload, indent=2, default=str)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n{payload['summary']['ticks']} ticks; jobs run "
          f"{payload['summary']['jobs_run']}", file=sys.stderr)
    print("dry run: cadence is real, execution is stubbed.", file=sys.stderr)
    return 0


def _run_live(runner, clock, args, session) -> int:
    """Tick against the wall clock until the stop time, then report.

    Sleeps between ticks rather than spinning. The scheduler decides what is
    due on each one, so the tick interval only bounds latency -- shortening it
    does not make a 30-minute agent run more often.
    """
    import time as _time

    stop_at = (
        datetime.combine(date.today(), args.until, tzinfo=ET)
        if args.until
        else datetime.combine(date.today(), clock.windows.market_close, tzinfo=ET)
        + timedelta(minutes=10)
    )
    started = datetime.now(tz=ET)
    state = clock.state(started)
    print(f"live session: {state.describe()} -> stopping {stop_at:%H:%M %Z}",
          file=sys.stderr)
    if args.paper_dry:
        print("paper-dry: real reads and real agent calls, NO orders", file=sys.stderr)

    if args.catch_up:
        runner.evaluate_halt(state)
        _catch_up(runner, state, session)

    while datetime.now(tz=ET) < stop_at:
        now = datetime.now(tz=ET)
        result = runner.tick(now)
        if result.did_work or result.skipped_halted:
            log.info("tick %s ran=%s skipped=%s failed=%s",
                     now.strftime("%H:%M:%S"), result.ran,
                     result.skipped_halted, [j for j, _ in result.failed])
        remaining = (stop_at - datetime.now(tz=ET)).total_seconds()
        if remaining <= 0:
            break
        _time.sleep(min(args.tick.total_seconds(), remaining))

    summary = runner.summary()
    summary["live"] = {
        "entries": session.entries_this_session,
        "orders_placed": session.orders_placed,
        "fills": session.fills,
        "skips": len(session.skips),
        "eligible": sorted(session.eligible),
        "profiled": sorted(session.profiles),
        "decision_log": str(session.decision_log.path),
    }
    text = json.dumps(summary, indent=2, default=str)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)
    session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
