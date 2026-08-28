"""Run an offline replay. No broker, no provider, no network.

    python -m cli.replay_session --synthetic --sessions 120
    python -m cli.replay_session --bars data/bars --symbols SPY,QQQ
    python -m cli.replay_session --bars data/bars --record logs/agents.jsonl
    python -m cli.replay_session --bars data/bars --replay logs/agents.jsonl

Three transport modes, and which one to use depends on what is being tested:

``--rules`` (the default)
    Deterministic policy stubs. No model at all. Use this to test the
    *pipeline* -- prefilter, sizing, caps, exits -- with model variance removed
    entirely. It still renders real prompts, so ``prompts/`` is required; what
    it does not need is an API key or a network.

``--record PATH``
    Calls the real provider and writes every response to PATH. Costs money and
    needs ``prompts/`` and ``ANTHROPIC_API_KEY``. Do this once.

``--replay PATH``
    Replays that recording offline, free and reproducible. This is the mode for
    iterating on anything downstream of the model's answers.

A recording is keyed by the hash of the rendered prompt, so editing a prompt
turns its entries into misses rather than stale hits, and the run stops naming
the agent that needs re-recording.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path

from src.config import ConfigError, load_config

from replay.bars import BarError, load_directory, synthetic_set
from replay.broker import FillModel
from replay.chain import ChainModel
from replay.harness import ReplayHarness, ReplaySettings
from replay.rules import rule_transports
from replay.transport import RecordedTransport, RecordingMiss, RecordingTransport

log = logging.getLogger("replay")

AGENTS = ("a1", "a2", "a3", "a4", "a5", "a6")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m cli.replay_session",
        description="Replay a session offline against the full pipeline.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bars", type=Path, help="directory of <SYMBOL>.csv daily bars")
    source.add_argument(
        "--synthetic", action="store_true",
        help="generate a deterministic bar series instead of loading one",
    )

    parser.add_argument("--symbols", default="SPY,QQQ", help="comma-separated")
    parser.add_argument("--sessions", type=int, default=180,
                        help="synthetic series length")
    parser.add_argument("--start", type=_as_date, help="first session, YYYY-MM-DD")
    parser.add_argument("--end", type=_as_date, help="last session, YYYY-MM-DD")
    parser.add_argument("--equity", type=float, default=100_000.0)
    parser.add_argument("--warmup", type=int, default=60,
                        help="sessions of bars before the first decision")
    parser.add_argument(
        "--cross", type=float, default=1.0,
        help="fraction of the half-spread a fill crosses; 1.0 pays the full ask",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--rules", action="store_true",
                      help="deterministic stubs, no model (default)")
    mode.add_argument("--record", type=Path, metavar="PATH",
                      help="call the real provider and record every response")
    mode.add_argument("--replay", type=Path, metavar="PATH",
                      help="replay a recording, offline")

    parser.add_argument("--out", type=Path, help="write the JSON report here")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def _as_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


def build_transports(args: argparse.Namespace, config) -> dict:
    """Wire one transport per agent for the chosen mode."""
    if args.replay:
        try:
            return {
                agent: RecordedTransport.from_file(args.replay, agent=agent)
                for agent in AGENTS
            }
        except RecordingMiss as exc:
            raise SystemExit(f"replay: {exc}")

    if args.record:
        from src.agents.transport import AnthropicTransport

        real = AnthropicTransport(config)
        return {
            agent: RecordingTransport(inner=real, path=args.record, agent=agent)
            for agent in AGENTS
        }

    return rule_transports(symbols=None)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    symbols = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())
    if not symbols:
        raise SystemExit("--symbols is empty")

    try:
        config = load_config()
    except ConfigError as exc:
        raise SystemExit(f"config: {exc}")

    try:
        if args.synthetic:
            start = args.start or date(2026, 1, 2)
            bars = synthetic_set(symbols, start, args.sessions)
        else:
            bars = load_directory(args.bars, symbols)
    except BarError as exc:
        raise SystemExit(f"bars: {exc}")

    settings = ReplaySettings(
        symbols=symbols,
        starting_equity=args.equity,
        chain=ChainModel(),
        fills=FillModel(cross_fraction=args.cross),
        warmup_sessions=args.warmup,
    )

    harness = ReplayHarness(config, bars, settings, build_transports(args, config))
    try:
        report = harness.run(start=args.start, end=args.end)
    except RecordingMiss as exc:
        raise SystemExit(
            f"replay: {exc}\nRe-record with --record before replaying an edited prompt."
        )
    finally:
        harness.close()

    payload = report.as_dict()
    text = json.dumps(payload, indent=2, default=str)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        log.info("report written to %s", args.out)
    print(text)

    _print_summary(payload)
    return 0


def _print_summary(payload: dict) -> None:
    broker = payload["broker"]
    print(
        f"\n{payload['sessions_replayed']} sessions | "
        f"{payload['entries']} entries | {payload['exits']} exits | "
        f"{broker['trades']} closed ({broker['wins']}W/{broker['losses']}L) | "
        f"P&L {broker['realized_pnl']:+.2f}",
        file=sys.stderr,
    )
    print(
        "P&L above is a comparison between runs, not a market result -- the "
        "chain is modelled. See 'caveats' in the report.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    raise SystemExit(main())
