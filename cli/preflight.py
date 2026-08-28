"""Verify the account the keys actually address, before a session runs.

    python -m cli.preflight
    python -m cli.preflight --expect-account PA0000000000 --expect-equity 100000
    python -m cli.preflight --require-flat --require-level 3

Read-only. Places no orders, and is safe to run on a schedule.

**The account number is never stored in this repository.** It is operator state
that lives in the Alpaca dashboard and on the submission form, and
``CLAUDE.md`` is explicit that nothing in source or config branches on one. So
``--expect-account`` is an argument supplied at invocation, defaulting to no
check -- the number reaches this process from the operator's command line or
their scheduled task, never from a file that could be committed.

**Why this exists at all.** Which account a key pair addresses is a property of
the keys, not a configured value, so swapping accounts is a key change and
nothing in the code can tell you it happened. The only way to know which
account you are pointed at is to ask the broker, which is exactly what
``account_summary`` already does. Running this after a key swap is the
difference between believing the switch worked and knowing it did.

Exit codes are distinct so a scheduled run's failure is diagnosable from the
task history alone:

    0  every requested check passed
    2  configuration or credentials could not be loaded
    3  a check failed (wrong account, wrong equity, not flat, ...)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.brokers.alpaca.calendar import ET
from src.config import ConfigError, load_config

log = logging.getLogger("preflight")

OK, FAIL = "OK  ", "FAIL"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m cli.preflight")
    p.add_argument("--expect-account", default=None,
                   help="account number the keys must address; omit to skip the check")
    p.add_argument("--expect-equity", type=float, default=None,
                   help="equity the account should hold, within --equity-tolerance")
    p.add_argument("--equity-tolerance", type=float, default=1.0,
                   help="absolute dollars of slack on the equity check")
    p.add_argument("--require-level", type=int, default=None,
                   help="minimum options trading level (3 for debit spreads)")
    p.add_argument("--require-flat", action="store_true",
                   help="fail if any position is open")
    p.add_argument("--require-paper", action="store_true", default=True,
                   help="fail if the base URL is not a paper endpoint (default on)")
    p.add_argument("--out", type=Path, default=None, help="append a JSON line here")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    started = datetime.now(tz=timezone.utc)
    print(f"preflight at {started.astimezone(ET):%Y-%m-%d %H:%M:%S %Z}")

    try:
        config = load_config()
        from src.brokers.alpaca.client import account_summary, build_clients
        from src.brokers.alpaca.positions import reconcile

        clients = build_clients(config)
        summary = account_summary(clients)
        book = reconcile(clients)
    except ConfigError as exc:
        print(f"config: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 -- credentials or network
        print(f"could not reach the broker: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    checks: list[tuple[str, bool, str]] = []

    number = str(summary.get("account_number") or "")
    if args.expect_account:
        # Compared, never logged in full. A preflight that printed the account
        # number into a file would put operator state somewhere it can be
        # committed by accident.
        matched = number == args.expect_account.strip()
        checks.append((
            "account matches the expected one", matched,
            f"...{number[-4:]}" if number else "no account number returned",
        ))
    else:
        checks.append(("account number read back", bool(number), f"...{number[-4:]}"))

    equity = float(summary.get("equity") or 0.0)
    if args.expect_equity is not None:
        within = abs(equity - args.expect_equity) <= args.equity_tolerance
        checks.append((
            f"equity is {args.expect_equity:,.2f} (+/-{args.equity_tolerance:,.2f})",
            within, f"{equity:,.2f}",
        ))
    else:
        checks.append(("equity is positive", equity > 0, f"{equity:,.2f}"))

    level = summary.get("options_trading_level")
    if args.require_level is not None:
        ok = isinstance(level, int) and level >= args.require_level
        checks.append((f"options level >= {args.require_level}", ok, str(level)))

    checks.append((
        "trading is not blocked", not summary.get("trading_blocked"),
        str(summary.get("trading_blocked")),
    ))

    if args.require_paper:
        base = config.env.alpaca_base_url or ""
        checks.append(("base URL is a paper endpoint", "paper-api" in base, base))

    if args.require_flat:
        checks.append((
            "no open positions", len(book) == 0,
            f"{len(book)} position(s): {', '.join(book.symbols) or 'none'}",
        ))
    else:
        checks.append(("positions read", True, f"{len(book)} open"))

    checks.append((
        "mock mode is off", not clients.mock, str(clients.mock),
    ))

    print()
    width = max(len(name) for name, _, _ in checks)
    for name, passed, detail in checks:
        print(f"  [{OK if passed else FAIL}] {name:<{width}}  {detail}")

    failed = [name for name, passed, _ in checks if not passed]
    print()
    if failed:
        print(f"PREFLIGHT FAILED: {len(failed)} check(s) -- {'; '.join(failed)}")
    else:
        print("PREFLIGHT OK")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "at": started.isoformat(),
                "at_et": started.astimezone(ET).isoformat(),
                # Last four only. The full number is operator state.
                "account_suffix": number[-4:],
                "equity": equity,
                "options_level": level,
                "open_positions": len(book),
                "passed": not failed,
                "failed_checks": failed,
            }) + "\n")

    return 3 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
