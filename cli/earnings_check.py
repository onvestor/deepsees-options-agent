"""Earnings preflight -- refresh the calendar and assert the universe resolves.

Run before any session that can place an order. It refetches every symbol,
then refuses to return 0 unless each one resolves to either a real earnings
date or an explicit ``no_earnings`` declaration in ``config/universe.yaml``.

    python -m cli.earnings_check                 # refresh, assert, report
    python -m cli.earnings_check --no-refresh    # assert against the cache only
    python -m cli.earnings_check --json out.json

The assertion is the point. The exclusion's other three rules make a broken
feed *safe* -- an unknown date excludes, so nothing trades on it -- and that
safety is exactly what hides the break: a provider returning nothing for every
symbol looks identical to a quiet week. This is the one check that refuses to
read silence as an answer.

Exit codes: 0 resolved, 2 unresolved, 3 refresh failed, 4 config/credentials.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone

from src.brokers.alpaca.calendar import TradingCalendar
from src.brokers.alpaca.client import build_clients
from src.config import ConfigError, load_config
from src.earnings.calendar import (
    EarningsCalendar,
    EarningsError,
    assert_universe_resolves,
    evaluate_exclusion,
)

log = logging.getLogger("earnings_check")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-refresh", action="store_true",
                        help="assert against the cache without refetching")
    parser.add_argument("--json", metavar="PATH", help="write the report as JSON")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        config = load_config()
    except ConfigError as exc:
        log.error("%s", exc)
        return 4

    limits = config.limits
    symbols = list(config.universe.symbols)
    no_earnings = config.universe.no_earnings_symbols
    max_age = limits.get_float("earnings.max_cache_age_hours")
    max_hold = limits.get_int("earnings.max_hold_sessions")
    buffer = limits.get_int("earnings.buffer_sessions")
    require_confirmed = limits.get_bool("earnings.require_confirmed")
    post_print = limits.get_int("earnings.post_print_buffer_sessions")

    calendar = EarningsCalendar.from_config(config)
    now = datetime.now(tz=timezone.utc)

    if not args.no_refresh:
        # Only the symbols that can actually have a print. Asking the provider
        # about an ETF earns a 402 on this tier and tells us nothing either way.
        wanted = [s for s in symbols if s not in no_earnings]
        try:
            calendar.refresh(wanted)
        except EarningsError as exc:
            log.error("refresh failed: %s", exc)
            return 3

    try:
        clients = build_clients(config)
        # back_days must clear a full quarter: the post-print buffer counts
        # sessions since the last report, and the last report is a quarter back.
        trading_calendar = TradingCalendar.around(clients, now.date(), 200, back_days=150)
    except Exception as exc:  # noqa: BLE001 -- reported, not swallowed
        log.error("could not build the trading calendar: %s", exc)
        return 4
    order_session = trading_calendar.order_session()

    rows = []
    for symbol in symbols:
        verdict = evaluate_exclusion(
            symbol=symbol,
            entry=calendar.get(symbol),
            order_session=order_session,
            trading_calendar=trading_calendar,
            now=now,
            max_hold_sessions=max_hold,
            buffer_sessions=buffer,
            max_cache_age_hours=max_age,
            require_confirmed=require_confirmed,
            no_earnings=symbol in no_earnings,
            post_print_buffer_sessions=post_print,
        )
        rows.append(verdict.to_dict())

    horizon = max_hold + buffer
    print(f"order session {order_session} | forward horizon {max_hold}+{buffer}={horizon} "
          f"sessions | post-print buffer {post_print} | require_confirmed={require_confirmed}")
    print(f"{'SYMBOL':<8}{'EXCLUDED':<10}{'NEXT':<13}{'PREV':<13}{'CONF':<7}"
          f"{'AHEAD':<7}{'SINCE':<7}REASON")
    for row in rows:
        confirmed = "-" if row["confirmed"] is None else str(row["confirmed"])
        ahead = "-" if row["sessions_until"] is None else row["sessions_until"]
        since = "-" if row["sessions_since_last"] is None else row["sessions_since_last"]
        entry = calendar.get(row["symbol"])
        previous = (entry.previous_date if entry else None) or "-"
        print(f"{row['symbol']:<8}{str(row['excluded']):<10}"
              f"{row['earnings_date'] or '-':<13}{previous:<13}{confirmed:<7}"
              f"{ahead:<7}{since:<7}{row['reason']}")

    status, code = "resolved", 0
    try:
        resolution = assert_universe_resolves(
            symbols, calendar, no_earnings, now, max_age, post_print
        )
    except EarningsError as exc:
        log.error("%s", exc)
        resolution, status, code = {}, "unresolved", 2

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({
                "generated_at": now.isoformat(),
                "order_session": order_session.isoformat(),
                "horizon_sessions": horizon,
                "post_print_buffer_sessions": post_print,
                "status": status,
                "resolution": resolution,
                "verdicts": rows,
            }, handle, indent=2)
        print(f"wrote {args.json}")

    if code == 0:
        blocked = [r["symbol"] for r in rows if r["excluded"]]
        print(f"universe resolved: {len(resolution)}/{len(symbols)} symbol(s). "
              f"blocked this session: {', '.join(blocked) or 'none'}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
