"""Monday's measurement: DTE band x strike window, decided by data not opinion.

Two open questions block Step 7, and neither can be answered from Friday-close
data. Spreads at the close are not spreads at 10:30, and greeks coverage on a
weekend feed is not coverage during a session.

**Question 1 -- DTE band.** Longer DTE buys lower theta and costs wider
spreads, more vega, thinner liquidity, and a higher chance of spanning an
earnings print. The number that decides it is total friction per unit of
delta, not theta alone.

**Question 2 -- strike window.** The +/-10% window left only 8 survivors on SPY
and 5 on NVDA after the delta band moved to 0.55-0.75. That is too thin for
Agent 4 to make a real choice. Sweeping the window says whether widening it
recovers candidates or merely buys back the missing-greeks problem that
narrowing solved.

Run it read-only, during the session:

    python -m cli.monday_measurement                    # SPY NVDA AMD
    python -m cli.monday_measurement --symbols SPY --json out.json

**The capital constraint is likely to be binding**, so the number of contracts
clearing the caps is reported per bucket. A bucket where the answer is one or
zero is not viable regardless of its theta profile.

Places no orders. Reads only.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame

from src.brokers.alpaca.cache import MarketDataCache
from src.brokers.alpaca.calendar import TradingCalendar, now_et
from src.brokers.alpaca.client import build_clients, sizing_capital, with_retry
from src.brokers.alpaca.contracts import fetch as fetch_contracts
from src.brokers.alpaca.quotes import fetch_snapshots
from src.config import Section, load_config
from src.earnings.calendar import EarningsCalendar, EarningsError, spans_earnings
from src.options.metrics import CONTRACT_MULTIPLIER, MetricError, compute_metrics, modeled_hold_hours, realized_volatility
from src.options.prefilter import evaluate_candidates
from src.risk.sizing import AccountState, SizingLimits, compute_size
from src.signals.indicators import atr as atr_indicator

log = logging.getLogger("measure")

DEFAULT_SYMBOLS = ("SPY", "NVDA", "AMD")

# The three buckets from the strategy revision, in calendar days.
DTE_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("30-45", 30, 45),
    ("60-90", 60, 90),
    ("120+", 120, 180),
)

STRIKE_WINDOWS: tuple[float, ...] = (0.10, 0.15, 0.20)



@dataclass
class Cell:
    """One (symbol, DTE bucket, strike window) measurement."""

    symbol: str
    bucket: str
    window_pct: float
    spot: float
    atr: float
    realized_vol: float

    contracts: int = 0
    with_greeks: int = 0
    greeks_coverage: float = 0.0
    survivors: int = 0

    theta_pct_3_sessions: float | None = None
    round_trip_spread_vs_move: float | None = None
    friction_per_unit_delta: float | None = None
    vega_pct_of_premium: float | None = None
    median_premium_per_contract: float | None = None
    contracts_clearing_caps: int | None = None
    monthly_survivors: int = 0
    weekly_survivors: int = 0
    oi_min: int | None = None
    oi_median: float | None = None
    oi_max: int | None = None
    oi_clearing_500: int | None = None
    spread_pct_median: float | None = None
    spread_pct_max: float | None = None
    pct_spanning_earnings: float | None = None
    earnings_date: str | None = None
    earnings_known: bool = False

    note: str = ""

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def limits_with(limits: Section, **overrides: Any) -> Section:
    """Clone the limits Section with dotted-key overrides.

    Lets the sweep vary dte and strike bounds without mutating config or
    duplicating the filter logic.
    """
    data = limits.as_dict()
    for dotted, value in overrides.items():
        node = data
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return Section(data, limits.source)


def measure_symbol(
    clients: Any,
    symbol: str,
    calendar: TradingCalendar,
    order_session: date,
    cache: MarketDataCache,
    earnings: EarningsCalendar | None,
    sizing_limits: SizingLimits,
    account: AccountState,
    monthly_only: bool = False,
) -> list[Cell]:
    config = clients.config
    limits = config.limits

    spot = float(
        with_retry(config, f"spot({symbol})", lambda: clients.stocks.get_stock_latest_trade(
            StockLatestTradeRequest(symbol_or_symbols=symbol, feed=clients.equities_feed)
        ))[symbol].price
    )
    frame = with_retry(config, f"bars({symbol})", lambda: clients.stocks.get_stock_bars(
        StockBarsRequest(
            symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
            start=date.today() - timedelta(days=120), feed=clients.equities_feed,
        )
    )).df
    daily = frame.xs(symbol) if hasattr(frame.index, "levels") else frame
    daily = daily.rename(columns=str.lower)

    realized = realized_volatility(
        list(daily["close"]), window=limits.get_int("metrics.realized_vol_window_days")
    )
    atr = float(atr_indicator(daily, limits.get_int("signals.atr_period")).iloc[-1])

    entry = earnings.get(symbol) if earnings else None
    earnings_date = entry.as_date if entry else None

    hold_hours = modeled_hold_hours(limits)
    hold_sessions = limits.get_int("metrics.modeled_hold_sessions")
    theta_day_hours = limits.get_float("metrics.theta_day_hours")

    cells: list[Cell] = []
    for bucket, lo_days, hi_days in DTE_BUCKETS:
        for window in STRIKE_WINDOWS:
            cell = Cell(symbol=symbol, bucket=bucket, window_pct=window,
                        spot=spot, atr=atr, realized_vol=realized,
                        earnings_date=earnings_date.isoformat() if earnings_date else None,
                        earnings_known=earnings_date is not None)

            gte = date.today() + timedelta(days=lo_days)
            lte = date.today() + timedelta(days=hi_days)
            strike_gte = round(spot * (1 - window), 2)
            strike_lte = round(spot * (1 + window), 2)

            try:
                specs = fetch_contracts(
                    clients, symbol, expiry_gte=gte, expiry_lte=lte,
                    option_type="call", strike_gte=strike_gte, strike_lte=strike_lte,
                    cache=cache,
                )
            except Exception as exc:  # noqa: BLE001
                cell.note = f"contract fetch failed: {exc}"
                cells.append(cell)
                continue

            if not specs:
                cell.note = "no contracts in bucket"
                cells.append(cell)
                continue

            quotes = fetch_snapshots(clients, [s.symbol for s in specs], cache=cache)
            cell.contracts = len(specs)
            cell.with_greeks = sum(1 for s in specs if quotes[s.symbol].has_greeks)
            cell.greeks_coverage = round(cell.with_greeks / len(specs), 4)

            # Session-DTE gate must not fire inside the sweep: the bucket IS
            # the DTE question. Widen it and let the bucket bounds do the work.
            swept = limits_with(
                limits,
                **{"prefilter.dte_min": 0, "prefilter.dte_max": 10_000,
                   "prefilter.strike_window_pct": window,
                   "prefilter.require_monthly_expiry": monthly_only},
            )
            candidates = evaluate_candidates(
                specs, quotes, calendar, order_session, spot, atr, realized, swept
            )
            survivors = [c for c in candidates if c.survived]
            cell.survivors = len(survivors)

            spanning = [spans_earnings(s.expiry, earnings_date) for s in specs]
            known = [v for v in spanning if v is not None]
            cell.pct_spanning_earnings = (
                round(sum(known) / len(known), 4) if known else None
            )

            if not survivors:
                cell.note = "no survivors"
                cells.append(cell)
                continue

            thetas, frictions, vegas, premiums, spread_moves, clearing = [], [], [], [], [], 0
            for candidate in survivors:
                quote = candidate.quote
                metrics = candidate.metrics
                if metrics is None:
                    continue
                premium = metrics.premium
                thetas.append(metrics.theta_pct_per_day * hold_sessions)

                # Round-trip friction against the move one ATR actually buys.
                expected_move = abs(metrics.delta) * atr
                round_trip = 2.0 * metrics.spread
                spread_moves.append(round_trip / expected_move if expected_move > 0 else float("inf"))
                frictions.append(
                    (round_trip + abs(metrics.theta) * hold_sessions)
                    / abs(metrics.delta)
                )
                if quote.vega is not None and premium > 0:
                    vegas.append(quote.vega / premium)
                premiums.append(metrics.cost_per_contract)

                sized = compute_size(
                    cost_per_contract=metrics.cost_per_contract,
                    max_risk_per_contract=metrics.max_risk,
                    account=account, limits=sizing_limits,
                )
                if sized.final_contracts > 0:
                    clearing += 1

            cell.theta_pct_3_sessions = _median(thetas)
            cell.round_trip_spread_vs_move = _median(spread_moves)
            cell.friction_per_unit_delta = _median(frictions)
            cell.vega_pct_of_premium = _median(vegas)
            cell.median_premium_per_contract = _median(premiums)
            cell.contracts_clearing_caps = clearing
            cell.monthly_survivors = sum(1 for c in survivors if c.is_monthly)
            cell.weekly_survivors = len(survivors) - cell.monthly_survivors
            ois = [c.spec.open_interest for c in survivors]
            if ois:
                cell.oi_min, cell.oi_max = min(ois), max(ois)
                cell.oi_median = _median([float(o) for o in ois])
                cell.oi_clearing_500 = sum(1 for o in ois if o >= 500)
            spreads = [c.quote.spread_pct_of_mid for c in survivors
                       if c.quote.spread_pct_of_mid is not None]
            if spreads:
                cell.spread_pct_median = _median(spreads)
                cell.spread_pct_max = max(spreads)
            cells.append(cell)

    return cells


def print_report(cells: list[Cell], no_earnings: frozenset[str] = frozenset()) -> None:
    def fmt(value: Any, spec: str = ".2f") -> str:
        if value is None:
            return "    -"
        if isinstance(value, float):
            return format(value, spec)
        return str(value)

    print("\n=== STRIKE WINDOW SWEEP: survivors and greeks coverage ===")
    for symbol in dict.fromkeys(c.symbol for c in cells):
        print(f"\n  {symbol}")
        print(f"    {'bucket':<8} {'window':>7} {'contracts':>10} {'greeks':>8} "
              f"{'survivors':>10} {'clearing caps':>14}")
        for cell in [c for c in cells if c.symbol == symbol]:
            print(f"    {cell.bucket:<8} {cell.window_pct:>6.0%} {cell.contracts:>10} "
                  f"{cell.greeks_coverage:>7.0%} {cell.survivors:>10} "
                  f"{fmt(cell.contracts_clearing_caps):>14}"
                  + (f"   [{cell.note}]" if cell.note else ""))

    print("\n=== EXPIRY TYPE, OPEN INTEREST AND SPREAD (survivors) ===")
    for symbol in dict.fromkeys(c.symbol for c in cells):
        print(f"\n  {symbol}")
        print(f"    {'bucket':<9}{'window':>7}{'surv':>6}{'mthly':>7}{'wkly':>6}"
              f"{'OI min':>8}{'OI med':>8}{'OI max':>8}{'OI>=500':>9}"
              f"{'spr med':>9}{'spr max':>9}")
        for cell in (c for c in cells if c.symbol == symbol):
            print(f"    {cell.bucket:<9}{cell.window_pct:>6.0%} {cell.survivors:>6}"
                  f"{cell.monthly_survivors:>7}{cell.weekly_survivors:>6}"
                  f"{fmt(cell.oi_min, '.0f'):>8}{fmt(cell.oi_median, '.0f'):>8}"
                  f"{fmt(cell.oi_max, '.0f'):>8}{fmt(cell.oi_clearing_500, '.0f'):>9}"
                  f"{fmt(cell.spread_pct_median, '.1%'):>9}"
                  f"{fmt(cell.spread_pct_max, '.1%'):>9}")

    print("\n=== DTE BUCKET COMPARISON (median over survivors, 3-session hold) ===")
    for symbol in dict.fromkeys(c.symbol for c in cells):
        print(f"\n  {symbol}")
        print(f"    {'bucket':<8} {'window':>7} {'theta%x3':>9} {'rt spread':>10} "
              f"{'friction/d':>11} {'vega%prem':>10} {'premium $':>10} {'spans ER':>9}")
        for cell in [c for c in cells if c.symbol == symbol]:
            print(f"    {cell.bucket:<8} {cell.window_pct:>6.0%} "
                  f"{fmt(cell.theta_pct_3_sessions, '.1%'):>9} "
                  f"{fmt(cell.round_trip_spread_vs_move, '.1%'):>10} "
                  f"{fmt(cell.friction_per_unit_delta, '.3f'):>11} "
                  f"{fmt(cell.vega_pct_of_premium, '.1%'):>10} "
                  f"{fmt(cell.median_premium_per_contract, '.0f'):>10} "
                  f"{fmt(cell.pct_spanning_earnings, '.0%'):>9}")

    # A blank 'spans ER' means two entirely different things, and reporting
    # them as one reads a healthy universe as a broken one: a declared
    # print-free instrument has no date BY DEFINITION and trades normally,
    # while a genuinely unknown date fails closed and excludes.
    blank = {c.symbol for c in cells if not c.earnings_known}
    declared = sorted(blank & no_earnings)
    unknown = sorted(blank - no_earnings)
    if declared:
        print(f"\n  NOTE: {declared} are declared no-earnings instruments in "
              "universe.yaml. Blank 'spans ER' is correct for them and they are "
              "NOT excluded.")
    if unknown:
        print(f"\n  WARNING: earnings date unknown for {unknown} -- not declared "
              "print-free either. The 'spans ER' column is blank and those symbols "
              "WOULD BE EXCLUDED in live trading (fail closed). Run "
              "`python -m cli.earnings_check` before trusting this run.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Monday DTE and strike-window measurement")
    parser.add_argument("--symbols", nargs="*", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--json", type=Path, help="write the raw cells here")
    parser.add_argument("--skip-earnings", action="store_true",
                        help="run without the earnings column (no FMP key)")
    parser.add_argument("--monthly-only", action="store_true",
                        help="restrict survivors to standard monthly expiries")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    config = load_config()
    clients = build_clients(config)
    cache = MarketDataCache.from_config(config)
    symbols = [s.upper() for s in args.symbols]

    earnings: EarningsCalendar | None = None
    if not args.skip_earnings:
        try:
            earnings = EarningsCalendar.from_config(config)
            earnings.refresh(symbols)
            print(f"earnings refreshed for {len(symbols)} symbol(s)")
        except EarningsError as exc:
            print(f"earnings unavailable: {exc}\n"
                  "  -> continuing with the 'spans ER' column blank. In live trading "
                  "this state EXCLUDES every symbol.", file=sys.stderr)
            earnings = None

    account_obj = with_retry(config, "get_account", clients.trading.get_account)
    account = AccountState(
        equity=float(account_obj.equity),
        options_buying_power=sizing_capital(account_obj),
    )
    sizing_limits = SizingLimits.from_limits(config.limits)

    calendar = TradingCalendar.around(clients, date.today(), forward_days=240)
    order_session = calendar.order_session(now_et())

    print(f"session {order_session} | equity {account.equity:,.2f} | "
          f"options BP {account.options_buying_power:,.2f}")

    cells: list[Cell] = []
    for symbol in symbols:
        print(f"measuring {symbol} ...", flush=True)
        cells.extend(measure_symbol(
            clients, symbol, calendar, order_session, cache, earnings,
            sizing_limits, account, monthly_only=args.monthly_only,
        ))

    print_report(cells, frozenset(config.universe.no_earnings_symbols))

    if args.json:
        args.json.write_text(
            json.dumps({
                "session": order_session.isoformat(),
                "equity": account.equity,
                "options_buying_power": account.options_buying_power,
                "cells": [c.as_row() for c in cells],
            }, indent=2),
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
