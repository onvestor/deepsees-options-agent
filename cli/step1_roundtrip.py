"""Step 1 -- broker round trip, now through the real order path.

Originally a self-contained spike: it carried its own chain fetch, its own
snapshot batching, its own candidate filter and its own order construction,
because none of those existed yet. All four exist now, so the duplicates are
gone and this drives the same modules a live session drives:

* ``src.brokers.alpaca.contracts`` and ``.quotes`` for the chain
* ``src.options.prefilter`` for the survivor set -- the real one, bands and all
* ``src.brokers.alpaca.orders`` for the passive mid entry and the stepped exit
* ``src.brokers.alpaca.positions`` for a reconciled read-back

    python -m cli.step1_roundtrip --symbol SPY --dry-run   # reads only
    python -m cli.step1_roundtrip --symbol SPY             # places real orders

**What this proves is different now.** The original proved the *broker* works.
This proves the *order builder* works -- that a contract chosen by the real
prefilter, priced by the real quote layer, sized as one contract, entered at a
passive mid and exited on the stepped ladder, completes a round trip and reads
back closed.

``--dry-run`` performs every read and places nothing. Without it this places
real paper orders, unless ``ALPACA_MOCK`` is set, which makes every write
raise.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from typing import Any

from src.brokers.alpaca.cache import MarketDataCache
from src.brokers.alpaca.calendar import (
    DteError,
    TradingCalendar,
    assert_dte_within_band,
    now_et,
)
from src.brokers.alpaca.client import (
    AlpacaClients,
    BrokerError,
    account_summary,
    build_clients,
    with_retry,
)
from src.brokers.alpaca.orders import (
    SHARES_PER_CONTRACT,
    ExecutionLimits,
    is_filled,
    place_entry,
    place_exit,
    step_ladder,
    urgency_for,
)
from src.brokers.alpaca.positions import read_position
from src.brokers.alpaca.quotes import fetch_snapshots
from src.config import Config, ConfigError, load_config
from src.options.metrics import realized_volatility
from src.options.prefilter import Candidate, PrefilterResult, run_prefilter
from src.signals.indicators import atr as atr_indicator

log = logging.getLogger("step1")

EXIT_REASON = "roundtrip"
"""Not one of the real exit reasons.

``urgency_for`` maps an unknown reason to HIGH, which is what this wants: the
round trip is closing a position it opened seconds ago purely to prove the
path, and there is nothing to be patient for.
"""


def underlying_bars(clients: AlpacaClients, symbol: str, days: int = 120) -> Any:
    """Daily bars for spot, ATR and realised vol -- the prefilter's inputs."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=date.today() - timedelta(days=days),
        feed=clients.equities_feed,
    )
    frame = clients.stocks.get_stock_bars(request).df
    if frame.empty:
        raise BrokerError(f"{symbol}: no daily bars returned")
    if "symbol" in frame.index.names:
        frame = frame.xs(symbol, level="symbol")
    return frame


def scan(clients: AlpacaClients, symbol: str, calendar: TradingCalendar,
         order_session: date, cache: MarketDataCache) -> PrefilterResult:
    """The real prefilter. No local reimplementation of any of it."""
    limits = clients.config.limits
    bars = underlying_bars(clients, symbol)
    spot = float(bars["close"].iloc[-1])
    atr = float(atr_indicator(bars, limits.get_int("signals.atr_period")).iloc[-1])
    rv = realized_volatility(
        list(bars["close"]), limits.get_int("metrics.realized_vol_window_days")
    )
    log.info("%s spot %.2f atr %.3f rv %.1f%%", symbol, spot, atr, rv * 100)

    return run_prefilter(
        clients, symbol, spot, atr, rv, calendar, order_session,
        option_type=limits.get_str("roundtrip.option_type"),
        cache=cache,
    )


def select(result: PrefilterResult) -> Candidate:
    """The top-ranked survivor.

    The prefilter already ranks by modelled P&L per unit of spread cost, which
    is the same ordering Agent 4 is offered. Taking the top one makes this
    round trip exercise the contract the live path would actually reach for --
    the old spike's own nearest-expiry-then-nearest-delta rule tested a
    selection nothing else used.
    """
    if not result.top:
        raise BrokerError(
            f"{result.symbol}: no contract survived the prefilter "
            f"({len(result.candidates)} scanned). Reasons: {result.reason_counts}"
        )
    return result.top[0]


def print_rejections(result: PrefilterResult) -> None:
    print(f"\n--- prefilter ({len(result.candidates)} contracts, "
          f"expiry {result.target_expiry}, {result.target_session_dte} sessions) ---")
    print(f"  {'reason':<26} {'fails':>6} {'sole':>6}")
    sole = result.sole_reason
    for reason, count in result.reason_counts.items():
        print(f"  {reason:<26} {count:>6} {sole.get(reason, 0):>6}")
    print(f"  {'-' * 40}")
    print(f"  {'SURVIVORS':<26} {len(result.survivors):>6}")


def _report(title: str, rows: dict[str, Any]) -> None:
    print(f"\n--- {title} ---")
    for key, value in rows.items():
        print(f"  {key:<38} {value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Step 1 broker round trip")
    parser.add_argument("--symbol", help="underlying; defaults to the first universe symbol")
    parser.add_argument("--dry-run", action="store_true", help="read-only; place no orders")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    try:
        config = load_config()
        clients = build_clients(config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    symbol = (args.symbol or config.universe.symbols[0]).upper()

    account = account_summary(clients)
    _report("account", account)
    if account["trading_blocked"]:
        print("trading is blocked on this account", file=sys.stderr)
        return 3

    clock = with_retry(config, "get_clock", clients.trading.get_clock)
    dte_max = config.limits.get_int("prefilter.dte_max")
    calendar = TradingCalendar.around(clients, date.today(), forward_days=dte_max * 3 + 21)
    order_session = calendar.order_session(now_et())
    _report("clock", {
        "is_open": clock.is_open,
        "next_open": clock.next_open,
        "order would belong to session": order_session,
        "mock mode (ALPACA_MOCK)": clients.mock,
    })

    cache = MarketDataCache.from_config(config)
    try:
        result = scan(clients, symbol, calendar, order_session, cache)
    except DteError as exc:
        print(f"\nscan refused: {exc}", file=sys.stderr)
        return 9
    print_rejections(result)

    try:
        chosen = select(result)
    except BrokerError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 8

    quote = chosen.quote
    metrics = chosen.metrics
    _report("selected contract", {
        "symbol": chosen.symbol,
        "underlying": f"{symbol} @ {result.spot:.2f}",
        "strike": chosen.spec.strike,
        "expiry": f"{chosen.spec.expiry}  ({chosen.session_dte} sessions out)",
        "bid/ask": f"{quote.bid:.2f} / {quote.ask:.2f}",
        "mid": f"{quote.mid:.2f}",
        "spread": f"{quote.spread:.2f} ({quote.spread_pct_of_mid:.2%} of mid)",
        "delta": f"{quote.delta:.4f}",
        "open_interest": chosen.spec.open_interest,
        "pnl_to_spread_ratio": f"{metrics.pnl_to_spread_ratio:.2f}",
        "premium (1 contract)": f"{quote.mid * SHARES_PER_CONTRACT:.2f}",
    })

    limits = ExecutionLimits.from_limits(config.limits)
    _report("planned orders", {
        "entry": f"passive mid limit @ {quote.mid:.2f}",
        "entry timeout": f"{limits.entry_fill_timeout_seconds}s",
        "exit reason": f"{EXIT_REASON} -> urgency {urgency_for(EXIT_REASON).value}",
        "exit ladder": step_ladder(quote, limits.exit_steps[urgency_for(EXIT_REASON)]),
    })

    if args.dry_run:
        print("\ndry run -- no order placed")
        return 0

    if not clock.is_open:
        print("\nmarket is closed; an options day order will not fill. "
              "Re-run during regular hours, or use --dry-run.", file=sys.stderr)
        return 4

    # Re-checked here, against the session this order will actually belong to.
    # A contract selected minutes ago must not be bought at 0DTE.
    try:
        assert_dte_within_band(
            calendar, chosen.spec.expiry, order_session,
            config.limits.get_int("prefilter.dte_min"),
            config.limits.get_int("prefilter.dte_max"),
            chosen.symbol,
        )
    except DteError as exc:
        print(f"\nentry rejected: {exc}", file=sys.stderr)
        return 9

    entry = place_entry(
        clients, symbol=chosen.symbol, qty=config.limits.get_int("roundtrip.qty"),
        quote=quote, limits=limits,
    )
    _report("entry order", {
        "id": entry.order_id, "status": entry.status,
        "limit": entry.limit_prices, "filled_qty": entry.filled,
        "filled_avg_price": entry.fill_price,
    })
    if not is_filled(entry.order):
        print(f"\nentry did not fill (status={entry.status}). A passive mid limit "
              "fills or it does not -- that is the design, not a failure.",
              file=sys.stderr)
        return 5

    position = read_position(clients, chosen.symbol)
    if position is None:
        print("\nfilled but no position read back -- investigate", file=sys.stderr)
        return 6
    _report("position read back (reconciled from Alpaca)", {
        "symbol": position.symbol, "underlying": position.underlying,
        "qty": position.qty, "avg_entry_price": position.avg_entry_price,
        "current_price": position.current_price,
        "market_value": position.market_value,
        "unrealized_pl": position.unrealized_pl,
        "pnl_pct_of_premium": (
            f"{position.pnl_pct():.2f}%" if position.pnl_pct() is not None else "unmarked"
        ),
    })

    fresh = fetch_snapshots(clients, [chosen.symbol])[chosen.symbol]
    exit_order = place_exit(
        clients, symbol=chosen.symbol, qty=abs(position.qty), quote=fresh,
        reason=EXIT_REASON, limits=limits,
        quote_reader=lambda: fetch_snapshots(clients, [chosen.symbol])[chosen.symbol],
    )
    _report("exit order", {
        "id": exit_order.order_id, "status": exit_order.status,
        "ladder used": exit_order.limit_prices, "steps": exit_order.attempts,
        "filled_qty": exit_order.filled, "filled_avg_price": exit_order.fill_price,
    })

    remaining = read_position(clients, chosen.symbol)
    entry_fill, exit_fill = entry.fill_price, exit_order.fill_price
    cost = None
    if entry_fill and exit_fill:
        cost = (entry_fill - exit_fill) / entry_fill * 100.0
    _report("round trip", {
        "contract": chosen.symbol,
        "entry_fill": entry_fill,
        "exit_fill": exit_fill,
        "realised round trip cost": f"{cost:.2f}% of premium" if cost is not None else "n/a",
        "quoted spread at entry": f"{quote.spread_pct_of_mid:.2%} of mid",
        "position_closed": remaining is None,
    })
    return 0 if remaining is None else 7


if __name__ == "__main__":
    raise SystemExit(main())
