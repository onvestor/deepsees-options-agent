"""Step 1 -- broker round trip. The milestone that de-risks everything downstream.

Connect, fetch a chain for one symbol, pick one contract by a deterministic
rule, buy to open, read the position back, close it.

No agents. No indicators. No signal engine. The point is to find out what
Alpaca actually does before any of that is built on top of assumptions.

    python -m cli.step1_roundtrip --symbol SPY
    python -m cli.step1_roundtrip --symbol SPY --dry-run   # no order placed

``--dry-run`` performs every read and prints the contract it *would* buy.
Without it, this places real paper orders -- unless ``ALPACA_MOCK`` is set,
which makes every write raise.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable

from alpaca.data.requests import OptionSnapshotRequest, StockLatestTradeRequest
from alpaca.trading.enums import (
    ContractType,
    OrderSide,
    OrderStatus,
    PositionIntent,
    TimeInForce,
)
from alpaca.trading.requests import GetOptionContractsRequest, LimitOrderRequest

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
    sizing_capital,
    with_retry,
)
from src.config import Config, ConfigError, load_config
from src.options.occ import parse as parse_occ

log = logging.getLogger("step1")

# One contract controls 100 shares. Not a tunable -- it is the contract spec.
SHARES_PER_CONTRACT = 100


# ---------------------------------------------------------------------------
# Chain discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One contract with live quote data and every filter verdict recorded."""

    symbol: str
    strike: float
    expiry: date
    session_dte: int
    open_interest: int
    bid: float
    ask: float
    delta: float | None
    failures: tuple[str, ...] = field(default=())

    @property
    def survived(self) -> bool:
        return not self.failures

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def spread_pct_of_mid(self) -> float:
        return self.spread / self.mid if self.mid > 0 else float("inf")


def underlying_price(clients: AlpacaClients, symbol: str) -> float:
    """Latest trade price. Basic plan means IEX only -- a partial view of the
    tape, which is fine for picking a strike and is disclosed in the write-up."""
    request = StockLatestTradeRequest(symbol_or_symbols=symbol, feed=clients.equities_feed)
    trades = with_retry(
        clients.config, f"latest_trade({symbol})",
        lambda: clients.stocks.get_stock_latest_trade(request),
    )
    return float(trades[symbol].price)


def fetch_contracts(
    clients: AlpacaClients,
    symbol: str,
    spot: float,
    calendar: TradingCalendar,
    order_session: date,
) -> list[Any]:
    """Paginated contract discovery, bounded by *trading sessions*.

    Three things CLAUDE.md warns about, handled explicitly: the default
    ``expiration_date_lte`` is next weekend, so both date bounds are always
    passed; results are paginated via ``page_token``; and the bounds are
    derived from the session an order would belong to, not from today. A scan
    run on Saturday must not treat Monday's expiry as 2 days out -- it is zero
    sessions out, and unbuyable.
    """
    cfg = clients.config
    dte_min = cfg.limits.get_int("prefilter.dte_min")
    dte_max = cfg.limits.get_int("prefilter.dte_max")
    window = cfg.limits.get_float("roundtrip.strike_window_pct")
    option_type = cfg.limits.get_str("roundtrip.option_type")
    page_limit = cfg.limits.get_int("broker.contracts_page_limit")

    gte = calendar.session_offset(order_session, dte_min)
    lte = calendar.session_offset(order_session, dte_max)
    log.info(
        "order session %s -> expiry window %s..%s (%d-%d trading sessions)",
        order_session, gte, lte, dte_min, dte_max,
    )

    collected: list[Any] = []
    page_token: str | None = None
    while True:
        request = GetOptionContractsRequest(
            underlying_symbols=[symbol],
            status="active",
            expiration_date_gte=gte,
            expiration_date_lte=lte,
            strike_price_gte=str(round(spot * (1 - window), 2)),
            strike_price_lte=str(round(spot * (1 + window), 2)),
            type=ContractType(option_type),
            limit=page_limit,
            page_token=page_token,
        )
        page = with_retry(
            cfg, f"get_option_contracts({symbol})",
            lambda r=request: clients.trading.get_option_contracts(r),
        )
        collected.extend(page.option_contracts or [])
        page_token = getattr(page, "next_page_token", None)
        if not page_token:
            break

    log.info("contracts: %d, strikes within %.0f%% of %.2f", len(collected), window * 100, spot)
    return collected


def fetch_snapshots(clients: AlpacaClients, symbols: list[str]) -> dict[str, Any]:
    """Quotes and greeks. Batched -- the contract list can be a few hundred wide."""
    cfg = clients.config
    out: dict[str, Any] = {}
    batch_size = cfg.limits.get_int("broker.contracts_page_limit")
    for start in range(0, len(symbols), batch_size):
        batch = symbols[start : start + batch_size]
        request = OptionSnapshotRequest(symbol_or_symbols=batch, feed=clients.options_feed)
        page = with_retry(
            cfg, f"option_snapshots({len(batch)})",
            lambda r=request: clients.options.get_option_snapshot(r),
        )
        out.update(page)
    return out


def build_candidates(
    clients: AlpacaClients,
    contracts: Iterable[Any],
    calendar: TradingCalendar,
    order_session: date,
) -> list[Candidate]:
    """Evaluate **every** filter against **every** contract, recording all failures.

    First-match attribution lies. On Saturday's SPY scan it reported
    ``no delta = 0`` while 294 contracts were in fact missing greeks -- they
    had already been consumed by an earlier filter, so the counter never
    fired, and the breakdown looked like greeks were universally available.

    A contract may therefore appear under several reasons, and the reason
    counts sum to more than the number rejected. That is the point: the
    breakdown answers "how many contracts fail this test", which is the
    question worth asking when tuning a threshold.
    """
    cfg = clients.config
    min_oi = cfg.limits.get_int("prefilter.min_open_interest")
    min_bid = cfg.limits.get_float("prefilter.min_bid")
    max_spread_pct = cfg.limits.get_float("prefilter.max_spread_pct_of_mid")
    max_spread_abs = cfg.limits.get_float("prefilter.max_spread_abs")
    delta_min = cfg.limits.get_float("prefilter.delta_min")
    delta_max = cfg.limits.get_float("prefilter.delta_max")
    dte_min = cfg.limits.get_int("prefilter.dte_min")
    dte_max = cfg.limits.get_int("prefilter.dte_max")

    by_symbol = {c.symbol: c for c in contracts}
    snapshots = fetch_snapshots(clients, list(by_symbol))

    out: list[Candidate] = []
    for symbol, contract in by_symbol.items():
        failures: list[str] = []
        snapshot = snapshots.get(symbol)
        quote = getattr(snapshot, "latest_quote", None) if snapshot else None
        greeks = getattr(snapshot, "greeks", None) if snapshot else None

        bid = float(getattr(quote, "bid_price", 0) or 0.0)
        ask = float(getattr(quote, "ask_price", 0) or 0.0)
        delta = None
        if greeks is not None and getattr(greeks, "delta", None) is not None:
            delta = float(greeks.delta)

        # --- every test runs; none short-circuits ---
        if quote is None:
            failures.append("no quote")
        if bid <= 0 or ask <= 0 or ask < bid:
            failures.append("crossed/empty quote")
        if bid < min_bid:
            failures.append("bid below floor")
        if int(contract.open_interest or 0) < min_oi:
            failures.append("open interest")

        mid = (bid + ask) / 2.0
        if mid <= 0 or (ask - bid) > max_spread_abs or (ask - bid) / mid > max_spread_pct:
            failures.append("spread")

        if greeks is None:
            failures.append("no greeks")
        if delta is None:
            failures.append("no delta")
        elif not (delta_min <= abs(delta) <= delta_max):
            failures.append("delta band")

        session_dte = calendar.sessions_until(contract.expiration_date, order_session)
        if session_dte < dte_min or session_dte > dte_max:
            failures.append(f"session dte ({session_dte})" if session_dte >= 0 else "expired")

        # Cheap integrity check: our parse must agree with Alpaca's own fields.
        try:
            parsed = parse_occ(symbol)
            if parsed.strike != float(contract.strike_price) or parsed.expiry != contract.expiration_date:
                failures.append("occ mismatch")
        except Exception:  # noqa: BLE001
            failures.append("occ unparseable")

        out.append(
            Candidate(
                symbol=symbol,
                strike=float(contract.strike_price),
                expiry=contract.expiration_date,
                session_dte=session_dte,
                open_interest=int(contract.open_interest or 0),
                bid=bid,
                ask=ask,
                delta=delta,
                failures=tuple(failures),
            )
        )
    return out


def rejection_report(candidates: list[Candidate]) -> dict[str, Any]:
    """Multi-label breakdown. Reason counts intentionally sum above the total."""
    counts: dict[str, int] = {}
    sole: dict[str, int] = {}
    for candidate in candidates:
        for reason in candidate.failures:
            counts[reason] = counts.get(reason, 0) + 1
        if len(candidate.failures) == 1:
            only = candidate.failures[0]
            sole[only] = sole.get(only, 0) + 1

    rejected = sum(1 for c in candidates if not c.survived)
    return {
        "total": len(candidates),
        "survivors": len(candidates) - rejected,
        "rejected": rejected,
        "counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "sole_reason": sole,
    }


def print_rejection_report(report: dict[str, Any]) -> None:
    print(f"\n--- liquidity filter ({report['total']} contracts) ---")
    print(f"  {'reason':<24} {'fails':>6} {'sole':>6}")
    for reason, count in report["counts"].items():
        print(f"  {reason:<24} {count:>6} {report['sole_reason'].get(reason, 0):>6}")
    print(f"  {'-' * 38}")
    print(f"  {'rejected':<24} {report['rejected']:>6}")
    print(f"  {'SURVIVORS':<24} {report['survivors']:>6}")
    print("  (a contract can fail several tests; 'fails' sums above 'rejected'.")
    print("   'sole' is how many were rejected by that reason alone.)")


def select_contract(config: Config, candidates: list[Candidate]) -> Candidate:
    """Nearest expiry, then closest to the target delta. Deterministic and total."""
    survivors = [c for c in candidates if c.survived]
    if not survivors:
        raise BrokerError("no contract survived the deterministic filter")
    target = config.limits.get_float("roundtrip.target_delta")
    return min(survivors, key=lambda c: (c.expiry, abs(abs(c.delta or 0.0) - target), c.symbol))


# ---------------------------------------------------------------------------
# Order round trip
# ---------------------------------------------------------------------------


def _round_penny(value: float) -> float:
    return round(value + 1e-9, 2)


def check_affordable(clients: AlpacaClients, candidate: Candidate, qty: int) -> float:
    """Premium must fit inside sizing capital -- never margin buying power."""
    account = with_retry(clients.config, "get_account", clients.trading.get_account)
    capital = sizing_capital(account)
    premium = candidate.ask * SHARES_PER_CONTRACT * qty
    if premium > capital:
        raise BrokerError(
            f"premium {premium:.2f} exceeds sizing capital {capital:.2f} "
            f"(options_buying_power/equity, not margin buying power)"
        )
    return premium


def wait_for_fill(clients: AlpacaClients, order_id: str) -> Any:
    """Poll until filled, terminal, or the configured timeout expires.

    On timeout the order is cancelled rather than left resting. A resting entry
    order that fills later, unattended, is exactly what the time-stop exists to
    prevent.
    """
    cfg = clients.config
    timeout = cfg.limits.get_int("roundtrip.fill_timeout_seconds")
    interval = cfg.limits.get_int("exits.poll_interval_seconds")
    terminal = {OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.EXPIRED, OrderStatus.REJECTED}

    deadline = time.monotonic() + timeout
    order = with_retry(cfg, "get_order", lambda: clients.trading.get_order_by_id(order_id))
    while order.status not in terminal and time.monotonic() < deadline:
        time.sleep(min(interval, max(1, int(deadline - time.monotonic()))))
        order = with_retry(cfg, "get_order", lambda: clients.trading.get_order_by_id(order_id))

    if order.status not in terminal:
        log.warning("order %s still %s after %ds -- cancelling", order_id, order.status, timeout)
        with_retry(cfg, "cancel_order", lambda: clients.trading.cancel_order_by_id(order_id))
        order = with_retry(cfg, "get_order", lambda: clients.trading.get_order_by_id(order_id))
    return order


def submit(clients: AlpacaClients, request: LimitOrderRequest, description: str) -> Any:
    clients.assert_writable(description)
    order = with_retry(clients.config, description, lambda: clients.trading.submit_order(request))
    log.info("%s submitted: id=%s %s %s @ %s", description, order.id, order.side,
             order.symbol, order.limit_price)
    return wait_for_fill(clients, str(order.id))


def buy_to_open(
    clients: AlpacaClients,
    candidate: Candidate,
    calendar: TradingCalendar,
) -> Any:
    """Marketable limit, never a bare market order.

    The session-DTE floor is re-checked *here*, against the session this order
    will actually belong to, computed now rather than at scan time. That is the
    check that stops a contract selected hours earlier from being bought at
    0DTE.
    """
    cfg = clients.config
    qty = cfg.limits.get_int("roundtrip.qty")
    pad = cfg.limits.get_float("roundtrip.entry_limit_pad_pct")

    order_session = calendar.order_session(now_et())
    dte = assert_dte_within_band(
        calendar,
        candidate.expiry,
        order_session,
        cfg.limits.get_int("prefilter.dte_min"),
        cfg.limits.get_int("prefilter.dte_max"),
        candidate.symbol,
    )
    log.info("entry check: %s is %d session(s) from the %s session", candidate.symbol, dte, order_session)
    check_affordable(clients, candidate, qty)

    request = LimitOrderRequest(
        symbol=candidate.symbol,
        qty=qty,
        side=OrderSide.BUY,
        type="limit",
        time_in_force=TimeInForce.DAY,
        limit_price=_round_penny(candidate.ask * (1 + pad)),
        position_intent=PositionIntent.BUY_TO_OPEN,
    )
    return submit(clients, request, "buy_to_open")


def sell_to_close(clients: AlpacaClients, position: Any) -> Any:
    cfg = clients.config
    pad = cfg.limits.get_float("roundtrip.exit_limit_pad_pct")
    quotes = fetch_snapshots(clients, [position.symbol])
    quote = getattr(quotes.get(position.symbol), "latest_quote", None)
    bid = float(quote.bid_price) if quote and quote.bid_price else float(position.current_price or 0)
    request = LimitOrderRequest(
        symbol=position.symbol,
        qty=abs(int(float(position.qty))),
        side=OrderSide.SELL,
        type="limit",
        time_in_force=TimeInForce.DAY,
        limit_price=max(0.01, _round_penny(bid * (1 - pad))),
        position_intent=PositionIntent.SELL_TO_CLOSE,
    )
    return submit(clients, request, "sell_to_close")


def read_position(clients: AlpacaClients, symbol: str) -> Any | None:
    try:
        return with_retry(
            clients.config, "get_position", lambda: clients.trading.get_open_position(symbol)
        )
    except Exception as exc:  # noqa: BLE001 -- absence is a normal answer here
        log.info("no open position for %s (%s)", symbol, exc)
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


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

    spot = underlying_price(clients, symbol)
    contracts = fetch_contracts(clients, symbol, spot, calendar, order_session)
    candidates = build_candidates(clients, contracts, calendar, order_session)
    print_rejection_report(rejection_report(candidates))

    try:
        chosen = select_contract(config, candidates)
    except BrokerError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 8

    _report("selected contract", {
        "symbol": chosen.symbol,
        "underlying": f"{symbol} @ {spot:.2f}",
        "strike": chosen.strike,
        "expiry": f"{chosen.expiry}  ({chosen.session_dte} trading sessions out)",
        "bid/ask": f"{chosen.bid:.2f} / {chosen.ask:.2f}",
        "spread": f"{chosen.spread:.2f} ({chosen.spread_pct_of_mid:.1%} of mid)",
        "delta": chosen.delta,
        "open_interest": chosen.open_interest,
        "premium (1 contract)": f"{chosen.ask * SHARES_PER_CONTRACT:.2f}",
    })

    if args.dry_run:
        print("\ndry run -- no order placed")
        return 0

    if not clock.is_open:
        print("\nmarket is closed; an options day order will not fill. "
              "Re-run during regular hours, or use --dry-run.", file=sys.stderr)
        return 4

    try:
        entry = buy_to_open(clients, chosen, calendar)
    except DteError as exc:
        print(f"\nentry rejected: {exc}", file=sys.stderr)
        return 9

    _report("entry order", {
        "id": entry.id, "status": entry.status, "filled_qty": entry.filled_qty,
        "filled_avg_price": entry.filled_avg_price, "limit_price": entry.limit_price,
    })
    if entry.status != OrderStatus.FILLED:
        print(f"\nentry did not fill (status={entry.status}); nothing to close", file=sys.stderr)
        return 5

    position = read_position(clients, chosen.symbol)
    if position is None:
        print("\nfilled but no position read back -- investigate before Step 2", file=sys.stderr)
        return 6
    _report("position read back", {
        "symbol": position.symbol, "qty": position.qty,
        "avg_entry_price": position.avg_entry_price, "current_price": position.current_price,
        "market_value": position.market_value, "unrealized_pl": position.unrealized_pl,
    })

    exit_order = sell_to_close(clients, position)
    _report("exit order", {
        "id": exit_order.id, "status": exit_order.status,
        "filled_qty": exit_order.filled_qty, "filled_avg_price": exit_order.filled_avg_price,
    })

    remaining = read_position(clients, chosen.symbol)
    _report("round trip", {
        "contract": chosen.symbol,
        "entry_fill": entry.filled_avg_price,
        "exit_fill": exit_order.filled_avg_price,
        "position_closed": remaining is None,
    })
    return 0 if remaining is None else 7


if __name__ == "__main__":
    raise SystemExit(main())
