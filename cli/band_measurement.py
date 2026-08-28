"""Survivor counts per symbol, with the metrics acceptance bands off and on.

    python -m cli.band_measurement
    python -m cli.band_measurement --symbols SPY,QQQ,NVDA --type call

The six ``metrics.*`` bands were declared in config and read by nothing until
they were wired into the prefilter as rejection gates. Turning a gate on that
has never run is exactly the change that can empty a survivor set silently, so
this reports the same live chain scored both ways and names which band did the
rejecting.

Read-only. It fetches contracts and snapshots and places no orders.
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
from datetime import date, timedelta
from typing import Any

from src.brokers.alpaca.cache import MarketDataCache
from src.brokers.alpaca.calendar import TradingCalendar
from src.brokers.alpaca.client import build_clients
from src.brokers.alpaca.contracts import fetch as fetch_contracts
from src.brokers.alpaca.quotes import fetch_snapshots
from src.config import ConfigError, load_config
from src.options.metrics import realized_volatility
from src.options.prefilter import assemble, plan_scan
from src.signals.indicators import atr as atr_indicator

log = logging.getLogger("bands")

BAND_REASONS = (
    "theta too high",
    "gamma too low",
    "iv rich",
    "spread cost vs premium",
    "breakeven too far",
    "modeled pnl too low",
)


class _Override:
    """A limits view with one key overridden.

    Used to score the identical chain twice without editing config. Wrapping
    rather than mutating matters: the second scoring must differ from the first
    in exactly one value, and a config edit between two live fetches would also
    change the chain.
    """

    def __init__(self, inner: Any, key: str, value: Any) -> None:
        self._inner = inner
        self._key = key
        self._value = value

    def get_bool(self, key: str):
        return self._value if key == self._key else self._inner.get_bool(key)

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


def underlying_bars(clients, symbol: str, days: int = 120):
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
        raise RuntimeError(f"{symbol}: no daily bars returned")
    if "symbol" in frame.index.names:
        frame = frame.xs(symbol, level="symbol")
    return frame


def scan(clients, calendar, symbol: str, option_type: str, cache) -> dict[str, Any]:
    limits = clients.config.limits
    bars = underlying_bars(clients, symbol)
    spot = float(bars["close"].iloc[-1])
    atr = float(atr_indicator(bars, limits.get_int("signals.atr_period")).iloc[-1])
    rv = realized_volatility(
        list(bars["close"]), limits.get_int("metrics.realized_vol_window_days")
    )
    session = calendar.order_session()
    plan = plan_scan(limits, calendar, session, symbol, spot)

    specs = fetch_contracts(
        clients, symbol,
        expiry_gte=plan.expiry_gte, expiry_lte=plan.expiry_lte,
        option_type=option_type,
        strike_gte=plan.strike_gte, strike_lte=plan.strike_lte,
        cache=cache,
    )
    quotes = fetch_snapshots(clients, [s.symbol for s in specs], cache=cache)

    # The same chain, scored twice. Only the flag differs.
    off = assemble(symbol, specs, quotes, calendar, session, spot, atr, rv,
                   _Override(limits, "prefilter.apply_metric_bands", False), plan)
    on = assemble(symbol, specs, quotes, calendar, session, spot, atr, rv,
                  _Override(limits, "prefilter.apply_metric_bands", True), plan)

    band_counts = {r: n for r, n in on.reason_counts.items() if r in BAND_REASONS}
    sole = {r: n for r, n in on.sole_reason.items() if r in BAND_REASONS}

    # What the bands actually removed, and how each of them scored.
    lost = [c for c in off.survivors if c.symbol not in {s.symbol for s in on.survivors}]
    detail = []
    for c in lost[:6]:
        m = c.metrics
        detail.append({
            "symbol": c.symbol,
            "failed": [f for f in _refail(c, on) if f in BAND_REASONS],
            "theta_pct_per_day": round(m.theta_pct_per_day, 4),
            "gamma_per_1pct": round(m.gamma_per_1pct, 5),
            "iv_vs_rv": round(m.iv_vs_rv, 3),
            "spread_pct_of_premium": round(m.spread_pct_of_premium, 4),
            "breakeven_move_pct": round(m.breakeven_move_pct, 4),
            "pnl_to_spread_ratio": round(m.pnl_to_spread_ratio, 2),
        })

    return {
        "symbol": symbol,
        "spot": round(spot, 2),
        "atr": round(atr, 3),
        "realized_vol": round(rv, 4),
        "expiry": plan.target_expiry.isoformat(),
        "session_dte": plan.target_session_dte,
        "contracts_scanned": len(specs),
        "survivors_bands_off": len(off.survivors),
        "survivors_bands_on": len(on.survivors),
        "removed_by_bands": len(off.survivors) - len(on.survivors),
        "band_reason_counts": band_counts,
        "band_sole_reason": sole,
        "examples_removed": detail,
    }


def _refail(candidate, result) -> tuple[str, ...]:
    for c in result.candidates:
        if c.symbol == candidate.symbol:
            return c.failures
    return ()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cli.band_measurement")
    parser.add_argument("--symbols", default=None, help="comma-separated; default universe")
    parser.add_argument("--type", dest="option_type", default="call", choices=("call", "put"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    try:
        config = load_config()
    except ConfigError as exc:
        raise SystemExit(f"config: {exc}")

    clients = build_clients(config)
    cache = MarketDataCache.from_config(config)
    today = date.today()
    calendar = TradingCalendar.fetch(clients, today - timedelta(days=10),
                                     today + timedelta(days=120))

    symbols = (
        [s.strip().upper() for s in args.symbols.split(",")]
        if args.symbols else list(config.universe.symbols)
    )

    rows, failures = [], []
    for symbol in symbols:
        try:
            rows.append(scan(clients, calendar, symbol, args.option_type, cache))
        except Exception as exc:  # noqa: BLE001 -- a per-symbol failure is data
            failures.append({"symbol": symbol, "error": f"{type(exc).__name__}: {exc}"})
            log.warning("%s: %s", symbol, exc)

    _print(rows, failures)
    payload = {"rows": rows, "failures": failures, "option_type": args.option_type}
    if args.out:
        from pathlib import Path

        Path(args.out).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return 0


def _print(rows: list[dict], failures: list[dict]) -> None:
    print(f"\n{'symbol':<8}{'spot':>9}{'dte':>5}{'scanned':>9}"
          f"{'off':>6}{'on':>5}{'lost':>6}   binding bands (sole reason)")
    print("-" * 88)
    totals = collections.Counter()
    for r in rows:
        sole = ", ".join(f"{k} x{v}" for k, v in r["band_sole_reason"].items()) or "-"
        print(f"{r['symbol']:<8}{r['spot']:>9.2f}{r['session_dte']:>5}"
              f"{r['contracts_scanned']:>9}{r['survivors_bands_off']:>6}"
              f"{r['survivors_bands_on']:>5}{r['removed_by_bands']:>6}   {sole}")
        totals["off"] += r["survivors_bands_off"]
        totals["on"] += r["survivors_bands_on"]
        for k, v in r["band_reason_counts"].items():
            totals[k] += v

    print("-" * 88)
    print(f"{'TOTAL':<8}{'':>9}{'':>5}{'':>9}{totals['off']:>6}{totals['on']:>5}"
          f"{totals['off'] - totals['on']:>6}")
    emptied = [r["symbol"] for r in rows
               if r["survivors_bands_off"] > 0 and r["survivors_bands_on"] == 0]
    if emptied:
        print(f"\nEMPTIED BY THE BANDS: {', '.join(emptied)}")
    print("\nBand rejections across all scanned contracts (not just survivors):")
    for reason in BAND_REASONS:
        print(f"  {reason:<26} {totals[reason]}")
    for f in failures:
        print(f"  FAILED {f['symbol']}: {f['error']}")


if __name__ == "__main__":
    raise SystemExit(main())
