"""Bar series for replay, and the one rule that makes replay worth running.

**No lookahead, enforced rather than intended.** Every read of a bar series
goes through :meth:`BarSeries.through`, which slices at a session and cannot
return a bar after it. A harness that accidentally evaluated indicators on the
full frame would produce results that look excellent and mean nothing, and the
mistake is invisible in the output -- it shows up as a strategy that works.
So the slice is the only accessor, the full frame is private, and there is a
test asserting the last bar of every window is the session being evaluated.

Bars are daily. The swing revision moved signal evaluation to daily bars with
hourly confirmation, and a daily frame is what an offline replay can honestly
reconstruct: the intraday tape on the Basic plan is IEX-only and partial, so
replaying it would model a fill environment that never existed.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.signals.indicators import REQUIRED_BAR_COLUMNS

log = logging.getLogger(__name__)


class BarError(RuntimeError):
    """Bar data that cannot be replayed, with the reason named."""


@dataclass(frozen=True)
class BarSeries:
    """One symbol's daily bars, readable only up to a session.

    ``frame`` is not part of the public surface by convention alone -- it is
    the thing every lookahead bug reaches for, so the accessors below are the
    ones the harness uses.
    """

    symbol: str
    frame: pd.DataFrame

    def __post_init__(self) -> None:
        missing = [c for c in REQUIRED_BAR_COLUMNS if c not in self.frame.columns]
        if missing:
            raise BarError(f"{self.symbol}: bars missing column(s) {missing}")
        if not isinstance(self.frame.index, pd.DatetimeIndex):
            raise BarError(f"{self.symbol}: bars must have a DatetimeIndex")
        if not self.frame.index.is_monotonic_increasing:
            raise BarError(f"{self.symbol}: bars must be sorted ascending by time")

    # -- the only ways to read ---------------------------------------------

    def through(self, session: date) -> pd.DataFrame:
        """Every bar up to and including ``session``. Never one bar further."""
        cutoff = pd.Timestamp(session)
        if self.frame.index.tz is not None:
            cutoff = cutoff.tz_localize(self.frame.index.tz)
        return self.frame.loc[self.frame.index <= cutoff]

    def close_on(self, session: date) -> float | None:
        """The close of ``session`` itself, or None if the symbol did not trade."""
        window = self.through(session)
        if window.empty:
            return None
        last = window.index[-1].date()
        if last != session:
            return None
        return float(window["close"].iloc[-1])

    def sessions(self) -> tuple[date, ...]:
        return tuple(ts.date() for ts in self.frame.index)

    def has(self, session: date) -> bool:
        return self.close_on(session) is not None

    def __len__(self) -> int:
        return len(self.frame)


@dataclass(frozen=True)
class BarSet:
    """Every symbol's bars for one replay."""

    series: dict[str, BarSeries]

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(sorted(self.series))

    def get(self, symbol: str) -> BarSeries:
        try:
            return self.series[symbol.upper()]
        except KeyError:
            raise BarError(
                f"no bars for {symbol!r}; loaded symbols are {list(self.symbols)}"
            ) from None

    def common_sessions(self) -> tuple[date, ...]:
        """Sessions where at least one symbol traded, ascending.

        Union rather than intersection: a symbol that was halted for a day
        should not stop the whole replay, and the per-symbol ``has`` check
        already skips it.
        """
        seen: set[date] = set()
        for series in self.series.values():
            seen.update(series.sessions())
        return tuple(sorted(seen))


# --- loading ---------------------------------------------------------------


def load_csv(path: Path, symbol: str | None = None) -> BarSeries:
    """Load one symbol's daily bars from CSV.

    Expects a date column (``date``, ``timestamp`` or the index) plus the five
    OHLCV columns. Column names are lowercased, so a file exported with
    ``Close`` loads without editing.
    """
    frame = pd.read_csv(path)
    frame.columns = [str(c).strip().lower() for c in frame.columns]

    stamp_column = next(
        (c for c in ("timestamp", "date", "time", "datetime") if c in frame.columns),
        None,
    )
    if stamp_column is None:
        raise BarError(f"{path}: no date column (looked for timestamp/date/time/datetime)")

    frame[stamp_column] = pd.to_datetime(frame[stamp_column], utc=False)
    frame = frame.set_index(stamp_column).sort_index()
    frame.index.name = None

    missing = [c for c in REQUIRED_BAR_COLUMNS if c not in frame.columns]
    if missing:
        raise BarError(f"{path}: missing column(s) {missing}")

    return BarSeries(symbol=(symbol or path.stem).upper(), frame=frame[list(REQUIRED_BAR_COLUMNS)])


def load_directory(directory: Path, symbols: Sequence[str] | None = None) -> BarSet:
    """Load ``<SYMBOL>.csv`` for each symbol in a directory.

    A missing file names the symbol and the path it looked for. Replay data is
    operator-supplied like everything else under a gitignored directory, and a
    bare FileNotFoundError from inside a session loop is not a useful error.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise BarError(f"bar directory {directory} does not exist")

    if symbols is None:
        paths = sorted(directory.glob("*.csv"))
        if not paths:
            raise BarError(f"no CSV files in {directory}")
    else:
        paths = []
        for symbol in symbols:
            path = directory / f"{symbol.upper()}.csv"
            if not path.is_file():
                raise BarError(f"no bars for {symbol}: expected {path}")
            paths.append(path)

    return BarSet({p.stem.upper(): load_csv(p) for p in paths})


# --- synthetic series, for tests and for a runnable harness ----------------


def synthetic_series(
    symbol: str,
    start: date,
    sessions: int,
    *,
    start_price: float = 500.0,
    drift_per_session: float = 0.0015,
    amplitude: float = 0.012,
    period: int = 14,
    daily_range_pct: float = 0.011,
    volume: int = 5_000_000,
) -> BarSeries:
    """A deterministic bar series: a drifting trend with a sine oscillation.

    Deterministic on purpose -- no RNG, seeded or otherwise. A replay whose
    input changes between runs cannot be used to compare two prompts, which is
    the main thing the harness exists for.

    Weekends are skipped so the index looks like a session series. Holidays are
    not modelled here; the trading calendar the harness is given is what
    decides which sessions actually run.
    """
    if sessions < 1:
        raise BarError(f"sessions must be >= 1, got {sessions}")

    stamps: list[pd.Timestamp] = []
    day = start
    while len(stamps) < sessions:
        if day.weekday() < 5:
            stamps.append(pd.Timestamp(day))
        day += timedelta(days=1)

    closes = []
    for i in range(sessions):
        trend = start_price * (1.0 + drift_per_session) ** i
        wave = 1.0 + amplitude * math.sin(2.0 * math.pi * i / period)
        closes.append(trend * wave)

    close = np.array(closes)
    half = daily_range_pct / 2.0
    frame = pd.DataFrame(
        {
            "open": np.round(close * (1.0 - half / 2.0), 4),
            "high": np.round(close * (1.0 + half), 4),
            "low": np.round(close * (1.0 - half), 4),
            "close": np.round(close, 4),
            "volume": np.full(sessions, volume, dtype=np.int64),
        },
        index=pd.DatetimeIndex(stamps),
    )
    return BarSeries(symbol=symbol.upper(), frame=frame)


def synthetic_set(
    symbols: Iterable[str], start: date, sessions: int, **kwargs
) -> BarSet:
    """A BarSet of synthetic series, one per symbol, each offset slightly.

    The offset keeps the symbols from being identical series under different
    names, which would make any per-symbol behaviour in the pipeline
    untestable.
    """
    out: dict[str, BarSeries] = {}
    for index, symbol in enumerate(symbols):
        params = dict(kwargs)
        params.setdefault("start_price", 500.0)
        params["start_price"] = float(params["start_price"]) * (1.0 + 0.11 * index)
        params.setdefault("period", 14 + index * 3)
        out[symbol.upper()] = synthetic_series(symbol, start, sessions, **params)
    return BarSet(out)
