"""Technical indicators as pure functions over pandas frames.

**This module performs no I/O and makes no network calls.** Neither does
anything else under ``src/signals/``. That purity is what makes the replay
harness possible, and the replay harness is what lets prompt iteration happen
offline instead of burning live market sessions. ``tests/test_signals_purity``
enforces it rather than trusting the convention.

Conventions, chosen so results are hand-checkable in tests:

* **EMA** seeds on the first observation and smooths with ``adjust=False``.
  ``ema[i] = a*x[i] + (1-a)*ema[i-1]`` with ``a = 2/(span+1)``.
* **RSI and ATR** use Wilder's smoothing: a simple average over the first
  ``period`` observations, then ``prev + (x - prev)/period``. This is the
  textbook definition, not the ``ewm`` approximation, so a test can compute
  the expected value by hand.
* **VWAP** is session-anchored and resets at each new session date, which is
  what an intraday VWAP means. A VWAP running continuously across days is a
  different and far less useful number.

Every function returns a Series aligned to the input index, NaN during warm-up.
No function mutates its argument.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "atr",
    "ema",
    "percentile_rank",
    "rsi",
    "true_range",
    "vwap",
    "REQUIRED_BAR_COLUMNS",
]

REQUIRED_BAR_COLUMNS = ("open", "high", "low", "close", "volume")


def _validate_span(span: int, name: str = "span") -> None:
    if not isinstance(span, (int, np.integer)) or isinstance(span, bool):
        raise TypeError(f"{name} must be an int, got {type(span).__name__}")
    if span < 1:
        raise ValueError(f"{name} must be >= 1, got {span}")


def _as_float_series(values: pd.Series, name: str) -> pd.Series:
    if not isinstance(values, pd.Series):
        raise TypeError(f"{name} must be a pandas Series, got {type(values).__name__}")
    return values.astype("float64")


def _require_columns(bars: pd.DataFrame) -> None:
    if not isinstance(bars, pd.DataFrame):
        raise TypeError(f"bars must be a DataFrame, got {type(bars).__name__}")
    missing = [c for c in REQUIRED_BAR_COLUMNS if c not in bars.columns]
    if missing:
        raise ValueError(f"bars is missing required column(s): {missing}")


def ema(values: pd.Series, span: int) -> pd.Series:
    """Exponential moving average, seeded on the first observation.

    ``span=1`` returns the input unchanged, which is the correct degenerate
    case and a useful test anchor.
    """
    _validate_span(span)
    series = _as_float_series(values, "values")
    return series.ewm(span=span, adjust=False).mean()


def _wilder_smooth(values: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing: SMA seed over the first ``period``, then recursive.

    Written explicitly rather than as an ``ewm`` call so the seeding matches
    the textbook exactly. Intraday frames are hundreds of rows, so the loop
    costs nothing and buys a definition a test can verify by hand.
    """
    array = values.to_numpy(dtype="float64")
    out = np.full(array.shape, np.nan, dtype="float64")
    if len(array) < period:
        return pd.Series(out, index=values.index)

    seed_window = array[:period]
    if np.isnan(seed_window).any():
        return pd.Series(out, index=values.index)

    previous = seed_window.mean()
    out[period - 1] = previous
    for i in range(period, len(array)):
        current = array[i]
        if np.isnan(current):
            out[i] = previous
            continue
        previous = previous + (current - previous) / period
        out[i] = previous
    return pd.Series(out, index=values.index)


def true_range(bars: pd.DataFrame) -> pd.Series:
    """max(high-low, |high-prev_close|, |low-prev_close|).

    The first bar has no previous close, so its true range is simply its own
    high-low rather than NaN -- otherwise ATR loses a bar of warm-up for no
    reason.
    """
    _require_columns(bars)
    high = bars["high"].astype("float64")
    low = bars["low"].astype("float64")
    previous_close = bars["close"].astype("float64").shift(1)

    spans = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    )
    result = spans.max(axis=1)
    if len(result):
        result.iloc[0] = float(high.iloc[0] - low.iloc[0])
    return result


def atr(bars: pd.DataFrame, period: int) -> pd.Series:
    """Average true range, Wilder-smoothed."""
    _validate_span(period, "period")
    return _wilder_smooth(true_range(bars), period)


def rsi(values: pd.Series, period: int) -> pd.Series:
    """Relative strength index, Wilder-smoothed.

    Two boundary conventions, both deliberate and both tested:

    * no losses and some gains  -> 100.0
    * no losses and no gains (a flat series) -> 50.0, not 100. A market that
      has not moved is neutral, not maximally overbought.
    """
    _validate_span(period, "period")
    series = _as_float_series(values, "values")
    delta = series.diff()
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)

    # diff() makes row 0 NaN; drop it so the Wilder seed averages `period`
    # real changes, then reindex back onto the original frame.
    average_gain = _wilder_smooth(gains.iloc[1:], period).reindex(series.index)
    average_loss = _wilder_smooth(losses.iloc[1:], period).reindex(series.index)

    result = pd.Series(np.nan, index=series.index, dtype="float64")
    valid = average_gain.notna() & average_loss.notna()

    flat = valid & (average_loss == 0.0) & (average_gain == 0.0)
    all_gain = valid & (average_loss == 0.0) & (average_gain > 0.0)
    normal = valid & (average_loss > 0.0)

    rs = average_gain[normal] / average_loss[normal]
    result[normal] = 100.0 - (100.0 / (1.0 + rs))
    result[all_gain] = 100.0
    result[flat] = 50.0
    return result


def vwap(bars: pd.DataFrame, session_anchored: bool = True) -> pd.Series:
    """Volume-weighted average price on the typical price (H+L+C)/3.

    Session-anchored by default: the accumulation resets at each new date in
    the index, because an intraday VWAP that never resets stops being VWAP by
    the second day.

    Zero-volume bars carry the previous value forward rather than producing a
    divide-by-zero -- illiquid options underlyings do print empty bars.
    """
    _require_columns(bars)
    if not isinstance(bars.index, pd.DatetimeIndex) and session_anchored:
        raise TypeError("session-anchored vwap requires a DatetimeIndex")

    typical = (
        bars["high"].astype("float64")
        + bars["low"].astype("float64")
        + bars["close"].astype("float64")
    ) / 3.0
    volume = bars["volume"].astype("float64")
    notional = typical * volume

    if session_anchored:
        sessions = pd.Series(bars.index.date, index=bars.index)
        cumulative_notional = notional.groupby(sessions).cumsum()
        cumulative_volume = volume.groupby(sessions).cumsum()
    else:
        cumulative_notional = notional.cumsum()
        cumulative_volume = volume.cumsum()

    with np.errstate(invalid="ignore", divide="ignore"):
        result = cumulative_notional / cumulative_volume
    return result.replace([np.inf, -np.inf], np.nan).ffill()


def percentile_rank(values: pd.Series, window: int) -> pd.Series:
    """Rolling percentile rank of the latest value within its trailing window.

    Returns 0.0-1.0. Used for IV rank and range-position style features.
    """
    _validate_span(window, "window")
    series = _as_float_series(values, "values")
    if window == 1:
        return pd.Series(1.0, index=series.index).where(series.notna())

    def rank(sample: np.ndarray) -> float:
        latest = sample[-1]
        if np.isnan(latest):
            return np.nan
        prior = sample[:-1]
        prior = prior[~np.isnan(prior)]
        if prior.size == 0:
            return np.nan
        return float((prior <= latest).sum() / prior.size)

    return series.rolling(window=window, min_periods=window).apply(rank, raw=True)
