"""Indicator tests on synthetic frames with hand-computed answers.

Every expected value here is derived from the definition by hand, not from a
previous run of this code. A test that asserts "the same thing it did last
time" would pass just as happily on a wrong implementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.signals.indicators import atr, ema, percentile_rank, rsi, true_range, vwap


def make_bars(closes, highs=None, lows=None, volumes=None, start="2026-08-24 09:30", freq="1min"):
    index = pd.date_range(start=start, periods=len(closes), freq=freq)
    closes = np.asarray(closes, dtype="float64")
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes if highs is None else np.asarray(highs, dtype="float64"),
            "low": closes if lows is None else np.asarray(lows, dtype="float64"),
            "close": closes,
            "volume": np.full(len(closes), 100.0) if volumes is None else np.asarray(volumes, "float64"),
        },
        index=index,
    )


# --- EMA -------------------------------------------------------------------


def test_ema_of_a_constant_is_that_constant():
    series = pd.Series([5.0] * 20)
    assert ema(series, 9).eq(5.0).all()


def test_ema_span_one_is_the_identity():
    series = pd.Series([1.0, 7.0, 3.0, 9.0])
    pd.testing.assert_series_equal(ema(series, 1), series, check_names=False)


def test_ema_matches_hand_computation():
    """span=3 -> alpha = 2/(3+1) = 0.5; seed on the first value."""
    series = pd.Series([10.0, 20.0, 30.0])
    result = ema(series, 3)
    assert result.iloc[0] == pytest.approx(10.0)          # seed
    assert result.iloc[1] == pytest.approx(15.0)          # .5*20 + .5*10
    assert result.iloc[2] == pytest.approx(22.5)          # .5*30 + .5*15


def test_ema_lags_a_rising_series():
    series = pd.Series(np.arange(1.0, 51.0))
    result = ema(series, 10)
    assert (result.iloc[10:] < series.iloc[10:]).all()


def test_ema_does_not_mutate_input():
    series = pd.Series([1.0, 2.0, 3.0])
    before = series.copy()
    ema(series, 2)
    pd.testing.assert_series_equal(series, before)


@pytest.mark.parametrize("bad", [0, -1])
def test_ema_rejects_non_positive_span(bad):
    with pytest.raises(ValueError):
        ema(pd.Series([1.0]), bad)


def test_ema_rejects_bool_span():
    with pytest.raises(TypeError):
        ema(pd.Series([1.0]), True)


# --- true range and ATR ----------------------------------------------------


def test_true_range_first_bar_is_its_own_range():
    bars = make_bars([10.0, 11.0], highs=[12.0, 13.0], lows=[9.0, 10.0])
    assert true_range(bars).iloc[0] == pytest.approx(3.0)


def test_true_range_uses_the_gap_when_it_dominates():
    """Prev close 10, next bar 20-21: the gap (11) beats the intrabar range (1)."""
    bars = make_bars([10.0, 20.5], highs=[10.0, 21.0], lows=[10.0, 20.0])
    assert true_range(bars).iloc[1] == pytest.approx(11.0)


def test_atr_of_constant_range_is_that_range():
    n = 30
    closes = np.full(n, 100.0)
    bars = make_bars(closes, highs=closes + 1.0, lows=closes - 1.0)
    result = atr(bars, 14)
    assert np.isnan(result.iloc[12])                      # warm-up
    assert result.iloc[13] == pytest.approx(2.0)          # seed = mean of 14 TRs
    assert result.iloc[-1] == pytest.approx(2.0)


def test_atr_wilder_step_matches_hand_computation():
    """After the seed, atr = prev + (tr - prev)/period."""
    n = 20
    closes = np.full(n, 100.0)
    highs = closes + 1.0
    lows = closes - 1.0
    highs[15] = 105.0                                     # one wide bar, TR = 6
    bars = make_bars(closes, highs=highs, lows=lows)
    result = atr(bars, 14)
    assert result.iloc[14] == pytest.approx(2.0)
    expected = 2.0 + (6.0 - 2.0) / 14
    assert result.iloc[15] == pytest.approx(expected)


def test_atr_warmup_is_nan():
    bars = make_bars(np.full(10, 50.0))
    assert atr(bars, 14).isna().all()


# --- RSI -------------------------------------------------------------------


def test_rsi_of_monotonic_rise_is_one_hundred():
    series = pd.Series(np.arange(1.0, 40.0))
    assert rsi(series, 14).iloc[-1] == pytest.approx(100.0)


def test_rsi_of_monotonic_fall_is_zero():
    series = pd.Series(np.arange(40.0, 1.0, -1.0))
    assert rsi(series, 14).iloc[-1] == pytest.approx(0.0)


def test_rsi_of_a_flat_series_is_fifty_not_one_hundred():
    """A market that has not moved is neutral, not maximally overbought."""
    series = pd.Series([25.0] * 40)
    assert rsi(series, 14).iloc[-1] == pytest.approx(50.0)


def test_rsi_alternating_equal_moves_hovers_near_fifty_and_leans_on_the_last_bar():
    """Wilder smoothing is recursive, so it does not settle exactly on 50.

    With equal up and down moves the average gain and average loss stay close,
    but the most recent change still carries the heaviest weight. Ending on an
    up bar must read just above 50, ending on a down bar just below -- and
    symmetrically so.
    """
    up_last = pd.Series([100.0 + (1.0 if i % 2 else 0.0) for i in range(60)])
    down_last = pd.Series([100.0 + (0.0 if i % 2 else 1.0) for i in range(60)])

    ends_up = rsi(up_last, 14).iloc[-1]
    ends_down = rsi(down_last, 14).iloc[-1]

    assert 50.0 < ends_up < 55.0
    assert 45.0 < ends_down < 50.0
    assert ends_up - 50.0 == pytest.approx(50.0 - ends_down, abs=1e-9)


def test_rsi_is_bounded():
    rng = np.random.default_rng(20260822)
    series = pd.Series(100 + rng.standard_normal(500).cumsum())
    values = rsi(series, 14).dropna()
    assert values.between(0.0, 100.0).all()


def test_rsi_warmup_is_nan():
    series = pd.Series(np.arange(1.0, 10.0))
    assert rsi(series, 14).isna().all()


# --- VWAP ------------------------------------------------------------------


def test_vwap_of_constant_price_is_that_price():
    bars = make_bars([50.0] * 10)
    assert vwap(bars).eq(50.0).all()


def test_vwap_is_volume_weighted_not_a_mean():
    """Typical price = (H+L+C)/3; here H=L=C so typical == close.

    Bar 1: 10 @ vol 1.  Bar 2: 20 @ vol 3.
    VWAP = (10*1 + 20*3) / 4 = 17.5, not the simple mean of 15.
    """
    bars = make_bars([10.0, 20.0], volumes=[1.0, 3.0])
    assert vwap(bars).iloc[-1] == pytest.approx(17.5)


def test_vwap_uses_typical_price():
    bars = make_bars([10.0], highs=[12.0], lows=[8.0], volumes=[5.0])
    assert vwap(bars).iloc[0] == pytest.approx(10.0)      # (12+8+10)/3


def test_vwap_resets_each_session():
    day_one = make_bars([10.0, 10.0], volumes=[1.0, 1.0], start="2026-08-24 09:30")
    day_two = make_bars([90.0, 90.0], volumes=[1.0, 1.0], start="2026-08-25 09:30")
    bars = pd.concat([day_one, day_two])
    result = vwap(bars, session_anchored=True)
    assert result.iloc[1] == pytest.approx(10.0)
    assert result.iloc[2] == pytest.approx(90.0)          # reset, not blended
    assert result.iloc[3] == pytest.approx(90.0)


def test_vwap_unanchored_runs_continuously():
    day_one = make_bars([10.0, 10.0], volumes=[1.0, 1.0], start="2026-08-24 09:30")
    day_two = make_bars([90.0, 90.0], volumes=[1.0, 1.0], start="2026-08-25 09:30")
    bars = pd.concat([day_one, day_two])
    assert vwap(bars, session_anchored=False).iloc[-1] == pytest.approx(50.0)


def test_vwap_survives_zero_volume_bars():
    bars = make_bars([10.0, 20.0, 30.0], volumes=[0.0, 0.0, 0.0])
    result = vwap(bars)
    assert result.isna().all() or np.isfinite(result.dropna()).all()


def test_vwap_requires_datetime_index_when_anchored():
    bars = make_bars([1.0, 2.0]).reset_index(drop=True)
    with pytest.raises(TypeError):
        vwap(bars, session_anchored=True)


# --- percentile rank -------------------------------------------------------


def test_percentile_rank_of_the_maximum_is_one():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 10.0])
    assert percentile_rank(series, 5).iloc[-1] == pytest.approx(1.0)


def test_percentile_rank_of_the_minimum_is_zero():
    series = pd.Series([5.0, 4.0, 3.0, 2.0, 0.5])
    assert percentile_rank(series, 5).iloc[-1] == pytest.approx(0.0)


def test_percentile_rank_midpoint():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 2.5])
    assert percentile_rank(series, 5).iloc[-1] == pytest.approx(0.5)


def test_percentile_rank_warmup_is_nan():
    series = pd.Series([1.0, 2.0])
    assert percentile_rank(series, 5).isna().all()


# --- shared contract -------------------------------------------------------


@pytest.mark.parametrize("column", ["open", "high", "low", "close", "volume"])
def test_frame_functions_require_ohlcv(column):
    bars = make_bars([1.0, 2.0, 3.0]).drop(columns=[column])
    with pytest.raises(ValueError):
        true_range(bars)


def test_indicators_preserve_the_index():
    bars = make_bars(np.linspace(10.0, 20.0, 40))
    for result in (atr(bars, 14), vwap(bars), ema(bars["close"], 9), rsi(bars["close"], 14)):
        pd.testing.assert_index_equal(result.index, bars.index)
