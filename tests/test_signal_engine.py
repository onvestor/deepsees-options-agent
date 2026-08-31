"""Signal engine tests on synthetic frames built to trigger one gate at a time.

Each frame is constructed so that exactly one condition is in question and
everything else is comfortably satisfied. A test that fails because three
gates changed at once tells you nothing about which one broke.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.signals.engine import SignalEvaluation, SignalProfile, SignalSettings, evaluate

SETTINGS = SignalSettings(
    ema_slow=21,
    atr_period=14,
    rsi_period=14,
    rsi_long_max=78.0,
    rsi_short_min=22.0,
    min_bars=60,
    vwap_session_anchored=True,
)

PROFILE = SignalProfile(
    ema_fast=9,
    confirmation_bars=2,
    require_vwap_alignment=True,
    min_atr_multiple=0.6,
    allowed_direction="both",
)


def frame(closes, highs=None, lows=None, volumes=None, start="2026-08-24 09:30"):
    closes = np.asarray(closes, dtype="float64")
    index = pd.date_range(start=start, periods=len(closes), freq="1min")
    noise = 0.25
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes + noise if highs is None else np.asarray(highs, "float64"),
            "low": closes - noise if lows is None else np.asarray(lows, "float64"),
            "close": closes,
            "volume": np.full(len(closes), 1000.0) if volumes is None else np.asarray(volumes, "float64"),
        },
        index=index,
    )


# A perfectly straight line reads RSI exactly 100 and is correctly rejected by
# the overbought guard. Real trends breathe, so these carry a deterministic
# pullback -- amplitude chosen so RSI lands inside the guard band (75.7 up,
# 25.1 down) while the EMA relationship stays unambiguous.
_PULLBACK = 1.2


def uptrend(n=120, slope=0.20, base=100.0):
    wave = _PULLBACK * np.sin(np.arange(n) / 2.0)
    return frame(base + slope * np.arange(n) + wave)


def downtrend(n=120, slope=0.20, base=140.0):
    wave = _PULLBACK * np.sin(np.arange(n) / 2.0)
    return frame(base - slope * np.arange(n) + wave)


def flat(n=120, level=100.0):
    rng = np.random.default_rng(7)
    return frame(level + rng.normal(0, 0.02, n))


# --- happy paths -----------------------------------------------------------


def test_sustained_uptrend_triggers_long_calls():
    result = evaluate(uptrend(), PROFILE, SETTINGS)
    assert result.direction == "long_calls"
    assert result.triggered
    assert result.blocked_by == ()


def test_sustained_downtrend_triggers_long_puts():
    result = evaluate(downtrend(), PROFILE, SETTINGS)
    assert result.direction == "long_puts"
    assert result.triggered


def test_result_carries_every_metric():
    result = evaluate(uptrend(), PROFILE, SETTINGS)
    for key in ("close", "ema_fast", "ema_slow", "atr", "rsi", "vwap", "atr_multiple"):
        assert key in result.metrics
        assert np.isfinite(result.metrics[key])


def test_every_gate_is_reported_pass_or_fail():
    result = evaluate(uptrend(), PROFILE, SETTINGS)
    assert set(result.gates) == {
        "confirmation", "atr_displacement", "vwap_alignment", "rsi_guard", "direction_allowed",
    }


# --- gates, one at a time --------------------------------------------------


def test_allowed_direction_none_blocks_everything():
    profile = SignalProfile(9, 2, True, 0.6, allowed_direction="none")
    result = evaluate(uptrend(), profile, SETTINGS)
    assert result.direction == "none"
    assert not result.triggered
    assert "allowed_direction is none" in result.reasons[0]


def test_long_puts_only_blocks_a_bullish_setup():
    profile = SignalProfile(9, 2, True, 0.6, allowed_direction="long_puts")
    result = evaluate(uptrend(), profile, SETTINGS)
    assert result.direction == "none"
    assert result.gates["direction_allowed"] is False
    assert result.gates["confirmation"] is True          # the setup was valid


def test_long_calls_only_permits_the_bullish_setup():
    profile = SignalProfile(9, 2, True, 0.6, allowed_direction="long_calls")
    assert evaluate(uptrend(), profile, SETTINGS).direction == "long_calls"


def test_insufficient_history_blocks():
    result = evaluate(uptrend(n=30), PROFILE, SETTINGS)
    assert result.direction == "none"
    assert "insufficient history" in result.reasons[0]


def test_flat_market_fails_the_atr_displacement_gate():
    """Price sitting on its own slow EMA has nowhere near 0.6 ATR of travel."""
    result = evaluate(flat(), PROFILE, SETTINGS)
    assert result.direction == "none"
    assert result.gates["atr_displacement"] is False


def test_zero_displacement_requirement_lets_a_flat_market_through_confirmation():
    """Isolates the ATR gate: with the floor at zero, only agreement matters."""
    lenient = SignalProfile(9, 2, False, 0.0, allowed_direction="both")
    result = evaluate(flat(), lenient, SETTINGS)
    assert result.gates["atr_displacement"] is True


def test_vwap_alignment_can_veto_an_otherwise_valid_long():
    """Rallying, but still underwater against the session's real average price.

    Heavy early volume at 130 anchors VWAP there; the later rally from 100 to
    110 turns the EMAs bullish on thin volume. Price is rising and still 20
    points below where the session actually traded -- exactly the trap the
    VWAP gate exists to catch. RSI is unclamped here so only the VWAP gate
    differs between the two profiles.
    """
    closes = np.concatenate([np.full(60, 130.0), 100 + (10 / 59) * np.arange(60)])
    volumes = np.concatenate([np.full(60, 100_000.0), np.full(60, 10.0)])
    bars = frame(closes, volumes=volumes)
    lenient_rsi = SignalSettings(**{**SETTINGS.__dict__, "rsi_long_max": 100.0})

    strict = SignalProfile(9, 2, True, 0.0, allowed_direction="both")
    relaxed = SignalProfile(9, 2, False, 0.0, allowed_direction="both")

    blocked = evaluate(bars, strict, lenient_rsi)
    allowed = evaluate(bars, relaxed, lenient_rsi)

    assert blocked.metrics["close"] < blocked.metrics["vwap"]
    assert blocked.gates["vwap_alignment"] is False
    assert blocked.direction == "none"
    assert allowed.gates["vwap_alignment"] is True
    assert allowed.direction == "long_calls"


def test_rsi_guard_blocks_a_parabolic_long():
    """A vertical ramp is overbought; the guard refuses to chase it."""
    bars = frame(100 * 1.02 ** np.arange(120))
    tight = SignalSettings(**{**SETTINGS.__dict__, "rsi_long_max": 60.0})
    result = evaluate(bars, SignalProfile(9, 2, False, 0.0, "both"), tight)
    assert result.gates["rsi_guard"] is False
    assert result.direction == "none"


def test_rsi_guard_blocks_a_capitulation_short():
    bars = frame(200 * 0.98 ** np.arange(120))
    tight = SignalSettings(**{**SETTINGS.__dict__, "rsi_short_min": 40.0})
    result = evaluate(bars, SignalProfile(9, 2, False, 0.0, "both"), tight)
    assert result.gates["rsi_guard"] is False


def test_confirmation_bars_require_sustained_agreement():
    """One bar of crossover is not a trend.

    A long downtrend that has only just turned up satisfies a 1-bar
    confirmation but not a 10-bar one.
    """
    closes = np.concatenate([140 - 0.20 * np.arange(110), 118 + 1.2 * np.arange(6)])
    bars = frame(closes)
    lenient_rsi = SignalSettings(**{**SETTINGS.__dict__, "rsi_long_max": 100.0})

    # The EMA spread turns positive on the final bar only.
    lenient = evaluate(bars, SignalProfile(9, 1, False, 0.0, "both"), lenient_rsi)
    strict = evaluate(bars, SignalProfile(9, 10, False, 0.0, "both"), lenient_rsi)

    assert lenient.gates["confirmation"] is True
    assert lenient.direction == "long_calls"
    assert strict.direction == "none"
    assert "no directional agreement" in strict.reasons[0]


# --- input contract --------------------------------------------------------


def test_bars_must_be_sorted():
    bars = uptrend()
    with pytest.raises(ValueError):
        evaluate(bars.iloc[::-1], PROFILE, SETTINGS)


def test_bars_must_have_a_datetime_index():
    bars = uptrend().reset_index(drop=True)
    with pytest.raises(TypeError):
        evaluate(bars, PROFILE, SETTINGS)


@pytest.mark.parametrize("column", ["open", "high", "low", "close", "volume"])
def test_missing_columns_raise(column):
    with pytest.raises(ValueError):
        evaluate(uptrend().drop(columns=[column]), PROFILE, SETTINGS)


def test_evaluate_does_not_mutate_the_frame():
    bars = uptrend()
    before = bars.copy()
    evaluate(bars, PROFILE, SETTINGS)
    pd.testing.assert_frame_equal(bars, before)


def test_evaluation_is_deterministic():
    bars = uptrend()
    first = evaluate(bars, PROFILE, SETTINGS)
    second = evaluate(bars, PROFILE, SETTINGS)
    assert first == second


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ema_fast": 0},
        {"confirmation_bars": 0},
        {"min_atr_multiple": -1.0},
        {"allowed_direction": "short_calls"},
    ],
)
def test_profile_rejects_invalid_values(kwargs):
    base = {
        "ema_fast": 9, "confirmation_bars": 2, "require_vwap_alignment": True,
        "min_atr_multiple": 0.6, "allowed_direction": "both",
    }
    with pytest.raises(ValueError):
        SignalProfile(**{**base, **kwargs})


def test_settings_load_from_a_config_section():
    """SignalSettings.from_limits is the only config-shaped code in src/signals."""
    from src.config import load_config

    settings = SignalSettings.from_limits(load_config().limits)
    assert settings.min_bars > 0
    assert 0 < settings.rsi_short_min < settings.rsi_long_max < 100


def test_evaluation_is_frozen():
    result = evaluate(uptrend(), PROFILE, SETTINGS)
    assert isinstance(result, SignalEvaluation)
    with pytest.raises(Exception):
        result.direction = "long_puts"  # type: ignore[misc]


# --- vwap is degenerate on daily bars, and disabled there -------------------


def _daily(n=60, start=100.0):
    import numpy as np
    import pandas as pd

    idx = pd.date_range("2026-05-01", periods=n, freq="B")
    close = np.linspace(start, start * 1.15, n)
    return pd.DataFrame(
        {"open": close * 0.995, "high": close * 1.02, "low": close * 0.98,
         "close": close, "volume": np.full(n, 1_000_000)},
        index=idx,
    )


def test_session_anchored_vwap_collapses_to_one_bar_on_a_daily_frame():
    """The finding behind disabling the gate: with one bar per date, the
    accumulation group has a single row and VWAP becomes that bar's own
    (H+L+C)/3 -- carrying no information beyond the candle's shape."""
    from src.signals.indicators import vwap

    bars = _daily()
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    computed = vwap(bars, session_anchored=True)
    assert (computed - typical).abs().max() < 1e-9


def test_vwap_gate_is_forced_true_on_a_daily_frame():
    """Not retuned -- no threshold repairs a degenerate indicator."""
    from src.signals.engine import SignalProfile, SignalSettings, evaluate

    bars = _daily()
    # Close below the bar's own typical price: the gate would block if live.
    bars.loc[bars.index[-1], "close"] = bars["low"].iloc[-1]

    settings = SignalSettings(ema_slow=21, atr_period=14, rsi_period=14,
                              rsi_long_max=95.0, rsi_short_min=5.0, min_bars=30,
                              vwap_session_anchored=True)
    profile = SignalProfile(ema_fast=9, confirmation_bars=1,
                            require_vwap_alignment=True, min_atr_multiple=0.0,
                            allowed_direction="both")
    result = evaluate(bars, profile, settings)
    assert result.gates["vwap_alignment"] is True


def test_the_gate_still_applies_on_an_intraday_frame():
    """Disabled for daily only. A session-anchored VWAP over hourly bars means
    what its name says, and the gate must come back when one is evaluated."""
    import numpy as np
    import pandas as pd

    from src.signals.engine import SignalProfile, SignalSettings, evaluate
    from src.signals.engine import _is_daily

    idx = pd.date_range("2026-08-31 09:30", periods=60, freq="h")
    close = np.linspace(100.0, 90.0, 60)          # falling: closes under vwap
    bars = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": np.full(60, 1_000)},
        index=idx,
    )
    assert not _is_daily(bars)

    settings = SignalSettings(ema_slow=21, atr_period=14, rsi_period=14,
                              rsi_long_max=95.0, rsi_short_min=5.0, min_bars=30,
                              vwap_session_anchored=True)
    profile = SignalProfile(ema_fast=9, confirmation_bars=1,
                            require_vwap_alignment=True, min_atr_multiple=0.0,
                            allowed_direction="both")
    result = evaluate(bars, profile, settings)
    assert "vwap_alignment" in result.gates


def test_is_daily_distinguishes_the_two_frames():
    import pandas as pd

    from src.signals.engine import _is_daily

    assert _is_daily(_daily())
    hourly = _daily()
    hourly.index = pd.date_range("2026-08-31 09:30", periods=len(hourly), freq="h")
    assert not _is_daily(hourly)
