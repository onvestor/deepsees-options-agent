"""Signal evaluation: does a bar frame satisfy a given signal profile?

The division of labour matters here. **Agent 1 chooses the parameterization;
this module computes the signal.** The model never sees a price and never
decides whether to trade -- it picks ``ema_fast``, ``confirmation_bars`` and
so on, and everything downstream is arithmetic that runs identically in live
trading and in replay.

**No I/O, no network.** Settings arrive as a :class:`SignalSettings` value the
caller built from config; bars arrive as a DataFrame. Nothing in this module
reads a file or opens a socket.

Evaluation is a conjunction of independent gates. Every gate is recorded in
the result whether it passed or failed, so the decision log can answer "why
was there no signal at 10:42" without re-running anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

import pandas as pd

from src.signals.indicators import REQUIRED_BAR_COLUMNS, atr, ema, rsi, vwap

__all__ = [
    "Direction",
    "SignalEvaluation",
    "SignalProfile",
    "SignalSettings",
    "evaluate",
]

Direction = Literal["long_calls", "long_puts", "both", "none"]
Outcome = Literal["long_calls", "long_puts", "none"]


@dataclass(frozen=True)
class SignalSettings:
    """Engine-wide parameters. Same for every symbol; from ``signals.*``."""

    ema_slow: int
    atr_period: int
    rsi_period: int
    rsi_long_max: float
    rsi_short_min: float
    min_bars: int
    vwap_session_anchored: bool

    @classmethod
    def from_limits(cls, limits: Any) -> "SignalSettings":
        """Adapter over a ``config.Section``. The only config-shaped code here."""
        return cls(
            ema_slow=limits.get_int("signals.ema_slow"),
            atr_period=limits.get_int("signals.atr_period"),
            rsi_period=limits.get_int("signals.rsi_period"),
            rsi_long_max=limits.get_float("signals.rsi_long_max"),
            rsi_short_min=limits.get_float("signals.rsi_short_min"),
            min_bars=limits.get_int("signals.min_bars"),
            vwap_session_anchored=limits.get_bool("signals.vwap_session_anchored"),
        )


@dataclass(frozen=True)
class SignalProfile:
    """Agent 1's output, after validation and clamping.

    Constructed here as a plain value object. Clamping to the allowed sets is
    the validator's job (Step 7) and deliberately does not live in this module
    -- the engine must behave identically whether a profile came from a model
    or from a replay fixture.
    """

    ema_fast: int
    confirmation_bars: int
    require_vwap_alignment: bool
    min_atr_multiple: float
    allowed_direction: Direction = "both"

    def __post_init__(self) -> None:
        if self.ema_fast < 1:
            raise ValueError(f"ema_fast must be >= 1, got {self.ema_fast}")
        if self.confirmation_bars < 1:
            raise ValueError(f"confirmation_bars must be >= 1, got {self.confirmation_bars}")
        if self.min_atr_multiple < 0:
            raise ValueError(f"min_atr_multiple must be >= 0, got {self.min_atr_multiple}")
        if self.allowed_direction not in ("long_calls", "long_puts", "both", "none"):
            raise ValueError(f"unknown allowed_direction {self.allowed_direction!r}")


@dataclass(frozen=True)
class SignalEvaluation:
    """Every gate's verdict, not just the answer."""

    direction: Outcome
    triggered: bool
    reasons: tuple[str, ...]
    gates: Mapping[str, bool] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)

    @property
    def blocked_by(self) -> tuple[str, ...]:
        return tuple(name for name, passed in self.gates.items() if not passed)


def _no_signal(reason: str, **extra: Any) -> SignalEvaluation:
    return SignalEvaluation(
        direction="none",
        triggered=False,
        reasons=(reason,),
        gates=extra.pop("gates", {}),
        metrics=extra.pop("metrics", {}),
    )


def _is_daily(bars: pd.DataFrame) -> bool:
    """True when consecutive bars are a day or more apart.

    Measured from the frame rather than configured, so a caller cannot assert
    a timeframe the data does not have. Two bars are enough; a single-bar frame
    cannot be evaluated anyway.
    """
    if len(bars.index) < 2:
        return True
    deltas = bars.index.to_series().diff().dropna()
    if deltas.empty:
        return True
    return deltas.median() >= pd.Timedelta(hours=20)


def evaluate(
    bars: pd.DataFrame,
    profile: SignalProfile,
    settings: SignalSettings,
    partial_last_bar: bool = False,
) -> SignalEvaluation:
    """Evaluate ``profile`` against ``bars``. Pure; ``bars`` is not mutated.

    Bars must be ascending in time with a DatetimeIndex and the standard OHLCV
    columns. The final row is the bar being evaluated.

    **``partial_last_bar`` says the final bar is still forming**, which is the
    normal case for a daily frame read during a live session: its close is the
    current price, but its high, low and volume are only "so far today".

    That distinction matters for exactly one indicator. ``close``, the EMAs and
    RSI are functions of the close, and a forming close *is* the current price
    -- using it is the entire point of reading a live frame. ``atr`` is not: a
    partial bar's true range is whatever the day has managed by now, so
    including it drags the average down and shrinks the denominator of the
    displacement gate. A quiet morning would then look like a large move in ATR
    units, which is precisely backwards.

    So ATR is computed over completed bars only. The engine cannot detect this
    from the data -- a forming bar looks like any other -- so the caller must
    say, and the default is False.
    """
    missing = [column for column in REQUIRED_BAR_COLUMNS if column not in bars.columns]
    if missing:
        raise ValueError(f"bars is missing required column(s): {missing}")
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise TypeError("bars must have a DatetimeIndex")
    if not bars.index.is_monotonic_increasing:
        raise ValueError("bars must be sorted ascending by time")

    if profile.allowed_direction == "none":
        return _no_signal("allowed_direction is none")
    if len(bars) < settings.min_bars:
        return _no_signal(f"insufficient history: {len(bars)} bars < {settings.min_bars}")

    fast = ema(bars["close"], profile.ema_fast)
    slow = ema(bars["close"], settings.ema_slow)
    strength = rsi(bars["close"], settings.rsi_period)
    volume_weighted = vwap(bars, session_anchored=settings.vwap_session_anchored)

    # ATR from completed bars only. See the note in the docstring: a forming
    # bar's range is partial, and a shrunken ATR inflates every displacement
    # measured against it.
    completed = bars.iloc[:-1] if partial_last_bar and len(bars) > 1 else bars
    average_range = atr(completed, settings.atr_period)

    latest = {
        "close": float(bars["close"].iloc[-1]),
        "ema_fast": float(fast.iloc[-1]),
        "ema_slow": float(slow.iloc[-1]),
        "atr": float(average_range.iloc[-1]) if pd.notna(average_range.iloc[-1]) else float("nan"),
        "rsi": float(strength.iloc[-1]) if pd.notna(strength.iloc[-1]) else float("nan"),
        "vwap": float(volume_weighted.iloc[-1]) if pd.notna(volume_weighted.iloc[-1]) else float("nan"),
    }

    if any(pd.isna(value) for value in latest.values()):
        return _no_signal("warm-up incomplete: indicator still NaN", metrics=latest)
    if latest["atr"] <= 0:
        return _no_signal("atr is zero: no measurable range", metrics=latest)

    # --- direction from the fast/slow relationship, held for N bars ---------
    spread = fast - slow
    window = spread.iloc[-profile.confirmation_bars :]
    bullish_confirmed = bool((window > 0).all())
    bearish_confirmed = bool((window < 0).all())

    if bullish_confirmed:
        candidate: Outcome = "long_calls"
    elif bearish_confirmed:
        candidate = "long_puts"
    else:
        return _no_signal(
            f"no directional agreement across the last {profile.confirmation_bars} bar(s)",
            metrics=latest,
        )

    displacement = abs(latest["close"] - latest["ema_slow"]) / latest["atr"]
    latest["atr_multiple"] = displacement

    is_long = candidate == "long_calls"
    gates = {
        "confirmation": True,
        "atr_displacement": displacement >= profile.min_atr_multiple,
        # DISABLED ON DAILY BARS -- measured 31 Aug 2026, not retuned.
        #
        # `vwap(session_anchored=True)` groups by `index.date` and accumulates
        # within each group. On a DAILY frame every bar is its own date, so
        # each group holds one row and the VWAP collapses to that same bar's
        # (H+L+C)/3 -- verified identical to machine precision on NVDA, PLTR
        # and TSLA. The gate then asks "did it close above the midpoint of its
        # own last candle", a one-bar shape test that is close to a coin flip
        # and says nothing about a 1-5 session thesis. It silently blocked 126
        # of 189 evaluations in the 31 Aug session.
        #
        # Not retuned, because no threshold fixes a degenerate indicator. The
        # gate is forced True whenever the frame is daily; it stays live for
        # any intraday frame, where a session-anchored VWAP means what its name
        # says.
        "vwap_alignment": (
            True
            if (not profile.require_vwap_alignment or _is_daily(bars))
            else ((latest["close"] > latest["vwap"]) if is_long
                  else (latest["close"] < latest["vwap"]))
        ),
        "rsi_guard": (
            latest["rsi"] <= settings.rsi_long_max
            if is_long
            else latest["rsi"] >= settings.rsi_short_min
        ),
        "direction_allowed": profile.allowed_direction in ("both", candidate),
    }

    reasons: list[str] = []
    if not gates["atr_displacement"]:
        reasons.append(
            f"displacement {displacement:.2f} ATR below min_atr_multiple "
            f"{profile.min_atr_multiple:.2f}"
        )
    if not gates["vwap_alignment"]:
        side = "above" if is_long else "below"
        reasons.append(f"close {latest['close']:.2f} not {side} vwap {latest['vwap']:.2f}")
    if not gates["rsi_guard"]:
        bound = settings.rsi_long_max if is_long else settings.rsi_short_min
        reasons.append(f"rsi {latest['rsi']:.1f} beyond guard {bound:.1f}")
    if not gates["direction_allowed"]:
        reasons.append(f"{candidate} not permitted by allowed_direction={profile.allowed_direction}")

    triggered = all(gates.values())
    return SignalEvaluation(
        direction=candidate if triggered else "none",
        triggered=triggered,
        reasons=tuple(reasons) if reasons else (f"{candidate} confirmed",),
        gates=gates,
        metrics=latest,
    )
