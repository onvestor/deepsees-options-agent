"""Deterministic halts. Never consults a model, ever.

That is a structural property, not a convention: every function here takes
numbers and returns verdicts. There is no parameter through which a model
response could arrive, and ``tests/test_risk.py`` asserts this module imports
nothing from the agent layer.

The reason is the third invariant. A kill switch exists precisely for the
situations where the rest of the system may be wrong -- a bad regime call, a
mispriced chain, a broker misbehaving. Routing that judgment through the same
machinery that might be causing the problem defeats the purpose.

Kill switches halt **new entries**. They never force liquidation: dumping a
book into whatever liquidity exists at the moment a loss limit trips is its
own risk, and the deterministic exit layer already owns position-level exits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "KillSwitchState",
    "SwitchVerdict",
    "evaluate_kill_switches",
    "KillSwitchLimits",
]


@dataclass(frozen=True)
class KillSwitchLimits:
    daily_loss_halt_pct: float
    daily_loss_halt_abs: float
    consecutive_losing_trades: int
    broker_error_streak: int
    max_session_drawdown_pct: float
    halt_resets_next_session: bool

    @classmethod
    def from_limits(cls, limits: Any) -> "KillSwitchLimits":
        return cls(
            daily_loss_halt_pct=limits.get_float("killswitch.daily_loss_halt_pct"),
            daily_loss_halt_abs=limits.get_float("killswitch.daily_loss_halt_abs"),
            consecutive_losing_trades=limits.get_int("killswitch.consecutive_losing_trades"),
            broker_error_streak=limits.get_int("killswitch.broker_error_streak"),
            max_session_drawdown_pct=limits.get_float("killswitch.max_session_drawdown_pct"),
            halt_resets_next_session=limits.get_bool("killswitch.halt_resets_next_session"),
        )


@dataclass(frozen=True)
class KillSwitchState:
    """Observed session state. Numbers only -- no model output can enter here."""

    start_of_day_equity: float
    current_equity: float
    session_peak_equity: float
    consecutive_losing_trades: int = 0
    broker_error_streak: int = 0

    @property
    def session_pnl(self) -> float:
        return self.current_equity - self.start_of_day_equity

    @property
    def session_loss_pct(self) -> float:
        if self.start_of_day_equity <= 0:
            return 0.0
        return max(0.0, -self.session_pnl) / self.start_of_day_equity

    @property
    def drawdown_from_peak_pct(self) -> float:
        """Drawdown from the intraday equity peak, which is not the same as a
        loss from the open -- a session can be up on the day and still have
        given back more than the limit."""
        peak = max(self.session_peak_equity, self.start_of_day_equity)
        if peak <= 0:
            return 0.0
        return max(0.0, (peak - self.current_equity) / peak)


@dataclass(frozen=True)
class SwitchVerdict:
    switch: str
    threshold: float
    observed: float
    fired: bool
    scope: str = "session"
    halts_new_entries: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "switch": self.switch,
            "threshold": self.threshold,
            "observed": self.observed,
            "fired": self.fired,
            "scope": self.scope,
            "halts_new_entries": self.halts_new_entries,
        }


def evaluate_kill_switches(
    state: KillSwitchState, limits: KillSwitchLimits
) -> tuple[SwitchVerdict, ...]:
    """Evaluate **every** switch and return all verdicts, fired or not.

    Like the prefilter, nothing short-circuits: a report saying only which
    switch tripped first hides how close the others were, and "we were one
    trade from the consecutive-loss halt" is exactly what a post-session
    review needs to know.
    """
    return (
        SwitchVerdict(
            switch="daily_loss_halt_pct",
            threshold=limits.daily_loss_halt_pct,
            observed=state.session_loss_pct,
            fired=state.session_loss_pct >= limits.daily_loss_halt_pct,
            scope="account",
        ),
        SwitchVerdict(
            switch="daily_loss_halt_abs",
            threshold=limits.daily_loss_halt_abs,
            observed=max(0.0, -state.session_pnl),
            fired=max(0.0, -state.session_pnl) >= limits.daily_loss_halt_abs,
            scope="account",
        ),
        SwitchVerdict(
            switch="max_session_drawdown_pct",
            threshold=limits.max_session_drawdown_pct,
            observed=state.drawdown_from_peak_pct,
            fired=state.drawdown_from_peak_pct >= limits.max_session_drawdown_pct,
            scope="account",
        ),
        SwitchVerdict(
            switch="consecutive_losing_trades",
            threshold=float(limits.consecutive_losing_trades),
            observed=float(state.consecutive_losing_trades),
            fired=state.consecutive_losing_trades >= limits.consecutive_losing_trades,
            scope="session",
        ),
        SwitchVerdict(
            switch="broker_error_streak",
            threshold=float(limits.broker_error_streak),
            observed=float(state.broker_error_streak),
            fired=state.broker_error_streak >= limits.broker_error_streak,
            scope="session",
        ),
    )


def is_halted(verdicts: tuple[SwitchVerdict, ...]) -> bool:
    """Any fired switch that halts entries halts them. No quorum, no override."""
    return any(v.fired and v.halts_new_entries for v in verdicts)


def fired_switches(verdicts: tuple[SwitchVerdict, ...]) -> tuple[str, ...]:
    return tuple(v.switch for v in verdicts if v.fired)
