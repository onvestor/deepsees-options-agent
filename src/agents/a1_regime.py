"""Agent 1 -- Regime & Signal Profiler.

Selects the signal parameterisation. It does **not** compute the signal: the
numbers it reads were computed deterministically in ``src/signals/``, and the
profile it returns is fed back into those same pure functions. The model's
whole job is choosing parameters, which is a judgment; the arithmetic on
either side of it is not.

**The profile is locked for the session.** Under
``agents.a1.profile_locked_for_session`` the first accepted profile for a
symbol is reused for the rest of the session and no second call is made. This
began as a 30-minute anti-thrash window in the intraday design and became a
full-session lock with the swing revision -- a regime read for a 1-5 session
hold is a daily judgment, and re-deriving it hourly would let the parameters
drift underneath a thesis that has not changed.

The lock stores the *accepted* decision, after clamping and forcing. Caching
the raw model output would let a value the validator rejected re-enter through
the cache on the next call.

**Path is ENTRY**, so any failure -- malformed output, timeout, a schema
violation surviving one retry -- is a skip. No profile means no signal
evaluation, which means no trade. There is no default profile, because a
default here would be a guess about market regime, which is exactly the
judgment the agent exists to make.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

from src.agents.prompt_loader import load_and_render
from src.agents.runner import AgentPath, AgentRunner, RunResult
from src.agents.schemas import RegimeDecision

log = logging.getLogger(__name__)

PROMPT_NAME = "a1_regime.txt"


@dataclass(frozen=True)
class RegimeInputs:
    """Everything Agent 1 reads. All computed in code, none of it by a model."""

    symbol: str
    spot: float
    atr: float
    atr_pct_of_spot: float
    realized_vol: float
    rsi: float
    ema_fast_value: float
    ema_slow_value: float
    trend_pct_20d: float
    above_vwap: bool
    # Bounded observations carried over from Agent 6. Context only: they can
    # never modify a threshold, a cap, or a schema constraint.
    observations: tuple[str, ...] = ()

    def as_fields(self) -> dict[str, Any]:
        """Template fields. Numbers are pre-formatted so the prompt cannot
        depend on repr() drifting between Python versions."""
        return {
            "symbol": self.symbol,
            "spot": f"{self.spot:.2f}",
            "atr": f"{self.atr:.4f}",
            "atr_pct_of_spot": f"{self.atr_pct_of_spot:.4f}",
            "realized_vol": f"{self.realized_vol:.4f}",
            "rsi": f"{self.rsi:.2f}",
            "ema_fast_value": f"{self.ema_fast_value:.4f}",
            "ema_slow_value": f"{self.ema_slow_value:.4f}",
            "trend_pct_20d": f"{self.trend_pct_20d:.4f}",
            "above_vwap": "true" if self.above_vwap else "false",
            "observations": "\n".join(f"- {o}" for o in self.observations) or "(none)",
        }


@dataclass(frozen=True)
class ProfileResult:
    """A run, plus whether the session lock served it without a model call."""

    run: RunResult
    cached: bool = False

    @property
    def ok(self) -> bool:
        return self.run.ok

    @property
    def decision(self) -> RegimeDecision | None:
        return self.run.decision  # type: ignore[return-value]

    @property
    def blocks_action(self) -> bool:
        return self.run.blocks_action


class RegimeProfiler:
    """Agent 1, with the session lock. Construct once per session."""

    def __init__(
        self,
        config: Any,
        runner: AgentRunner,
        prompt_name: str = PROMPT_NAME,
        prompt_template_hash: str | None = None,
    ) -> None:
        self.config = config
        self.runner = runner
        self.prompt_name = prompt_name
        self.prompt_template_hash = prompt_template_hash
        self.locked = config.limits.get_bool("agents.a1.profile_locked_for_session")
        self._cache: dict[tuple[str, date], RunResult] = {}

    def cached_for(self, symbol: str, session: date) -> RunResult | None:
        return self._cache.get((symbol.upper(), session))

    def clear(self) -> None:
        """Drop the lock. For a new session, never mid-session."""
        self._cache.clear()

    def profile(
        self,
        inputs: RegimeInputs,
        session: date,
        transport: Callable[[str, str | None], Any],
        trace_id: str | None = None,
    ) -> ProfileResult:
        """Return the session's signal profile for one symbol.

        ``transport(prompt, feedback)`` performs the provider call. It is
        injected so the module is testable against stubs and provider-agnostic;
        the runner owns timeout, logging and validation around it.
        """
        key = (inputs.symbol.upper(), session)
        if self.locked and key in self._cache:
            log.debug("a1 profile for %s served from the session lock", inputs.symbol)
            return ProfileResult(self._cache[key], cached=True)

        prompt = load_and_render(self.config, self.prompt_name, inputs.as_fields())
        run = self.runner.run(
            "a1",
            AgentPath.ENTRY,
            prompt,
            lambda feedback: transport(prompt, feedback),
            symbol=inputs.symbol,
            trace_id=trace_id,
            prompt_template_hash=self.prompt_template_hash,
        )
        # Only an accepted decision locks. A failed call must not poison the
        # session -- the next cycle should be free to try again.
        if run.ok and self.locked:
            self._cache[key] = run
        return ProfileResult(run, cached=False)
