"""Agent 5 -- Exit Manager.

**The path semantics flip here, and that is the whole shape of the module.**
Agents 1-4 sit on the entry path, where any failure is a skip: no trade is the
safe answer because not trading is always available. Agent 5 manages positions
that are *already open*, and refusing to act is not a null outcome.

It is nonetheless safe to continue on failure, and safe **by construction**
rather than by optimism: the deterministic exits -- hard stop, profit target,
max hold, expiry-week flat -- are armed independently of any model and stay
armed whatever this agent does or fails to do. A silent Agent 5 leaves the
position exactly as protected as it was a minute earlier. Halting the loop on
a model failure would be strictly worse: it would stop the process that
manages open risk because the optional half of it broke.

So the model here is a *tightener*, never a guardian. Nothing it can fail to
say removes a protection.

**Every stop change is gated through ``ExitDecision.tightens()``.** The
schema already makes widening unrepresentable and the validator already forces
``hold`` when a proposed stop is not strictly tighter. This module gates
again anyway, on the value it is about to apply, because the alternative is a
single missing check between a model response and a live stop. A ``hold`` or
``scale_out_half`` carrying a ``new_stop_pct`` is *rejected* by the schema, not
quietly ignored -- silently dropping the field would let a model believe it had
moved a stop that never moved.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from src.agents.prompt_loader import load_and_render
from src.agents.runner import AgentPath, AgentRunner, RunResult
from src.agents.schemas import ExitAction, ExitDecision

log = logging.getLogger(__name__)

PROMPT_NAME = "a5_exit.txt"


@dataclass(frozen=True)
class ExitInputs:
    """One open position's state. Every number computed in code."""

    symbol: str
    contract_symbol: str
    entry_premium: float
    current_premium: float
    pnl_pct: float
    current_stop_pct: float
    target_pct: float
    sessions_held: int
    max_hold_sessions: int
    sessions_to_expiry: int
    contracts: int
    regime: str = "unknown"
    spans_earnings: bool | None = None
    observations: tuple[str, ...] = ()

    def as_fields(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "contract_symbol": self.contract_symbol,
            "entry_premium": f"{self.entry_premium:.2f}",
            "current_premium": f"{self.current_premium:.2f}",
            "pnl_pct": f"{self.pnl_pct:.2f}",
            "current_stop_pct": f"{self.current_stop_pct:.2f}",
            "target_pct": f"{self.target_pct:.2f}",
            "sessions_held": str(self.sessions_held),
            "max_hold_sessions": str(self.max_hold_sessions),
            "sessions_to_expiry": str(self.sessions_to_expiry),
            "contracts": str(self.contracts),
            "regime": self.regime,
            "spans_earnings": (
                "unknown" if self.spans_earnings is None
                else ("true" if self.spans_earnings else "false")
            ),
            "observations": "\n".join(f"- {o}" for o in self.observations) or "(none)",
        }


@dataclass(frozen=True)
class ExitPlan:
    """What to do with an open position, and what protection remains regardless."""

    action: ExitAction
    stop_pct: float
    stop_changed: bool = False
    run: RunResult | None = None
    model_failed: bool = False
    reason: str | None = None

    @property
    def blocks_action(self) -> bool:
        """Always False.

        There is no failure of this agent that should stop the exit loop. The
        deterministic exits do not depend on it.
        """
        return False

    @property
    def exits_now(self) -> bool:
        return self.action is ExitAction.EXIT_NOW

    @property
    def scales_out(self) -> bool:
        return self.action is ExitAction.SCALE_OUT_HALF


class ExitManager:
    """Agent 5. Construct once per session."""

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

    def manage(
        self,
        inputs: ExitInputs,
        transport: Callable[[str, str | None], Any],
        trace_id: str | None = None,
    ) -> ExitPlan:
        """Ask whether to tighten or leave. Never whether to loosen.

        On any failure the plan is ``hold`` at the **existing** stop. That is
        not a fallback guess -- it is the state the position was already in,
        and the deterministic exits are still armed on it.
        """
        prompt = load_and_render(self.config, self.prompt_name, inputs.as_fields())
        run = self.runner.run(
            "a5", AgentPath.EXIT, prompt,
            lambda feedback: transport(prompt, feedback),
            symbol=inputs.symbol, trace_id=trace_id,
            prompt_template_hash=self.prompt_template_hash,
            current_stop_pct=inputs.current_stop_pct,
        )

        if not run.ok:
            reason = "; ".join(run.outcome.errors) if run.outcome else (run.error or "failed")
            log.warning(
                "a5: %s no usable exit decision (%s) -- holding at the existing "
                "stop %.2f; deterministic exits remain armed",
                inputs.contract_symbol, reason, inputs.current_stop_pct,
            )
            return ExitPlan(ExitAction.HOLD, inputs.current_stop_pct,
                            run=run, model_failed=True, reason=reason)

        decision: ExitDecision = run.decision  # type: ignore[assignment]
        stop, changed = self._gated_stop(decision, inputs.current_stop_pct)
        return ExitPlan(decision.action, stop, stop_changed=changed,
                        run=run, reason=decision.reason)

    @staticmethod
    def _gated_stop(decision: ExitDecision, current: float) -> tuple[float, bool]:
        """The only path by which a stop value changes.

        ``tightens()`` returns False for every action that is not
        ``tighten_stop``, so a stop cannot move on a ``hold``, a
        ``scale_out_half`` or an ``exit_now`` even if some future change let
        one carry a value. The check is deliberately redundant with the schema
        and the validator: a stop is the last protection on an open position,
        and one missing check between a model response and a live stop is one
        too many.
        """
        if decision.tightens(current):
            return float(decision.new_stop_pct), True  # type: ignore[arg-type]
        return current, False
