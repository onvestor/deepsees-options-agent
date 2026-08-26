"""Agent 3 -- Risk Allocator.

**Base size is computed in code before the model is asked anything.** The
model returns a scalar in [0.0, 1.0] and the only thing it can do with it is
make the position smaller. That bound is a property of the schema type, not a
check here, so there is no value it could return -- and no bug in this module
-- that increases exposure.

The ordering is the safety property and it is not negotiable:

    base size (code)  ->  model multiplier (shrink only)  ->  hard caps (code)

Caps run **after** the model, so a multiplier of 1.0 cannot restore size that a
cap removed, and a cap can only ever reduce further. Both the requested and
the applied figures are recorded whenever a cap binds.

**A model failure is not a veto and not a full size.** Either default would be
a decision the model did not make: sizing at 1.0 on a timeout treats silence as
approval, and vetoing on a timeout means a broker hiccup silently stops
trading. This agent is on the ENTRY path, so a failure blocks the entry -- the
caller does not trade, and that is a different thing from trading at size zero
because it is visible as a skip rather than as a decision.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from src.agents.prompt_loader import load_and_render
from src.agents.runner import AgentPath, AgentRunner, RunResult
from src.agents.schemas import RiskDecision
from src.risk.sizing import AccountState, SizingLimits, SizingResult, compute_size

log = logging.getLogger(__name__)

PROMPT_NAME = "a3_risk.txt"


@dataclass(frozen=True)
class RiskInputs:
    """The deterministic size, and the context for judging whether to cut it."""

    symbol: str
    contract_symbol: str
    base_contracts: int
    cost_per_contract: float
    max_risk_per_contract: float
    risk_budget: float
    equity: float
    open_positions: int
    open_premium: float
    regime: str
    confidence: float
    bias_strength: float
    iv_assessment: str
    spans_earnings: bool | None = None
    observations: tuple[str, ...] = ()

    def as_fields(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "contract_symbol": self.contract_symbol,
            "base_contracts": str(self.base_contracts),
            "cost_per_contract": f"{self.cost_per_contract:.2f}",
            "max_risk_per_contract": f"{self.max_risk_per_contract:.2f}",
            "risk_budget": f"{self.risk_budget:.2f}",
            "equity": f"{self.equity:.2f}",
            "open_positions": str(self.open_positions),
            "open_premium": f"{self.open_premium:.2f}",
            "regime": self.regime,
            "confidence": f"{self.confidence:.2f}",
            "bias_strength": f"{self.bias_strength:.2f}",
            "iv_assessment": self.iv_assessment,
            "spans_earnings": (
                "unknown" if self.spans_earnings is None
                else ("true" if self.spans_earnings else "false")
            ),
            "observations": "\n".join(f"- {o}" for o in self.observations) or "(none)",
        }


@dataclass(frozen=True)
class AllocationResult:
    """The sized position, with every stage of how it got there."""

    sizing: SizingResult | None
    multiplier: float | None
    run: RunResult | None
    blocked: bool = False
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.sizing is not None and not self.blocked

    @property
    def contracts(self) -> int:
        """Contracts that may actually be traded. Zero whenever blocked.

        ``sizing`` still holds the pre-model base size for the log, but reading
        it as a tradeable quantity after a failed model call is exactly the
        mistake this agent exists to prevent -- so the number a caller reaches
        for first is the safe one, and the audit trail is the thing that has to
        be asked for explicitly.
        """
        if self.blocked or self.sizing is None:
            return 0
        return self.sizing.final_contracts

    @property
    def vetoed_by_model(self) -> bool:
        """The model deliberately sized to zero, as opposed to failing."""
        return self.multiplier == 0.0


class RiskAllocator:
    """Agent 3. Construct once per session."""

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
        self.limits = SizingLimits.from_limits(config.limits)

    def base_size(
        self, cost_per_contract: float, max_risk_per_contract: float, account: AccountState
    ) -> SizingResult:
        """The size before any model sees anything. Pure arithmetic."""
        return compute_size(
            cost_per_contract=cost_per_contract,
            max_risk_per_contract=max_risk_per_contract,
            account=account,
            limits=self.limits,
        )

    def allocate(
        self,
        inputs: RiskInputs,
        account: AccountState,
        transport: Callable[[str, str | None], Any],
        trace_id: str | None = None,
    ) -> AllocationResult:
        """Ask the model to shrink the deterministic size, then re-cap.

        The base size is recomputed here rather than trusted from ``inputs``:
        the inputs were rendered into a prompt and a model saw them, and the
        number that sizes the order must come from code that ran after that.
        """
        base = self.base_size(
            inputs.cost_per_contract, inputs.max_risk_per_contract, account
        )
        if base.final_contracts <= 0:
            # The caps already refuse this trade. Asking a model whether to
            # make zero smaller is a wasted call with no reachable outcome.
            log.info("a3: %s sized to zero before any model call (%s)",
                     inputs.symbol, base.rejected_reason or "caps")
            return AllocationResult(base, None, None, blocked=True,
                                    reason=base.rejected_reason or "capped to zero")

        prompt = load_and_render(self.config, self.prompt_name, inputs.as_fields())
        run = self.runner.run(
            "a3", AgentPath.ENTRY, prompt,
            lambda feedback: transport(prompt, feedback),
            symbol=inputs.symbol, trace_id=trace_id,
            prompt_template_hash=self.prompt_template_hash,
        )
        if not run.ok:
            # Neither 1.0 nor 0.0. Both would be a decision the model did not
            # make; the entry is skipped instead, which is visible as a skip.
            reason = "; ".join(run.outcome.errors) if run.outcome else (run.error or "failed")
            log.warning("a3: %s no usable multiplier (%s) -- skipping the entry",
                        inputs.symbol, reason)
            return AllocationResult(base, None, run, blocked=True, reason=reason)

        decision: RiskDecision = run.decision  # type: ignore[assignment]
        # Caps re-applied AFTER the multiplier. A multiplier of 1.0 cannot
        # restore anything a cap removed, and a cap may still cut further.
        final = compute_size(
            cost_per_contract=inputs.cost_per_contract,
            max_risk_per_contract=inputs.max_risk_per_contract,
            account=account,
            limits=self.limits,
            model_multiplier=decision.size_multiplier,
        )
        if final.final_contracts > base.final_contracts:  # pragma: no cover - invariant
            raise AssertionError(
                f"model multiplier increased size {base.final_contracts} -> "
                f"{final.final_contracts}; the monotone invariant is broken"
            )
        return AllocationResult(final, decision.size_multiplier, run)
