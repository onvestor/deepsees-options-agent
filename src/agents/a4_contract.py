"""Agent 4 -- Contract & Structure Selector.

Chooses among a deterministic survivor set. **The model narrows; it never
widens.** The set it sees was produced by the prefilter, ranked by modelled
P&L per unit of spread cost, and capped at
``prefilter.max_survivors_per_symbol``. A symbol outside that set is a
validation failure -- not a request to go and fetch it. There is no code path
by which a model names a contract into existence, and the survivor check in
the validator is what enforces it.

The cap exists because more candidates makes a model's decision worse, not
better. The **full** survivor set is still logged, so nothing is lost to later
analysis: the log records how many survived and which twelve were offered.

**The fallback must read differently from a success.** When the model's answer
cannot be honoured, the deterministic best-ratio survivor is taken as a single
leg. That is a working outcome and the session continues -- but the model's
choice was *not* honoured, and if that looked identical in the log to the model
picking that same contract itself, there would be no way to measure how often
the model is being overridden. So a fallback emits its own record, carries
``fallback_used`` on the call payload, and reports ``source="fallback"`` on the
result. A model that happens to choose the top-ranked survivor is still
``source="model"`` -- it chose, and that is the thing being measured.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

from src.agents.prompt_loader import load_and_render
from src.agents.runner import AgentPath, AgentRunner, RunResult
from src.agents.schemas import ContractDecision, Structure
from src.decisionlog.schema import AgentOverridePayload

log = logging.getLogger(__name__)

PROMPT_NAME = "a4_contract.txt"

SelectionSource = Literal["model", "fallback", "none"]


@dataclass(frozen=True)
class ContractInputs:
    """The survivor set plus the context needed to choose within it."""

    symbol: str
    spot: float
    atr: float
    # Prefilter survivors, already ranked. Full set: the cap is applied here so
    # the module controls what the model sees and what the log records.
    survivors: tuple[Any, ...]
    regime: str
    confidence: float
    directional_bias: str
    bias_strength: float
    iv_assessment: str
    target_expiry: str
    session_dte: int
    spans_earnings: bool | None = None
    observations: tuple[str, ...] = ()

    def offered(self, cap: int) -> tuple[Any, ...]:
        return tuple(self.survivors[:cap])

    def survivor_table(self, cap: int) -> str:
        """One line per offered contract. Numbers only -- no recommendations."""
        rows = []
        for candidate in self.offered(cap):
            metrics = getattr(candidate, "metrics", None)
            quote = getattr(candidate, "quote", None)
            spec = getattr(candidate, "spec", candidate)
            rows.append(
                f"- {getattr(spec, 'symbol', '?')} "
                f"strike {getattr(spec, 'strike', float('nan')):.2f} "
                f"expiry {getattr(spec, 'expiry', '?')} "
                f"delta {getattr(quote, 'delta', float('nan')):.4f} "
                f"mid {getattr(quote, 'mid', float('nan')):.2f} "
                f"spread_pct {getattr(quote, 'spread_pct_of_mid', float('nan')):.4f} "
                f"oi {getattr(spec, 'open_interest', 0)} "
                f"pnl_to_spread {getattr(metrics, 'pnl_to_spread_ratio', float('nan')):.3f} "
                f"expiry_type {getattr(candidate, 'expiry_type', '?')}"
            )
        return "\n".join(rows) or "(none)"

    def as_fields(self, cap: int) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "spot": f"{self.spot:.2f}",
            "atr": f"{self.atr:.4f}",
            "regime": self.regime,
            "confidence": f"{self.confidence:.2f}",
            "directional_bias": self.directional_bias,
            "bias_strength": f"{self.bias_strength:.2f}",
            "iv_assessment": self.iv_assessment,
            "target_expiry": self.target_expiry,
            "session_dte": str(self.session_dte),
            "spans_earnings": (
                "unknown" if self.spans_earnings is None
                else ("true" if self.spans_earnings else "false")
            ),
            "survivors": self.survivor_table(cap),
            "survivor_count": str(len(self.offered(cap))),
            "observations": "\n".join(f"- {o}" for o in self.observations) or "(none)",
        }


@dataclass(frozen=True)
class SelectionResult:
    decision: ContractDecision | None
    source: SelectionSource
    run: RunResult | None = None
    offered: tuple[str, ...] = ()
    survivors_total: int = 0
    fallback_reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.decision is not None

    @property
    def used_fallback(self) -> bool:
        return self.source == "fallback"


class ContractSelector:
    """Agent 4. Construct once per session."""

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
        self.cap = config.limits.get_int("prefilter.max_survivors_per_symbol")
        self.fallback_hold = config.limits.get_int("metrics.modeled_hold_sessions")

    def select(
        self,
        inputs: ContractInputs,
        transport: Callable[[str, str | None], Any],
        trace_id: str | None = None,
    ) -> SelectionResult:
        offered = inputs.offered(self.cap)
        if not offered:
            # Nothing survived. Not a model failure and not a fallback -- there
            # is no contract to choose, and inventing one is the whole thing
            # this agent is forbidden to do.
            log.info("a4: %s has no survivors; nothing to select", inputs.symbol)
            return SelectionResult(None, "none", survivors_total=len(inputs.survivors))

        offered_symbols = tuple(getattr(getattr(c, "spec", c), "symbol") for c in offered)
        if len(inputs.survivors) > len(offered):
            log.info("a4: %s %d survivors, top %d offered (ranked by modelled "
                     "P&L per unit of spread cost); full set retained in the log",
                     inputs.symbol, len(inputs.survivors), len(offered))

        prompt = load_and_render(self.config, self.prompt_name, inputs.as_fields(self.cap))
        run = self.runner.run(
            "a4", AgentPath.ENTRY, prompt,
            lambda feedback: transport(prompt, feedback),
            symbol=inputs.symbol, trace_id=trace_id,
            prompt_template_hash=self.prompt_template_hash,
            # The model is held to exactly what it was shown, not to the full
            # survivor set -- it cannot pick something it never saw either.
            survivors=list(offered_symbols),
        )

        if run.ok:
            return SelectionResult(
                run.decision, "model", run=run, offered=offered_symbols,  # type: ignore[arg-type]
                survivors_total=len(inputs.survivors),
            )

        reason = "; ".join(run.outcome.errors) if run.outcome else (run.error or "no response")
        decision = self._deterministic_choice(offered[0])
        self._log_fallback(inputs.symbol, decision, reason, trace_id)
        log.warning("a4: %s falling back to the best-ratio survivor %s -- %s",
                    inputs.symbol, decision.primary_symbol, reason)
        return SelectionResult(
            decision, "fallback", run=run, offered=offered_symbols,
            survivors_total=len(inputs.survivors), fallback_reason=reason,
        )

    def _deterministic_choice(self, best: Any) -> ContractDecision:
        """Best-ratio survivor, single leg.

        Single leg on purpose: a vertical is a judgment about whether four
        bid-ask crossings will be earned over the hold, and the fallback path
        exists precisely because no judgment is available.

        The hold comes from ``metrics.modeled_hold_sessions`` -- the same value
        the ranking used to decide which survivor is best. Declaring a
        different hold than the one the ranking assumed would make the choice
        and its justification disagree.
        """
        spec = getattr(best, "spec", best)
        return ContractDecision(
            structure=Structure.SINGLE_LEG,
            primary_symbol=spec.symbol,
            expected_hold_sessions=self.fallback_hold,
            reason="deterministic fallback: best modelled P&L per unit of spread cost",
        )

    def _log_fallback(
        self, symbol: str, decision: ContractDecision, reason: str, trace_id: str | None
    ) -> None:
        """Its own record, so overrides can be counted separately from choices."""
        if self.runner.log is None:
            return
        self.runner.log.write(
            AgentOverridePayload(
                agent="a4_contract",
                override="force",
                field="decision",
                model_value=None,
                applied_value=decision.primary_symbol,
                rule="a4_deterministic_fallback",
                detail=f"model choice not honoured ({reason})",
            ),
            action="agent_fallback",
            symbol=symbol,
            trace_id=trace_id,
        )
