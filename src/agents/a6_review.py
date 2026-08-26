"""Agent 6 -- Nightly Reviewer.

Turns a session's decision log into bounded observations for tomorrow's Agent 1
and Agent 2 prompts.

**Observations are context, never instruction.** An observation is a string
that gets rendered into a prompt. It cannot modify a threshold, a cap, or a
schema constraint, and that is a structural property rather than a rule this
module follows: :meth:`ObservationStore.live_for` returns ``tuple[str, ...]``
and there is no other way out of the store. Nothing here writes config, and
nothing downstream reads an observation as anything but prompt text. A model
that produces "raise max_contracts_per_trade to 12" produces a *sentence* --
one that will be shown to another model as something a previous session
noticed, and which no code path can act on.

**Count and expiry are capped in code, not requested of the model.** The
validator drops surplus observations, truncates over-long text, and clamps
expiry against ``agents.a6.*``. A reviewer that returns forty observations with
90-session lifetimes gets eight with capped lifetimes, and the truncation is
logged. The cap matters because these accumulate: an unbounded observation set
becomes a second, unversioned configuration file that nobody reviews.

**Path is REVIEW.** It runs after the close, affects nothing today, and its
failure costs tomorrow's context and nothing else. It never blocks.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Iterable

from src.agents.prompt_loader import load_and_render
from src.agents.runner import AgentPath, AgentRunner, RunResult
from src.agents.schemas import Observation, ReviewDecision

log = logging.getLogger(__name__)

PROMPT_NAME = "a6_review.txt"

GLOBAL_SCOPE = "global"


@dataclass(frozen=True)
class ReviewInputs:
    """A session's counts. Computed from the decision log, not from a model."""

    session: date
    entries: int
    exits: int
    skips: int
    wins: int
    losses: int
    realized_pnl: float
    agent_clamps: int
    agent_forces: int
    agent_failures: int
    fallbacks: int
    symbols_traded: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def as_fields(self) -> dict[str, Any]:
        return {
            "session": self.session.isoformat(),
            "entries": str(self.entries),
            "exits": str(self.exits),
            "skips": str(self.skips),
            "wins": str(self.wins),
            "losses": str(self.losses),
            "realized_pnl": f"{self.realized_pnl:.2f}",
            "agent_clamps": str(self.agent_clamps),
            "agent_forces": str(self.agent_forces),
            "agent_failures": str(self.agent_failures),
            "fallbacks": str(self.fallbacks),
            "symbols_traded": ", ".join(self.symbols_traded) or "(none)",
            "notes": "\n".join(f"- {n}" for n in self.notes) or "(none)",
        }


@dataclass(frozen=True)
class StoredObservation:
    """An observation plus the session it was written in."""

    text: str
    scope: str
    expires_after_sessions: int
    created_session: date

    def is_live(self, sessions_elapsed: int) -> bool:
        return 0 <= sessions_elapsed < self.expires_after_sessions


@dataclass
class ObservationStore:
    """Carries observations forward and ages them out.

    The only accessor returns strings. There is deliberately no method that
    hands out a number, a config key, or a structured object -- an observation
    can reach a prompt and nothing else.
    """

    items: list[StoredObservation] = field(default_factory=list)

    def add(self, observations: Iterable[Observation], session: date) -> None:
        for obs in observations:
            self.items.append(StoredObservation(
                text=obs.text, scope=obs.scope,
                expires_after_sessions=obs.expires_after_sessions,
                created_session=session,
            ))

    def prune(self, session: date, sessions_between: Callable[[date, date], int]) -> int:
        """Drop expired observations. Returns how many were removed."""
        before = len(self.items)
        self.items = [
            item for item in self.items
            if item.is_live(sessions_between(item.created_session, session))
        ]
        return before - len(self.items)

    def live_for(
        self,
        symbol: str,
        session: date,
        sessions_between: Callable[[date, date], int],
    ) -> tuple[str, ...]:
        """Observation **text** for one symbol, plus global ones. Strings only."""
        wanted = {symbol.upper(), GLOBAL_SCOPE}
        return tuple(
            item.text for item in self.items
            if item.scope.upper() in {w.upper() for w in wanted}
            and item.is_live(sessions_between(item.created_session, session))
        )


@dataclass(frozen=True)
class ReviewResult:
    observations: tuple[Observation, ...] = ()
    run: RunResult | None = None
    model_failed: bool = False
    reason: str | None = None

    @property
    def blocks_action(self) -> bool:
        """Always False. The review runs after the close and affects nothing today."""
        return False


class NightlyReviewer:
    """Agent 6. Runs after the close."""

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

    def review(
        self,
        inputs: ReviewInputs,
        transport: Callable[[str, str | None], Any],
        store: ObservationStore | None = None,
        trace_id: str | None = None,
    ) -> ReviewResult:
        """Produce tomorrow's observations. Failure costs context, nothing else."""
        prompt = load_and_render(self.config, self.prompt_name, inputs.as_fields())
        run = self.runner.run(
            "a6", AgentPath.REVIEW, prompt,
            lambda feedback: transport(prompt, feedback),
            trace_id=trace_id, prompt_template_hash=self.prompt_template_hash,
        )
        if not run.ok:
            reason = "; ".join(run.outcome.errors) if run.outcome else (run.error or "failed")
            log.warning("a6: review produced nothing usable (%s) -- tomorrow runs "
                        "without new observations", reason)
            return ReviewResult(run=run, model_failed=True, reason=reason)

        decision: ReviewDecision = run.decision  # type: ignore[assignment]
        if store is not None:
            store.add(decision.observations, inputs.session)
        return ReviewResult(decision.observations, run=run)
