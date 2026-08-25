"""Agent 2 -- Context & Eligibility.

Reads headlines and numeric context and decides tradeability. Two things about
its position in the pipeline shape the whole module.

**The earnings exclusion runs in code, before the model is called.** An
excluded symbol is never put in front of Agent 2 at all -- it is not passed
with a warning, not described as risky, not mentioned. The model cannot
override a rule it never sees, and that is the point: earnings proximity is a
mechanical test with a calendar answer, not a judgment, and the one failure
mode worth engineering against is a persuasive model talking its way past it.
Exclusions are recorded separately so the log still shows why a symbol was
absent.

**Truncation is not failure.** A model returning twelve eligible names against
a cap of ten has not answered badly; it has offered more than the system will
act on. The surplus is dropped by ``bias_strength`` rank after validation and
logged as an override with the full ranked list and the kept list side by side.
Treating it as a validation error would discard ten good answers over a
formatting-shaped disagreement.

**Path is ENTRY.** A failure for one symbol removes that symbol and nothing
else -- the screen is per-symbol, so one bad response does not cost the
session its whole eligible set. That is a deliberate difference from Agent 1,
where the profile is the session's and its loss is total.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Callable, Sequence

from src.agents.prompt_loader import load_and_render
from src.agents.runner import AgentPath, AgentRunner, RunResult
from src.agents.schemas import ContextDecision
from src.agents.validator import Override, truncate_eligible
from src.decisionlog.schema import AgentOverridePayload
from src.earnings.calendar import EarningsVerdict, evaluate_exclusion

log = logging.getLogger(__name__)

PROMPT_NAME = "a2_context.txt"


@dataclass(frozen=True)
class ContextInputs:
    """One candidate symbol's computed context. No model touched any of it."""

    symbol: str
    spot: float
    atr_pct_of_spot: float
    realized_vol: float
    iv_vs_rv20: float
    iv_percentile: float
    trend_pct_20d: float
    headlines: tuple[str, ...] = ()
    # Where the symbol sits in its earnings cycle. Supplied as context for the
    # iv_assessment judgment -- NOT as something the model may act on to admit
    # a symbol the code already excluded.
    sessions_until_earnings: int | None = None
    sessions_since_earnings: int | None = None
    observations: tuple[str, ...] = ()

    def as_fields(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "spot": f"{self.spot:.2f}",
            "atr_pct_of_spot": f"{self.atr_pct_of_spot:.4f}",
            "realized_vol": f"{self.realized_vol:.4f}",
            "iv_vs_rv20": f"{self.iv_vs_rv20:.4f}",
            "iv_percentile": f"{self.iv_percentile:.4f}",
            "trend_pct_20d": f"{self.trend_pct_20d:.4f}",
            "headlines": "\n".join(f"- {h}" for h in self.headlines) or "(none)",
            "sessions_until_earnings": (
                "unknown" if self.sessions_until_earnings is None
                else str(self.sessions_until_earnings)
            ),
            "sessions_since_earnings": (
                "unknown" if self.sessions_since_earnings is None
                else str(self.sessions_since_earnings)
            ),
            "observations": "\n".join(f"- {o}" for o in self.observations) or "(none)",
        }


@dataclass(frozen=True)
class ScreenResult:
    """The eligible set, plus everything that did not reach it and why."""

    eligible: tuple[ContextDecision, ...] = ()
    excluded_in_code: tuple[EarningsVerdict, ...] = ()
    ineligible: tuple[ContextDecision, ...] = ()
    failed: tuple[tuple[str, RunResult], ...] = ()
    overrides: tuple[Override, ...] = ()

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(d.symbol for d in self.eligible)

    @property
    def screened_symbols(self) -> tuple[str, ...]:
        """Symbols actually shown to the model. Excludes code-filtered ones."""
        return tuple(
            [d.symbol for d in self.eligible]
            + [d.symbol for d in self.ineligible]
            + [s for s, _ in self.failed]
        )


class ContextScreener:
    """Agent 2. Construct once per session."""

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

    # -- the code-side exclusion, before any model call ---------------------

    def earnings_exclusions(
        self,
        candidates: Sequence[ContextInputs],
        earnings: Any,
        trading_calendar: Any,
        order_session: date,
        now: datetime,
    ) -> tuple[tuple[ContextInputs, ...], tuple[EarningsVerdict, ...]]:
        """Split candidates into those the model may see and those it may not.

        Returns ``(admitted, excluded)``. Nothing in ``excluded`` is passed to
        a model in any form.
        """
        limits = self.config.limits
        no_earnings = {s.upper() for s in self.config.universe.no_earnings_symbols}
        admitted: list[ContextInputs] = []
        excluded: list[EarningsVerdict] = []

        for candidate in candidates:
            verdict = evaluate_exclusion(
                symbol=candidate.symbol,
                entry=earnings.get(candidate.symbol) if earnings else None,
                order_session=order_session,
                trading_calendar=trading_calendar,
                now=now,
                max_hold_sessions=limits.get_int("earnings.max_hold_sessions"),
                buffer_sessions=limits.get_int("earnings.buffer_sessions"),
                max_cache_age_hours=limits.get_float("earnings.max_cache_age_hours"),
                require_confirmed=limits.get_bool("earnings.require_confirmed"),
                no_earnings=candidate.symbol.upper() in no_earnings,
                post_print_buffer_sessions=limits.get_int(
                    "earnings.post_print_buffer_sessions"
                ),
            )
            if verdict.excluded:
                excluded.append(verdict)
            else:
                admitted.append(candidate)
        return tuple(admitted), tuple(excluded)

    # -- the model call, per symbol -----------------------------------------

    def screen(
        self,
        candidates: Sequence[ContextInputs],
        session: date,
        transport: Callable[[str, str | None], Any],
        earnings: Any = None,
        trading_calendar: Any = None,
        now: datetime | None = None,
        trace_id: str | None = None,
    ) -> ScreenResult:
        """Screen candidates and return the truncated eligible set.

        When ``earnings`` and ``trading_calendar`` are supplied the code-side
        exclusion runs first. They are optional only so the model path can be
        exercised on its own; a live session always supplies both, and
        ``assert_universe_resolves`` is what stops that being forgotten.
        """
        excluded: tuple[EarningsVerdict, ...] = ()
        if earnings is not None and trading_calendar is not None:
            candidates, excluded = self.earnings_exclusions(
                candidates, earnings, trading_calendar, session,
                now or datetime.now(),
            )
            for verdict in excluded:
                log.info("a2: %s excluded in code before any model call -- %s",
                         verdict.symbol, verdict.reason)

        eligible: list[ContextDecision] = []
        ineligible: list[ContextDecision] = []
        failed: list[tuple[str, RunResult]] = []

        for candidate in candidates:
            prompt = load_and_render(self.config, self.prompt_name, candidate.as_fields())
            run = self.runner.run(
                "a2", AgentPath.ENTRY, prompt,
                lambda feedback, p=prompt: transport(p, feedback),
                symbol=candidate.symbol, trace_id=trace_id,
                prompt_template_hash=self.prompt_template_hash,
            )
            if not run.ok:
                # Per-symbol failure. Costs this symbol and nothing else.
                failed.append((candidate.symbol, run))
                continue
            decision: ContextDecision = run.decision  # type: ignore[assignment]
            (eligible if decision.eligible else ineligible).append(decision)

        kept, overrides = truncate_eligible(eligible, self.config.limits)
        for override in overrides:
            log.info("a2: eligible set truncated -- %s", override.detail)
            self._log_override(override, trace_id)

        return ScreenResult(
            eligible=kept,
            excluded_in_code=excluded,
            ineligible=tuple(ineligible),
            failed=tuple(failed),
            overrides=tuple(overrides),
        )

    def _log_override(self, override: Override, trace_id: str | None) -> None:
        if self.runner.log is None:
            return
        self.runner.log.write(
            AgentOverridePayload(**override.as_payload_kwargs("a2")),
            action=f"agent_{override.kind.value}",
            trace_id=trace_id,
        )
