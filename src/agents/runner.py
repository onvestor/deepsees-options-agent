"""Generic agent call wrapper: timeout, logging, failure tracking, fallback.

The runner is the only place a model is actually invoked. Everything it does
is about what happens when the call goes wrong.

**Timeout behaviour depends on which path the call is on, and the runner has
to be told.** The two are not interchangeable:

* **Entry path** -- a timeout is a *skip*. No trade. There is no safe default
  entry, and a position opened on a guess is the one outcome this system is
  built to prevent.
* **Exit path** -- a timeout is logged and the session continues. This is safe
  *by construction*, not by optimism: the deterministic exits are always
  armed independently of any model, so Agent 5 failing to answer leaves the
  stop and target exactly where they were. Escalating here would be worse
  than continuing -- it would halt the loop that manages open risk.

Getting this backwards in either direction is dangerous, so ``path`` is a
required argument with no default.

**Prompts are never logged, only hashed.** This is already the log schema's
property, but the runner is where the temptation appears: the prompt is right
here in memory and putting it in the record would make debugging easy. It does
not go in. The record carries ``prompt_hash`` and ``prompt_chars``, and the
operator resolves the hash against their own gitignored ``prompts/``.

**The response is checked for restating the prompt.** A model that quotes its
instructions back turns ``response_raw`` into a prompt leak in a log that may
be published. Overlap is measured in word n-grams against the prompt that
produced it; above the configured share, the raw response is withheld and
replaced with a marker. Cheap, deterministic, and it closes the last path by
which prompt text can reach a committed artifact.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Deque

from pydantic import BaseModel

from src.agents.validator import (
    AGENT_LOG_NAMES,
    Override,
    ValidationOutcome,
    validate_with_retry,
)
from src.decisionlog.decision_log import prompt_hash
from src.decisionlog.schema import AgentCallPayload, AgentOverridePayload, ValidationResult

log = logging.getLogger(__name__)


class AgentPath(str, Enum):
    """Which decision path a call belongs to. Decides timeout semantics."""

    ENTRY = "entry"
    EXIT = "exit"
    REVIEW = "review"

    @property
    def timeout_is_fatal(self) -> bool:
        """True when a timeout must stop the action, not just be recorded.

        Only the entry path. Exit-path timeouts are covered by the always-armed
        deterministic exits; review runs offline and affects nothing today.
        """
        return self is AgentPath.ENTRY


# --- prompt-restatement backstop -------------------------------------------


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = text.lower().split()
    if len(words) < n:
        return set()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def prompt_overlap(response: str, prompt: str, n: int) -> float:
    """Share of the response's n-grams that also appear in the prompt.

    Word n-grams, not characters: a model restating instructions reproduces
    phrases, and an 8-word run matching verbatim is not coincidence. Short
    responses produce no n-grams and score 0.0, which is correct -- a
    three-word answer cannot leak a prompt.
    """
    response_grams = _ngrams(response, n)
    if not response_grams:
        return 0.0
    prompt_grams = _ngrams(prompt, n)
    if not prompt_grams:
        return 0.0
    return len(response_grams & prompt_grams) / len(response_grams)


@dataclass(frozen=True)
class ScrubResult:
    text: str | None
    truncated: bool
    overlap: float


def scrub_response(
    response: str | None, prompt: str, ngram_size: int, max_overlap: float
) -> ScrubResult:
    """Withhold a response that restates its own prompt.

    The marker records the measurement rather than the text, so the log still
    says *why* the response is missing and how close it was to the threshold.
    """
    if response is None:
        return ScrubResult(None, False, 0.0)
    overlap = prompt_overlap(response, prompt, ngram_size)
    if overlap <= max_overlap:
        return ScrubResult(response, False, overlap)
    return ScrubResult(
        f"[withheld: response reproduced {overlap:.0%} of the prompt's "
        f"{ngram_size}-word n-grams, above the {max_overlap:.0%} threshold. "
        f"Raw text not logged -- resolve prompt_hash against prompts/.]",
        True,
        overlap,
    )


# --- failure tracking for the kill switch ----------------------------------


@dataclass(frozen=True)
class FailureRate:
    calls: int
    failures: int
    rate: float
    halts_new_entries: bool
    reason: str = ""


class FailureTracker:
    """Rolling-window agent failure rate. Feeds the kill switch.

    A model failing repeatedly is a broken input to every decision downstream,
    so it halts new entries the same way a loss streak does.

    ``min_calls`` is not a nicety. One failure in one call is a 100% rate and
    means nothing; without a floor the switch fires on the session's first
    hiccup and halts a healthy system.
    """

    def __init__(self, window_minutes: int, rate_threshold: float, min_calls: int) -> None:
        self.window = timedelta(minutes=window_minutes)
        self.rate_threshold = rate_threshold
        self.min_calls = min_calls
        self._events: Deque[tuple[datetime, bool]] = deque()

    def record(self, failed: bool, at: datetime | None = None) -> None:
        moment = at or datetime.now(tz=timezone.utc)
        self._events.append((moment, failed))
        self._evict(moment)

    def _evict(self, now: datetime) -> None:
        cutoff = now - self.window
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def snapshot(self, at: datetime | None = None) -> FailureRate:
        now = at or datetime.now(tz=timezone.utc)
        self._evict(now)
        calls = len(self._events)
        failures = sum(1 for _, failed in self._events if failed)
        rate = failures / calls if calls else 0.0
        if calls < self.min_calls:
            return FailureRate(calls, failures, rate, False,
                               f"only {calls} call(s) in window; need {self.min_calls}")
        if rate > self.rate_threshold:
            return FailureRate(calls, failures, rate, True,
                               f"{failures}/{calls} failed ({rate:.0%}) above "
                               f"{self.rate_threshold:.0%} in the last "
                               f"{self.window.total_seconds() / 60:.0f} min")
        return FailureRate(calls, failures, rate, False, "")


# --- the runner ------------------------------------------------------------


@dataclass(frozen=True)
class RunResult:
    agent: str
    path: AgentPath
    outcome: ValidationOutcome | None
    timed_out: bool = False
    error: str | None = None
    latency_ms: float = 0.0
    overlap: float = 0.0
    response_withheld: bool = False

    @property
    def ok(self) -> bool:
        return self.outcome is not None and self.outcome.ok

    @property
    def decision(self) -> BaseModel | None:
        return self.outcome.decision if self.outcome else None

    @property
    def blocks_action(self) -> bool:
        """True when the caller must not act.

        On the entry path any failure blocks. On the exit path nothing blocks:
        the deterministic exits are already armed and the caller carries on
        with them.
        """
        return not self.ok and self.path.timeout_is_fatal


class AgentRunner:
    """Wraps every model call. Construct once per session."""

    def __init__(self, config: Any, decision_log: Any | None = None) -> None:
        limits = config.limits
        self.config = config
        self.log = decision_log
        self.model = limits.get_str("agent_runtime.model")
        self.timeout = limits.get_float("agent_runtime.timeout_seconds")
        self.log_full_response = limits.get_bool("agent_runtime.log_full_response")
        self.ngram_size = limits.get_int("agent_runtime.response_ngram_size")
        self.max_overlap = limits.get_float("agent_runtime.response_prompt_overlap_max")
        self.failures = FailureTracker(
            window_minutes=limits.get_int("killswitch.agent_failure_window_minutes"),
            rate_threshold=limits.get_float("killswitch.agent_failure_rate_pct"),
            min_calls=limits.get_int("killswitch.agent_failure_min_calls"),
        )
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="agent")

    # -- internals ----------------------------------------------------------

    def _call_with_timeout(self, call: Callable[[str | None], Any], feedback: str | None) -> Any:
        """Invoke with a wall-clock bound.

        The worker thread is not killed on timeout -- Python cannot -- so a
        hung provider call keeps its thread until the SDK's own socket timeout
        fires. What this guarantees is that *we* stop waiting, which is the
        part the trading loop needs.
        """
        future = self._pool.submit(call, feedback)
        try:
            return future.result(timeout=self.timeout)
        except FutureTimeout:
            future.cancel()
            raise TimeoutError(f"model call exceeded {self.timeout}s") from None

    def _emit(self, payload: Any, action: str, **kw: Any) -> None:
        if self.log is not None:
            self.log.write(payload, action=action, **kw)

    # -- public API ---------------------------------------------------------

    def run(
        self,
        agent_key: str,
        path: AgentPath,
        rendered_prompt: str,
        call: Callable[[str | None], Any],
        *,
        symbol: str | None = None,
        trace_id: str | None = None,
        prompt_template_hash: str | None = None,
        **context: Any,
    ) -> RunResult:
        """One agent call, validated, logged, and counted.

        ``call(feedback)`` performs the provider request and returns raw output;
        the validator may invoke it twice. ``rendered_prompt`` is used for its
        hash, its length, and the overlap check -- and for nothing else.
        """
        started = time.monotonic()
        raw_seen: list[Any] = []
        timed_out = False
        error: str | None = None
        outcome: ValidationOutcome | None = None
        collected: list[Override] = []

        def guarded(feedback: str | None) -> Any:
            raw = self._call_with_timeout(call, feedback)
            raw_seen.append(raw)
            if raw is None or (isinstance(raw, str) and not raw.strip()):
                # An empty response is a failure, not an empty object. Letting
                # it reach the parser would produce a schema error that reads
                # like malformed content rather than a silent provider.
                raise EmptyResponse("model returned an empty response")
            return raw

        try:
            outcome = validate_with_retry(
                agent_key, guarded, self.config.limits,
                on_override=lambda _a, o: collected.append(o),
                **context,
            )
        except TimeoutError as exc:
            timed_out, error = True, str(exc)
        except EmptyResponse as exc:
            error = str(exc)
        except Exception as exc:  # noqa: BLE001 -- provider errors are data here
            error = f"{type(exc).__name__}: {exc}"

        latency_ms = (time.monotonic() - started) * 1000.0
        failed = outcome is None or not outcome.ok
        self.failures.record(failed)

        raw_text = raw_seen[-1] if raw_seen else None
        if not isinstance(raw_text, str):
            raw_text = None if raw_text is None else repr(raw_text)
        scrub = scrub_response(raw_text, rendered_prompt, self.ngram_size, self.max_overlap)

        result = RunResult(
            agent=agent_key, path=path, outcome=outcome,
            timed_out=timed_out, error=error, latency_ms=latency_ms,
            overlap=scrub.overlap, response_withheld=scrub.truncated,
        )
        self._log_call(agent_key, path, rendered_prompt, prompt_template_hash,
                       outcome, scrub, timed_out, error, latency_ms, symbol, trace_id)
        for override in collected:
            self._emit(
                AgentOverridePayload(**override.as_payload_kwargs(agent_key)),
                action=f"agent_{override.kind.value}", symbol=symbol, trace_id=trace_id,
            )

        if failed:
            if path.timeout_is_fatal:
                log.warning("%s failed on the entry path (%s) -- skipping, no trade",
                            agent_key, error or "validation")
            else:
                log.warning("%s failed on the %s path (%s) -- continuing; "
                            "deterministic exits remain armed", agent_key, path.value,
                            error or "validation")
        return result

    def _log_call(self, agent_key, path, rendered_prompt, template_hash, outcome,
                  scrub, timed_out, error, latency_ms, symbol, trace_id) -> None:
        if timed_out:
            status = "timeout"
        elif outcome is None:
            status = "failed"
        else:
            status = outcome.status

        errors = list(outcome.errors) if outcome else []
        if error:
            errors.append(error)

        payload = AgentCallPayload(
            agent=AGENT_LOG_NAMES[agent_key],
            model=self.model,
            # The prompt itself never appears here. Only its fingerprint.
            prompt_hash=prompt_hash(rendered_prompt),
            prompt_template_hash=template_hash,
            prompt_chars=len(rendered_prompt),
            response_raw=scrub.text if self.log_full_response else None,
            response_parsed=(
                outcome.decision.model_dump(mode="json")
                if outcome and outcome.decision else None
            ),
            response_truncated=scrub.truncated,
            validation=ValidationResult(
                status=status,
                attempt=outcome.attempts if outcome else 1,
                errors=errors,
                clamps=[
                    {"kind": o.kind.value, "field": o.field,
                     "model_value": o.as_payload_kwargs(agent_key)["model_value"],
                     "applied_value": o.as_payload_kwargs(agent_key)["applied_value"],
                     "rule": o.rule}
                    for o in (outcome.overrides if outcome else ())
                ],
            ),
        )
        action = "skip" if (outcome is None or not outcome.ok) and path.timeout_is_fatal \
            else ("continue" if outcome is None or not outcome.ok else "accepted")
        self._emit(payload, action=action, symbol=symbol, trace_id=trace_id,
                   latency_ms=latency_ms)

    def failure_snapshot(self, at: datetime | None = None) -> FailureRate:
        """Current rolling failure rate, for the kill switch to consult."""
        return self.failures.snapshot(at)

    def close(self) -> None:
        self._pool.shutdown(wait=False)


class EmptyResponse(RuntimeError):
    """The provider returned nothing. Not a parse failure -- an absent answer."""
