"""The provider call. The only module in ``src/agents/`` that knows Anthropic exists.

Every agent takes a ``transport(prompt, feedback)`` callable and knows nothing
about who answers it. That indirection is what let all six agents be built and
tested against stubs before any provider existed, and it is worth keeping: a
wiring bug and a prompt bug look identical from outside, so the wiring was
proven first.

**This module does as little as possible, on purpose.** Timeout, validation,
the single retry, logging and the failure-rate counter all live in
:mod:`src.agents.runner`. A transport that also retried would multiply against
the runner's retry and quietly turn one skipped entry into four provider calls;
a transport that caught its own errors would hide them from the counter that
feeds the kill switch. So errors propagate untouched and the runner decides
what they mean.

**No prompt text lives here.** Not a system prompt, not an instruction, not an
example. That is the repository rule and this is exactly where it would be
convenient to break it. A system prompt, if the operator wants one, is a file
in the gitignored ``prompts/`` named at construction. The one string this
module does contain is a restatement of a validation error the validator
already produced -- mechanical, and carrying no strategy.

**No sampling parameters are sent.** ``temperature``, ``top_p`` and ``top_k``
were removed from the API on the current model generation (Fable 5, Opus 5,
Opus 4.8/4.7, Sonnet 5) and the 1.x SDK rejects them outright -- passing
``temperature`` raises ``TypeError: Messages.create() got an unexpected keyword
argument``, which surfaces as every agent failing validation at once. Determinism
now comes from the prompt and the schema, not from a sampling knob.

**``max_tokens`` must leave room for thinking.** Adaptive thinking is on by
default on the current models and its tokens count against ``max_tokens``. A
budget sized for the JSON alone truncates the JSON instead, and a truncated
object fails validation in a way that reads like a malformed model rather than
a budget that was too small -- which is why the truncation warning below exists.

**The SDK's own retries are disabled by default.** The runner bounds a call at
``agent_runtime.timeout_seconds`` in wall-clock, and the SDK's default retry
schedule can spend that budget without ever surfacing the first error -- the
call comes back as a timeout, which reads as a hung provider rather than the
429 it actually was. ``agent_runtime.sdk_max_retries`` makes that a configured
choice rather than a hidden one.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

log = logging.getLogger(__name__)

# Mechanical, not strategy: the validator produced these errors and the model
# needs them back to have a second attempt at the same question.
RETRY_PREFIX = (
    "Your previous response was rejected by schema validation with these "
    "errors:\n"
)
RETRY_SUFFIX = (
    "\nReturn a corrected JSON object. Same schema, same constraints, no prose "
    "and no code fence."
)


def build_retry_message(prompt: str, feedback: str) -> str:
    """The retry turn: the original question plus what was wrong with the answer.

    Stateless by choice. The transport is shared across agents and calls, and
    the alternative -- remembering the previous raw response to replay it as an
    assistant turn -- would make one agent's retry depend on another agent's
    last call under the runner's thread pool.
    """
    return f"{prompt}\n\n{RETRY_PREFIX}{feedback}{RETRY_SUFFIX}"


class AnthropicTransport:
    """A callable matching ``transport(prompt, feedback)``.

    Construct once per session and pass the instance to every agent::

        transport = AnthropicTransport(config)
        profiler.profile(inputs, session, transport)

    ``system_prompt_name``, when given, is resolved through
    :meth:`Config.prompt_path` -- so it is an operator file that a fresh clone
    does not have, and its absence raises a message naming it rather than
    silently sending no system prompt.
    """

    def __init__(
        self,
        config: Any,
        client: Any | None = None,
        system_prompt_name: str | None = None,
    ) -> None:
        limits = config.limits
        self.config = config
        self.model = limits.get_str("agent_runtime.model")
        self.max_tokens = limits.get_int("agent_runtime.max_output_tokens")
        self.effort = limits.get_str("agent_runtime.effort") or None
        self.timeout = limits.get_float("agent_runtime.timeout_seconds")
        self.max_retries = limits.get_int("agent_runtime.sdk_max_retries")
        self.system = self._load_system(system_prompt_name)
        self._client = client if client is not None else self._build_client()

    # -- construction -------------------------------------------------------

    def _load_system(self, name: str | None) -> str | None:
        if name is None:
            return None
        # Deliberately via prompt_path: an operator file, never a repo file.
        from src.agents.prompt_loader import load_template

        return load_template(self.config, name)

    def _build_client(self) -> Any:
        """Import the SDK at construction, not at import time.

        ``anthropic`` is a runtime dependency of trading, not of the test
        suite. Importing it at module scope would make every offline test that
        imports this module fail on a machine that has not installed it, which
        is the opposite of the fresh-clone property the rest of the repository
        maintains.
        """
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "the Anthropic SDK is not installed -- run 'pip install anthropic' "
                "(it is in requirements.txt). Agent calls cannot run without it."
            ) from exc

        return Anthropic(
            api_key=self.config.env.require_anthropic(),
            # Below the runner's wall-clock bound, so a stalled call surfaces as
            # a provider error with a cause rather than as a bare timeout -- and
            # so the worker thread is released instead of being held until the
            # SDK's own default fires. The runner cannot kill that thread.
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    # -- the call -----------------------------------------------------------

    def __call__(self, prompt: str, feedback: str | None = None) -> str:
        """One provider call. Returns raw text; raises on anything else.

        The return value is deliberately the unparsed string.
        ``validator.parse`` is the only thing that decides what a response
        means, and a transport that pre-parsed would be a second place where
        malformed output could be salvaged into a best guess.
        """
        content = build_retry_message(prompt, feedback) if feedback else prompt
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        if self.effort:
            # Thinking depth and overall token spend. These calls are small
            # structured extractions, so a low effort is usually right -- but
            # it is a configured choice, not one this module makes.
            kwargs["output_config"] = {"effort": self.effort}
        if self.system is not None:
            kwargs["system"] = self.system

        message = self._client.messages.create(**kwargs)
        text = extract_text(message)

        usage = getattr(message, "usage", None)
        if usage is not None:
            # Counts only. The prompt is not logged here, and neither is the
            # response -- the decision log owns that, with the restatement
            # backstop in front of it.
            log.debug(
                "anthropic call: model=%s in=%s out=%s stop=%s chars=%d",
                self.model,
                getattr(usage, "input_tokens", "?"),
                getattr(usage, "output_tokens", "?"),
                getattr(message, "stop_reason", "?"),
                len(text),
            )
        if getattr(message, "stop_reason", None) == "max_tokens":
            # Truncated JSON is unparseable, and the resulting error reads like
            # a malformed model rather than a budget that was too small.
            log.warning(
                "anthropic response hit max_tokens (%s) -- the JSON is truncated "
                "and will fail validation; raise agent_runtime.max_output_tokens",
                self.max_tokens,
            )
        return text


def extract_text(message: Any) -> str:
    """Concatenate the text blocks of a Messages response.

    Tolerant of blocks that are not text -- a response carrying anything else
    contributes nothing rather than raising here, because an empty string is
    already a failure the runner handles as an absent answer.
    """
    parts: list[str] = []
    for block in getattr(message, "content", ()) or ():
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", ""))
    return "".join(parts).strip()


def transport_from_config(
    config: Any, system_prompt_name: str | None = None
) -> Callable[[str, str | None], str]:
    """The callable every agent expects, wired to the configured provider."""
    return AnthropicTransport(config, system_prompt_name=system_prompt_name)
