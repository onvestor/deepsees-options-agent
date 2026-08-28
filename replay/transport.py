"""Transports for offline replay: canned, scripted, recorded, and replayed.

Every agent takes ``transport(prompt, feedback)`` and does not care who answers.
Four answers are useful offline, and they are different tools:

* :func:`canned` -- one fixed response. For exercising a path.
* :func:`scripted` -- a function of the rendered prompt. For deterministic
  policies: "always eligible", "veto anything above 0.6 bias".
* :class:`RecordingTransport` -- wraps a real transport and writes every
  response to a JSONL file, keyed by prompt hash.
* :class:`RecordedTransport` -- replays that file with no network at all.

**Record once, replay many is the point.** Prompt iteration against a live
provider costs money and a market session; against a recording it costs
nothing and is reproducible, which is the only way to compare two prompts
honestly. The recording is keyed by the hash of the *rendered* prompt, so
editing a prompt template is a cache miss rather than a stale hit -- the
harness says which agent missed and the recording is refreshed. Silently
serving the old answer for a new prompt would be the worst possible failure
here, because the replay would look like it was testing the new prompt.

**Recordings hold prompt hashes, never prompt text.** ``prompts/`` is the IP
and a recording file is exactly the artifact that would leak it. Only the hash,
the agent key and the model's response are written, which is the same rule the
decision log already follows.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from src.decisionlog.decision_log import prompt_hash

log = logging.getLogger(__name__)

Transport = Callable[[str, "str | None"], Any]


class RecordingMiss(RuntimeError):
    """A prompt with no recorded response, naming what to re-record."""


# --- the simple ones -------------------------------------------------------


def canned(response: Any) -> Transport:
    """Always return ``response``. The response may be a dict or a JSON string."""

    def call(prompt: str, feedback: str | None = None) -> Any:
        return response

    return call


def scripted(fn: Callable[[str, "str | None"], Any]) -> Transport:
    """A deterministic policy over the rendered prompt.

    Useful for driving a replay without a model at all: a rule that reads the
    numbers out of the prompt and answers consistently isolates the pipeline
    from model variance, which is what you want when the thing under test is
    the pipeline.
    """
    return fn


def sequence(responses: Iterable[Any], loop: bool = False) -> Transport:
    """Return each response in turn.

    Exhausting the sequence raises rather than repeating the last response: a
    replay that quietly reused one answer for every remaining session would
    produce a plausible-looking result from nothing.
    """
    items = list(responses)
    state: dict[str, int] = {"i": 0}

    def call(prompt: str, feedback: str | None = None) -> Any:
        i = state["i"]
        if i >= len(items):
            if not loop:
                raise RecordingMiss(
                    f"scripted sequence exhausted after {len(items)} response(s)"
                )
            i = 0
        state["i"] = i + 1
        return items[i]

    return call


# --- record and replay -----------------------------------------------------


@dataclass(frozen=True)
class RecordedCall:
    agent: str
    prompt_hash: str
    retry: bool
    response: Any

    def as_json(self) -> str:
        return json.dumps(
            {
                "agent": self.agent,
                "prompt_hash": self.prompt_hash,
                "retry": self.retry,
                "response": self.response,
            },
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, line: str) -> "RecordedCall":
        raw = json.loads(line)
        return cls(
            agent=raw.get("agent", "?"),
            prompt_hash=raw["prompt_hash"],
            retry=bool(raw.get("retry", False)),
            response=raw["response"],
        )


def _key(prompt: str, feedback: str | None) -> tuple[str, bool]:
    """A recorded call is identified by its prompt and whether it is the retry.

    The retry flag is part of the key because the retry sends a different
    question -- the original plus the validation error -- and answering it with
    the first attempt's response would replay a failure as a success.
    """
    return prompt_hash(prompt), feedback is not None


@dataclass
class RecordingTransport:
    """Wraps a transport and writes every response to a JSONL file.

    Appends as it goes rather than buffering to the end, so a run interrupted
    halfway still leaves a usable partial recording -- which is the common case
    when recording against a live session.
    """

    inner: Transport
    path: Path
    agent: str = "?"
    _seen: set[tuple[str, bool]] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, prompt: str, feedback: str | None = None) -> Any:
        response = self.inner(prompt, feedback)
        digest, retry = _key(prompt, feedback)
        record = RecordedCall(self.agent, digest, retry, response)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(record.as_json() + "\n")
        self._seen.add((digest, retry))
        return response

    @property
    def recorded(self) -> int:
        return len(self._seen)


@dataclass
class RecordedTransport:
    """Replays a recording. No network, no provider, no key required.

    ``on_miss`` decides what an unrecorded prompt means. The default raises,
    because a miss is nearly always an edited prompt and continuing would test
    something other than what was asked for. Passing a fallback response is the
    escape hatch for a replay that is deliberately exploring beyond the
    recording.
    """

    calls: dict[tuple[str, bool], Any]
    agent: str = "?"
    on_miss: Any | None = None
    raise_on_miss: bool = True
    misses: int = 0

    @classmethod
    def from_file(cls, path: Path, agent: str | None = None, **kw) -> "RecordedTransport":
        path = Path(path)
        if not path.is_file():
            raise RecordingMiss(
                f"no recording at {path} -- record one first with RecordingTransport"
            )
        calls: dict[tuple[str, bool], Any] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = RecordedCall.from_json(line)
            if agent is not None and record.agent not in (agent, "?"):
                continue
            # Last write wins: a re-recorded prompt should supersede its
            # earlier answer rather than being shadowed by it.
            calls[(record.prompt_hash, record.retry)] = record.response
        return cls(calls=calls, agent=agent or "?", **kw)

    def __call__(self, prompt: str, feedback: str | None = None) -> Any:
        key = _key(prompt, feedback)
        if key in self.calls:
            return self.calls[key]

        self.misses += 1
        if self.raise_on_miss:
            raise RecordingMiss(
                f"{self.agent}: no recorded response for prompt {key[0][:12]} "
                f"(retry={key[1]}). The prompt has changed since the recording "
                "was made -- re-record it rather than replaying a stale answer."
            )
        log.warning(
            "%s: recording miss for prompt %s, using the fallback response",
            self.agent, key[0][:12],
        )
        return self.on_miss

    def __len__(self) -> int:
        return len(self.calls)


def split_by_agent(path: Path) -> dict[str, RecordedTransport]:
    """One RecordedTransport per agent from a single recording file.

    The harness wires a transport per agent, so a recording made from a whole
    session splits back into the same shape it was recorded in.
    """
    path = Path(path)
    agents: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            agents.add(RecordedCall.from_json(line).agent)
    return {agent: RecordedTransport.from_file(path, agent=agent) for agent in sorted(agents)}


def iter_recording(path: Path) -> Iterator[RecordedCall]:
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield RecordedCall.from_json(line)
