"""The decision log record shape.

This is the most important schema in the project. The decision log backs the
demo video, the write-up, and any judge question about why the agent did
something -- so the contract here is that **a dashboard or a report can be
generated from the log alone, with no reprocessing and no access to the code
that wrote it.**

Three properties follow from that:

1. **Every record is self-describing.** ``schema_version``, ``kind``,
   ``symbol``, ``action`` and ``reasons`` sit at the top level of every record
   regardless of type, so a consumer can group, filter and count without
   understanding any payload. ``jq 'select(.kind=="signal_eval")'`` works on
   day one and still works in December.
2. **Records are denormalised.** A record repeats the session date, the symbol
   and the trace id rather than pointing at some other record for them. Joins
   are the thing that turns "read the log" into "reprocess the log".
3. **Nothing secret is representable.** There is no field for prompt text and
   no field for a credential. Prompts are operator-supplied IP: the schema
   carries ``prompt_hash`` and ``prompt_chars``, and there is deliberately
   nowhere to put the prompt itself even by accident.

Deterministic decisions are first-class here, not an afterthought. A session
that took no trades must still explain itself: every signal evaluation with
each gate's verdict, every prefilter rejection, every cap that bound, every
kill switch that fired.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, computed_field

__all__ = [
    "SCHEMA_VERSION",
    "AgentCallPayload",
    "AgentOverridePayload",
    "CapOverridePayload",
    "DecisionRecord",
    "KillSwitchPayload",
    "OrderPayload",
    "Payload",
    "PrefilterPayload",
    "SessionPayload",
    "SignalEvalPayload",
    "SizingPayload",
    "ValidationResult",
    "new_trace_id",
]

# Bump only for a breaking change. Additive optional fields do not bump it --
# that is the whole point of versioning the reader against it.
SCHEMA_VERSION = 1

Kind = Literal[
    "session",
    "signal_eval",
    "prefilter",
    "agent_call",
    "sizing",
    "cap_override",
    "killswitch",
    "order",
]


def new_trace_id() -> str:
    """Correlates every decision belonging to one candidate trade."""
    return uuid.uuid4().hex[:16]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------


class SessionPayload(_Base):
    """Session lifecycle: open, close, halt, configuration snapshot."""

    kind: Literal["session"] = "session"
    event: Literal["open", "close", "halt", "resume", "replay_start", "replay_end"]
    equity: float | None = None
    open_positions: int | None = None
    config_fingerprint: str | None = Field(
        default=None,
        description="sha256 of the limits/universe values in force, so a replay can prove "
        "which thresholds produced these decisions without storing them.",
    )
    notes: str | None = None


class SignalEvalPayload(_Base):
    """One signal evaluation, with every gate's verdict -- pass or fail.

    Wired straight from ``SignalEvaluation.gates``. A suppressed signal is as
    much a decision as a triggered one and is logged identically.
    """

    kind: Literal["signal_eval"] = "signal_eval"
    bar_ts: str
    bar_count: int
    direction: Literal["long_calls", "long_puts", "none"]
    triggered: bool
    gates: dict[str, bool] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    profile: dict[str, Any] = Field(
        default_factory=dict,
        description="The signal profile in force: ema_fast, confirmation_bars, "
        "require_vwap_alignment, min_atr_multiple, allowed_direction.",
    )
    profile_source: Literal["agent", "default", "locked"] = "default"


class PrefilterPayload(_Base):
    """Deterministic chain prefilter outcome, multi-label.

    ``reason_counts`` sums above ``rejected`` because a contract can fail
    several tests. ``sole_reason`` is how many a reason rejected on its own --
    the number that actually matters when tuning a threshold.
    """

    kind: Literal["prefilter"] = "prefilter"
    underlying_price: float | None = None
    total_contracts: int
    survivors: int
    rejected: int
    reason_counts: dict[str, int] = Field(default_factory=dict)
    sole_reason: dict[str, int] = Field(default_factory=dict)
    survivor_symbols: list[str] = Field(default_factory=list)
    rejections: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Per-contract failing reasons. Populated when "
        "decision_log.prefilter_detail is 'full'; empty when 'aggregate'.",
    )
    thresholds: dict[str, float] = Field(
        default_factory=dict,
        description="The prefilter thresholds in force for this scan, so the log "
        "explains itself without config/limits.yaml, which is never committed.",
    )


class ValidationResult(_Base):
    """What the validator did to a model response."""

    status: Literal["ok", "clamped", "failed", "timeout", "refused"]
    attempt: int = 1
    errors: list[str] = Field(default_factory=list)
    clamps: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "One entry per override: {kind, field, model_value, applied_value, "
            "rule}. `kind` is 'clamp' (the model returned an out-of-range value) "
            "or 'force' (the value was valid and a rule overrode it). Both are "
            "overrides; only 'clamp' means the model answered badly. Each also "
            "appears as its own AgentOverridePayload record."
        ),
    )


class AgentCallPayload(_Base):
    """One agent invocation.

    **There is no prompt field and there never will be.** Prompts are
    operator-supplied IP that never enters the repository, so the log carries
    a hash and a length instead. ``prompt_hash`` identifies the exact rendered
    prompt; ``prompt_template_hash`` identifies the template it came from, so
    a change in wording is visible across sessions without the wording itself
    ever being recorded.
    """

    kind: Literal["agent_call"] = "agent_call"
    agent: Literal["a1_regime", "a2_context", "a3_risk", "a4_contract", "a5_exit", "a6_review"]
    model: str
    prompt_hash: str
    prompt_template_hash: str | None = None
    prompt_chars: int
    response_raw: str | None = Field(
        default=None,
        description="Full model response as returned, after credential scrubbing.",
    )
    response_parsed: dict[str, Any] | None = None
    response_truncated: bool = False
    validation: ValidationResult
    input_tokens: int | None = None
    output_tokens: int | None = None
    fallback_used: bool = False


class AgentOverridePayload(_Base):
    """One override applied to a model response. One record each.

    The log must answer "what did the model say" separately from "what did the
    system do", so both values are always present and neither is inferable
    from the other.

    ``override`` distinguishes two things that are easy to conflate:

    * ``clamp`` -- the model returned a value outside an allowed range or set,
      and it was moved to the nearest permitted one. This is evidence the model
      answered badly.
    * ``force`` -- the model returned a perfectly valid value and a rule
      overrode it anyway (confidence below the floor, a regime on the forced
      list). This says nothing about the model's output quality; it is the
      system asserting policy.

    Counting these together would make a well-behaved model on a choppy day
    look identical to one emitting garbage.
    """

    kind: Literal["agent_override"] = "agent_override"
    agent: Literal["a1_regime", "a2_context", "a3_risk", "a4_contract", "a5_exit", "a6_review"]
    override: Literal["clamp", "force"]
    field: str
    model_value: Any = None
    applied_value: Any = None
    rule: str = Field(description="Config key or invariant that caused the override.")
    detail: str = ""


class SizingPayload(_Base):
    """Base size computed in code, before any model sees it."""

    kind: Literal["sizing"] = "sizing"
    sizing_capital: float
    capital_source: Literal["options_buying_power", "equity"]
    risk_per_trade: float
    premium_per_contract: float
    base_contracts: int
    model_multiplier: float | None = Field(
        default=None, description="Agent 3's scalar. Clamped to [0,1]: shrink or veto only."
    )
    final_contracts: int


class CapOverridePayload(_Base):
    """A hard cap bound. Caps always win; both values are recorded."""

    kind: Literal["cap_override"] = "cap_override"
    cap_name: str
    requested: float
    cap_value: float
    applied: float
    stage: Literal["sizing", "entry", "exit", "portfolio"]


class KillSwitchPayload(_Base):
    """A deterministic halt. Never consults a model."""

    kind: Literal["killswitch"] = "killswitch"
    switch: str
    threshold: float
    observed: float
    fired: bool
    halts_new_entries: bool = True
    scope: Literal["symbol", "session", "account"] = "session"


class OrderPayload(_Base):
    """An order actually sent to the broker, and what came back."""

    kind: Literal["order"] = "order"
    intent: Literal["buy_to_open", "sell_to_close", "cancel"]
    structure: Literal["single_leg", "debit_vertical"] = "single_leg"
    legs: list[str] = Field(default_factory=list)
    qty: int
    limit_price: float | None = None
    order_id: str | None = None
    status: str | None = None
    filled_qty: float | None = None
    filled_avg_price: float | None = None
    session_dte: int | None = None
    broker_error: str | None = None


class SkipPayload(_Base):
    """A stage that declined to act, and why.

    The other payloads all describe something that happened -- a chain scored,
    a size computed, an order sent. This one exists because **a skip is a
    decision**, and the sessions where the log matters most are the ones that
    traded nothing. "Why was there no trade all morning" is answerable only if
    every stage that declined said so with the same weight as a stage that
    acted.

    ``stage`` is the pipeline step, not the agent: an entry can die at the
    signal, the prefilter, the selector or the sizer, and which one it was is
    the first thing a reader needs.
    """

    kind: Literal["skip"] = "skip"
    stage: str
    reason: str
    detail: dict[str, Any] = Field(default_factory=dict)


Payload = Annotated[
    Union[
        SessionPayload,
        SignalEvalPayload,
        PrefilterPayload,
        AgentCallPayload,
        AgentOverridePayload,
        SizingPayload,
        CapOverridePayload,
        KillSwitchPayload,
        OrderPayload,
        SkipPayload,
    ],
    Field(discriminator="kind"),
]


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------


class DecisionRecord(BaseModel):
    """One decision. The unit the whole log is made of.

    ``kind`` is computed from the payload rather than stored twice, so the two
    can never drift -- but it is serialised at the top level, because a
    dashboard should be able to filter on it without parsing the payload.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = SCHEMA_VERSION
    record_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    seq: int
    ts_utc: str
    ts_et: str
    session_date: str
    trace_id: str | None = None
    symbol: str | None = None
    action: str = Field(
        description="What actually happened, in the past tense: the decision's effect. "
        "'no_trade', 'signal_suppressed', 'order_filled', 'size_reduced', 'halted'."
    )
    reasons: list[str] = Field(default_factory=list)
    latency_ms: float | None = None
    payload: Payload

    @computed_field  # type: ignore[prop-decorator]
    @property
    def kind(self) -> Kind:
        return self.payload.kind

    @staticmethod
    def utc_now() -> datetime:
        return datetime.now(tz=timezone.utc)
