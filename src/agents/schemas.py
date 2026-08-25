"""Pydantic contracts for all six agents. The boundary models, nothing else.

Three properties are structural here rather than enforced downstream, because
a runtime check can be bypassed and a type cannot:

* **Unknown fields are rejected.** ``extra="forbid"`` everywhere. A model that
  invents a field has not answered the question it was asked, and guessing
  which of its fields to honour is the opposite of failing closed.
* **Agent 3 can only shrink.** ``size_multiplier`` is bounded [0.0, 1.0] by the
  type. There is no value it can return that increases exposure.
* **Agent 5 cannot widen, add, or reverse.** Those actions are absent from the
  action enum, so they are not expressible. ``new_stop_pct`` is additionally
  bounded at or below zero -- a stop above the entry premium is not a stop.
  Whether a proposed stop is *tighter than the current one* needs runtime
  context and lives in :meth:`ExitDecision.tightens`, but the shape of the
  type already makes the dangerous half unrepresentable.

Threshold values are NOT here. Clamping to configured sets (allowed ema spans,
confidence floors, eligible-set truncation) is the validator's job -- this
module describes shape and hard bounds that hold regardless of tuning.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Every agent contract forbids unknown fields and rejects mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


# --- enums -----------------------------------------------------------------


class Regime(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGE_BOUND = "range_bound"
    CHOPPY = "choppy"
    GAP_FADE = "gap_fade"
    UNCLEAR = "unclear"


class Direction(str, Enum):
    LONG_CALLS = "long_calls"
    LONG_PUTS = "long_puts"
    BOTH = "both"
    NONE = "none"


class Bias(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class EventRisk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class IvAssessment(str, Enum):
    CHEAP = "cheap"
    FAIR = "fair"
    RICH = "rich"


class Structure(str, Enum):
    SINGLE_LEG = "single_leg"
    DEBIT_VERTICAL = "debit_vertical"


class ExitAction(str, Enum):
    """Deliberately incomplete.

    ``widen_stop``, ``add_to_position`` and ``reverse`` are absent by design,
    not by omission. Adding one to this enum would break the monotone-safety
    invariant, and any change here should be treated as a change to the risk
    model rather than to a schema.
    """

    HOLD = "hold"
    TIGHTEN_STOP = "tighten_stop"
    SCALE_OUT_HALF = "scale_out_half"
    EXIT_NOW = "exit_now"


# --- Agent 1: regime and signal profile ------------------------------------


class SignalProfile(StrictModel):
    ema_fast: int = Field(ge=1, le=200)
    confirmation_bars: int = Field(ge=0, le=10)
    require_vwap_alignment: bool
    min_atr_multiple: float = Field(ge=0.0, le=10.0)
    allowed_direction: Direction


class RegimeDecision(StrictModel):
    symbol: str = Field(min_length=1, max_length=16)
    regime: Regime
    confidence: float = Field(ge=0.0, le=1.0)
    signal_profile: SignalProfile
    rationale: str = Field(max_length=200)


# --- Agent 2: context and eligibility --------------------------------------


class ContextDecision(StrictModel):
    symbol: str = Field(min_length=1, max_length=16)
    eligible: bool
    hard_blocks: tuple[str, ...] = ()
    directional_bias: Bias
    bias_strength: float = Field(ge=0.0, le=1.0)
    event_risk: EventRisk
    iv_assessment: IvAssessment
    notes: str = Field(default="", max_length=300)

    @model_validator(mode="after")
    def _blocks_imply_ineligible(self) -> "ContextDecision":
        """A stated blocker and ``eligible: true`` is a contradiction.

        Resolved against the blocker, never against the flag: the model has
        named a reason not to trade, and honouring the flag would trade
        through it.
        """
        if self.hard_blocks and self.eligible:
            raise ValueError(
                f"hard_blocks {list(self.hard_blocks)} present but eligible=True "
                "-- a named blocker always wins"
            )
        return self


# --- Agent 3: risk allocation ----------------------------------------------


class RiskDecision(StrictModel):
    """Shrink or veto. There is no representable value above 1.0."""

    size_multiplier: float = Field(ge=0.0, le=1.0)
    reason: str = Field(max_length=200)


# --- Agent 4: contract and structure ---------------------------------------


class ContractDecision(StrictModel):
    structure: Structure
    primary_symbol: str = Field(min_length=1, max_length=32)
    short_symbol: str | None = Field(default=None, max_length=32)
    expected_hold_sessions: int = Field(ge=1, le=5)
    reason: str = Field(max_length=200)
    alternate_symbol: str | None = Field(default=None, max_length=32)

    @model_validator(mode="after")
    def _vertical_needs_two_legs(self) -> "ContractDecision":
        if self.structure is Structure.DEBIT_VERTICAL and not self.short_symbol:
            raise ValueError("debit_vertical requires a short_symbol")
        if self.structure is Structure.SINGLE_LEG and self.short_symbol:
            raise ValueError("single_leg must not carry a short_symbol")
        if self.short_symbol and self.short_symbol == self.primary_symbol:
            raise ValueError("short_symbol must differ from primary_symbol")
        return self


# --- Agent 5: exit management ----------------------------------------------


class ExitDecision(StrictModel):
    action: ExitAction
    new_stop_pct: float | None = Field(default=None, le=0.0, ge=-100.0)
    reason: str = Field(max_length=150)

    @model_validator(mode="after")
    def _stop_belongs_to_tighten(self) -> "ExitDecision":
        if self.action is ExitAction.TIGHTEN_STOP and self.new_stop_pct is None:
            raise ValueError("tighten_stop requires new_stop_pct")
        if self.action is not ExitAction.TIGHTEN_STOP and self.new_stop_pct is not None:
            raise ValueError(
                f"new_stop_pct is only meaningful with tighten_stop, got {self.action.value}"
            )
        return self

    def tightens(self, current_stop_pct: float) -> bool:
        """True only if this strictly tightens an existing stop.

        Runtime context the type cannot carry. Any action that is not
        ``tighten_stop`` returns False here, so a caller that gates on this
        can never widen a stop by accident.
        """
        if self.action is not ExitAction.TIGHTEN_STOP or self.new_stop_pct is None:
            return False
        return self.new_stop_pct > current_stop_pct


# --- Agent 6: nightly review -----------------------------------------------


class Observation(StrictModel):
    scope: str = Field(min_length=1, max_length=16)
    text: str = Field(max_length=200)
    expires_after_sessions: int = Field(ge=1, le=20)


class ReviewDecision(StrictModel):
    observations: tuple[Observation, ...] = ()


AGENT_SCHEMAS: dict[str, type[StrictModel]] = {
    "a1": RegimeDecision,
    "a2": ContextDecision,
    "a3": RiskDecision,
    "a4": ContractDecision,
    "a5": ExitDecision,
    "a6": ReviewDecision,
}
