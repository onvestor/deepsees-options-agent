"""Deterministic stand-ins for the six agents. No model, no network, no key.

**Why a replay wants a mode with no model in it at all.** When a replay
produces a surprising number, the first question is whether the pipeline did
something wrong or the model made an unusual judgment. Those are different
bugs with different fixes, and separating them after the fact is guesswork. A
run with the model replaced by a fixed policy answers the first question on its
own: whatever it does is the pipeline, because nothing else varies.

These are *not* a baseline strategy and no conclusion about the strategy should
be drawn from their P&L. They say yes to nearly everything, which is what makes
them useful for exercising the path -- and useless as a measure of anything.

They still render real prompts. Every agent loads its template and substitutes
its fields before the transport is reached, so this mode needs ``prompts/`` like
any other; what it does not need is an API key or a network. The symbol and the
contract are read back out of the rendered prompt, which is the only thing a
transport is given.
"""
from __future__ import annotations

import re
from typing import Any, Sequence

from replay.transport import Transport

# The OCC symbols the survivor table renders. Unpadded root, as Alpaca returns.
OCC_PATTERN = re.compile(r"\b([A-Z][A-Z0-9]{0,5}\d{6}[CP]\d{8})\b")

# A bare ticker. Bounded so it cannot match a word inside prose.
TICKER_PATTERN = re.compile(r"\b([A-Z]{1,6})\b")


class RuleError(RuntimeError):
    """A rule stub that cannot answer, with the reason named.

    Raised rather than guessed. A stub that invented a symbol would put a
    contract the prefilter never offered in front of the validator, and the
    resulting failure would look like a model error.
    """


def symbol_in(prompt: str, known: Sequence[str] | None = None) -> str:
    """The underlying this prompt is about.

    With ``known`` supplied the search is exact, which is what a replay does --
    it knows its own universe. Without it, the first plausible ticker is taken,
    which is good enough for a one-off and stated as such.
    """
    if known:
        for symbol in known:
            if re.search(rf"\b{re.escape(symbol.upper())}\b", prompt):
                return symbol.upper()
        raise RuleError(
            f"no symbol from {list(known)} appears in the rendered prompt"
        )
    match = TICKER_PATTERN.search(prompt)
    if not match:
        raise RuleError("no ticker found in the rendered prompt")
    return match.group(1)


def survivors_in(prompt: str) -> list[str]:
    """Every OCC symbol offered in the prompt, in the order they appear.

    Order matters: the prefilter ranks the survivor table, so the first match
    is the top-ranked candidate.
    """
    return list(dict.fromkeys(OCC_PATTERN.findall(prompt)))


# --- the six ---------------------------------------------------------------


def a1_rule(symbols: Sequence[str] | None = None) -> Transport:
    """Always a clean uptrend that permits long calls.

    ``require_vwap_alignment`` is False and ``min_atr_multiple`` sits at the
    permissive end on purpose: this stub exists to let signals through so the
    stages after it get exercised. A realistic profile would be a better
    strategy and a worse test.
    """

    def call(prompt: str, feedback: str | None = None) -> dict[str, Any]:
        return {
            "symbol": symbol_in(prompt, symbols),
            "regime": "trending_up",
            "confidence": 0.75,
            "signal_profile": {
                "ema_fast": 9,
                "confirmation_bars": 1,
                "require_vwap_alignment": False,
                "min_atr_multiple": 0.3,
                "allowed_direction": "long_calls",
            },
            "rationale": "replay rule stub: fixed trending_up profile",
        }

    return call


def a2_rule(symbols: Sequence[str] | None = None) -> Transport:
    """Everything is eligible, with a bias strong enough to clear the floor."""

    def call(prompt: str, feedback: str | None = None) -> dict[str, Any]:
        return {
            "symbol": symbol_in(prompt, symbols),
            "eligible": True,
            "hard_blocks": [],
            "directional_bias": "bullish",
            "bias_strength": 0.7,
            "event_risk": "low",
            "iv_assessment": "fair",
            "notes": "replay rule stub",
        }

    return call


def a3_rule(multiplier: float = 1.0) -> Transport:
    """Never shrink. The caps still run after this and can still cut."""

    def call(prompt: str, feedback: str | None = None) -> dict[str, Any]:
        return {
            "size_multiplier": multiplier,
            "reason": "replay rule stub: no reduction",
        }

    return call


def a4_rule(hold_sessions: int = 3) -> Transport:
    """Take the top-ranked survivor as a single leg.

    Deliberately the same choice the deterministic fallback would make. That
    makes a rules run a clean control: any difference between it and a model
    run is the model's contract selection and nothing else.
    """

    def call(prompt: str, feedback: str | None = None) -> dict[str, Any]:
        offered = survivors_in(prompt)
        if not offered:
            raise RuleError("no OCC symbols in the rendered prompt to choose from")
        return {
            "structure": "single_leg",
            "primary_symbol": offered[0],
            "short_symbol": None,
            "expected_hold_sessions": hold_sessions,
            "reason": "replay rule stub: top-ranked survivor",
            "alternate_symbol": offered[1] if len(offered) > 1 else None,
        }

    return call


def a5_rule() -> Transport:
    """Always hold. The deterministic exits do all the work.

    A stub that tightened would make every replay exit look like a stop, and
    the stop it hit would be one this file chose rather than one the config
    did.
    """

    def call(prompt: str, feedback: str | None = None) -> dict[str, Any]:
        return {
            "action": "hold",
            "new_stop_pct": None,
            "reason": "replay rule stub: no change",
        }

    return call


def a6_rule() -> Transport:
    """No observations.

    Emitting one would seed the next session's prompts with text this file
    wrote, and a replay would then be measuring its own stub.
    """

    def call(prompt: str, feedback: str | None = None) -> dict[str, Any]:
        return {"observations": []}

    return call


def rule_transports(
    symbols: Sequence[str] | None = None, size_multiplier: float = 1.0
) -> dict[str, Transport]:
    """One rule stub per agent, in the shape the harness expects."""
    return {
        "a1": a1_rule(symbols),
        "a2": a2_rule(symbols),
        "a3": a3_rule(size_multiplier),
        "a4": a4_rule(),
        "a5": a5_rule(),
        "a6": a6_rule(),
    }
