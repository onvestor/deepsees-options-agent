"""The operator's prompt templates must match what the modules can supply.

``prompts/`` is gitignored and absent from a fresh clone, so every test here
skips when it is missing -- the suite still runs offline against a bare
checkout. Where prompts *are* present, these are the checks worth having,
because the failure they catch is invisible until a live session:

``prompt_loader`` renders with ``string.Template.substitute``, which raises on
a placeholder the caller did not supply. A prompt naming ``$atr_pct`` when the
dataclass supplies ``atr_pct_of_spot`` does not send a slightly wrong prompt --
it raises ``ConfigError`` at the moment the agent is called, which on the entry
path is a skipped trade. Renaming a field on an ``Inputs`` dataclass is
therefore a breaking change to an operator file the repository cannot see, and
this test is the only thing that says so.

A bare ``$`` is the same class of failure. ``Template`` treats it as a malformed
placeholder and raises, so a prompt written with a dollar sign in front of a
number breaks the agent that loads it.
"""
from __future__ import annotations

import datetime
from pathlib import Path
from string import Template

import pytest

from src.agents.a1_regime import PROMPT_NAME as A1_PROMPT, RegimeInputs
from src.agents.a2_context import PROMPT_NAME as A2_PROMPT, ContextInputs
from src.agents.a3_risk import PROMPT_NAME as A3_PROMPT, RiskInputs
from src.agents.a4_contract import PROMPT_NAME as A4_PROMPT, ContractInputs
from src.agents.a5_exit import PROMPT_NAME as A5_PROMPT, ExitInputs
from src.agents.a6_review import PROMPT_NAME as A6_PROMPT, ReviewInputs

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"


def _fields() -> dict[str, dict[str, object]]:
    """One representative field set per agent, built from the real dataclasses.

    Values are placeholders -- this asserts on the *keys*, which is what the
    templates reference.
    """
    return {
        A1_PROMPT: RegimeInputs(
            symbol="NVDA", spot=180.0, atr=4.0, atr_pct_of_spot=0.022,
            realized_vol=0.41, rsi=58.0, ema_fast_value=178.0,
            ema_slow_value=172.0, trend_pct_20d=0.06, above_vwap=True,
        ).as_fields(),
        A2_PROMPT: ContextInputs(
            symbol="NVDA", spot=180.0, atr_pct_of_spot=0.022, realized_vol=0.41,
            iv_vs_rv20=1.1, iv_percentile=0.55, trend_pct_20d=0.06,
        ).as_fields(),
        A3_PROMPT: RiskInputs(
            symbol="NVDA", contract_symbol="NVDA261016C00185000", base_contracts=2,
            cost_per_contract=1200.0, max_risk_per_contract=1200.0,
            risk_budget=2500.0, equity=100000.0, open_positions=1,
            open_premium=1800.0, regime="trending_up", confidence=0.62,
            bias_strength=0.55, iv_assessment="fair",
        ).as_fields(),
        A4_PROMPT: ContractInputs(
            symbol="NVDA", spot=180.0, atr=4.0, survivors=(), regime="trending_up",
            confidence=0.62, directional_bias="bullish", bias_strength=0.55,
            iv_assessment="fair", target_expiry="2026-10-16", session_dte=34,
        ).as_fields(12),
        A5_PROMPT: ExitInputs(
            symbol="NVDA", contract_symbol="NVDA261016C00185000",
            entry_premium=12.0, current_premium=14.0, pnl_pct=16.7,
            current_stop_pct=-40.0, target_pct=75.0, sessions_held=2,
            max_hold_sessions=5, sessions_to_expiry=30, contracts=2,
        ).as_fields(),
        A6_PROMPT: ReviewInputs(
            session=datetime.date(2026, 8, 28), entries=2, exits=1, skips=9,
            wins=1, losses=0, realized_pnl=340.0, agent_clamps=1, agent_forces=3,
            agent_failures=0, fallbacks=0,
        ).as_fields(),
    }


def _template(name: str) -> Template:
    path = PROMPT_DIR / name
    if not path.is_file():
        pytest.skip(f"{name} is operator-supplied and absent from this checkout")
    return Template(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", sorted(_fields()))
def test_every_placeholder_is_supplied(name: str) -> None:
    """No template may reference a field its caller cannot provide."""
    fields = _fields()[name]
    template = _template(name)
    unknown = sorted(set(template.get_identifiers()) - set(fields))
    assert not unknown, (
        f"{name} references {unknown}, which {name.split('_')[0]}'s as_fields() "
        f"does not supply. Available: {sorted(fields)}"
    )


@pytest.mark.parametrize("name", sorted(_fields()))
def test_template_renders(name: str) -> None:
    """substitute() must not raise -- including on a stray dollar sign."""
    _template(name).substitute(_fields()[name])


@pytest.mark.parametrize("name", sorted(_fields()))
def test_prompt_demands_bare_json(name: str) -> None:
    """``parse()`` does not excavate JSON from prose or a fenced block.

    A prompt that does not say so produces responses that fail validation for a
    reason no threshold explains, so the instruction is load-bearing rather
    than stylistic.
    """
    text = _template(name).template.lower()
    assert "json object and nothing else" in text, (
        f"{name} does not tell the model to return bare JSON; parse() rejects "
        "prose and code fences rather than salvaging an object from them"
    )
