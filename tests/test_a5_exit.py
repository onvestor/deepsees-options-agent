"""Agent 5 against stubs. Failure continues; every stop change is gated."""
from __future__ import annotations

import time

import pytest

from src.agents.a5_exit import ExitInputs, ExitManager
from src.agents.runner import AgentRunner
from src.agents.schemas import ExitAction

TEMPLATE = (
    "Position $contract_symbol on $symbol: entry $entry_premium now "
    "$current_premium ($pnl_pct%). Stop $current_stop_pct target $target_pct. "
    "Held $sessions_held of $max_hold_sessions, $sessions_to_expiry to expiry, "
    "$contracts contracts, regime $regime, spans_earnings=$spans_earnings.\n"
    "Observations:\n$observations\n"
)


@pytest.fixture
def config(tmp_path, monkeypatch):
    from src.config import load_config

    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "a5_exit.txt").write_text(TEMPLATE, encoding="utf-8")
    monkeypatch.setenv("DEEPSEES_PROMPT_DIR", str(prompts))
    return load_config()


class RecordingLog:
    def __init__(self):
        self.records = []

    def write(self, payload, action, **kw):
        self.records.append({"payload": payload, "action": action, **kw})

    def of_kind(self, kind):
        return [r for r in self.records if r["payload"].kind == kind]


@pytest.fixture
def manager(config):
    log = RecordingLog()
    runner = AgentRunner(config, decision_log=log)
    m = ExitManager(config, runner)
    m.recording = log
    yield m
    runner.close()


def inputs(**kw):
    base = dict(symbol="SPY", contract_symbol="SPY261016C00760000",
                entry_premium=2000.0, current_premium=2200.0, pnl_pct=10.0,
                current_stop_pct=-40.0, target_pct=75.0, sessions_held=2,
                max_hold_sessions=5, sessions_to_expiry=37, contracts=2)
    base.update(kw)
    return ExitInputs(**base)


def say(action, **kw):
    base = {"action": action, "reason": "r"}
    base.update(kw)
    return base


# --- the path semantics flip -----------------------------------------------


@pytest.mark.parametrize("bad", ["{not json", "", "[1,2,3]", {"action": "sell_everything"}])
def test_any_failure_holds_at_the_existing_stop(manager, bad):
    """Not a fallback guess -- the state the position was already in."""
    res = manager.manage(inputs(current_stop_pct=-40.0), lambda p, f: bad)
    assert res.model_failed is True
    assert res.action is ExitAction.HOLD
    assert res.stop_pct == -40.0
    assert res.stop_changed is False


def test_failure_never_blocks_the_loop(manager):
    """Halting would stop the process that manages open risk."""
    res = manager.manage(inputs(), lambda p, f: "garbage")
    assert res.blocks_action is False


def test_a_timeout_continues(config):
    runner = AgentRunner(config, decision_log=RecordingLog())
    runner.timeout = 0.05
    m = ExitManager(config, runner)

    def slow(prompt, feedback):
        time.sleep(0.5)
        return say("exit_now")

    res = m.manage(inputs(), slow)
    assert res.run.timed_out is True
    assert res.blocks_action is False
    assert res.action is ExitAction.HOLD
    assert res.stop_pct == -40.0
    runner.close()


def test_failure_logs_continue_not_skip(manager):
    manager.manage(inputs(), lambda p, f: "garbage")
    [call] = manager.recording.of_kind("agent_call")
    assert call["action"] == "continue"


def test_a_failure_removes_no_protection(manager):
    """The armed stop is unchanged; nothing the model fails to say weakens it."""
    before = -35.0
    res = manager.manage(inputs(current_stop_pct=before), lambda p, f: "")
    assert res.stop_pct == before


# --- tightens() gates every stop change ------------------------------------


def test_a_real_tightening_is_applied(manager):
    res = manager.manage(inputs(current_stop_pct=-40.0),
                         lambda p, f: say("tighten_stop", new_stop_pct=-25.0))
    assert res.action is ExitAction.TIGHTEN_STOP
    assert res.stop_pct == -25.0
    assert res.stop_changed is True


def test_widening_never_reaches_the_stop(manager):
    res = manager.manage(inputs(current_stop_pct=-40.0),
                         lambda p, f: say("tighten_stop", new_stop_pct=-55.0))
    assert res.action is ExitAction.HOLD        # forced by the validator
    assert res.stop_pct == -40.0                # and the gate holds it anyway
    assert res.stop_changed is False


def test_an_equal_stop_is_not_a_tightening(manager):
    res = manager.manage(inputs(current_stop_pct=-40.0),
                         lambda p, f: say("tighten_stop", new_stop_pct=-40.0))
    assert res.stop_changed is False
    assert res.stop_pct == -40.0


def test_a_trivial_tightening_is_refused(manager):
    """Below min_stop_tightening_pct -- churn, not protection."""
    res = manager.manage(inputs(current_stop_pct=-40.0),
                         lambda p, f: say("tighten_stop", new_stop_pct=-39.0))
    assert res.action is ExitAction.HOLD
    assert res.stop_changed is False


@pytest.mark.parametrize("action", ["hold", "scale_out_half", "exit_now"])
def test_a_stop_on_a_non_tighten_action_is_rejected_not_ignored(manager, action):
    """Silently dropping it would let the model believe a stop had moved."""
    res = manager.manage(inputs(), lambda p, f: say(action, new_stop_pct=-20.0))
    assert res.model_failed is True             # rejected by the schema
    assert res.action is ExitAction.HOLD
    assert res.stop_pct == -40.0


@pytest.mark.parametrize("action", ["hold", "scale_out_half", "exit_now"])
def test_no_stop_moves_on_a_non_tighten_action(manager, action):
    res = manager.manage(inputs(current_stop_pct=-40.0), lambda p, f: say(action))
    assert res.stop_changed is False
    assert res.stop_pct == -40.0


def test_a_stop_beyond_the_floor_is_clamped_then_gated(manager):
    res = manager.manage(inputs(current_stop_pct=-90.0),
                         lambda p, f: say("tighten_stop", new_stop_pct=-80.0))
    assert res.stop_pct == -50.0               # clamped to max_stop_pct
    assert res.stop_changed is True


# --- the actions that are representable ------------------------------------


def test_exit_now_is_carried_through(manager):
    res = manager.manage(inputs(), lambda p, f: say("exit_now"))
    assert res.exits_now is True
    assert res.action is ExitAction.EXIT_NOW


def test_scale_out_is_carried_through(manager):
    res = manager.manage(inputs(), lambda p, f: say("scale_out_half"))
    assert res.scales_out is True


def test_hold_is_a_valid_answer(manager):
    res = manager.manage(inputs(), lambda p, f: say("hold"))
    assert res.action is ExitAction.HOLD
    assert res.model_failed is False           # answering "hold" is not failing


def test_adding_to_a_position_is_not_expressible(manager):
    for forbidden in ("add_to_position", "widen_stop", "reverse", "double_down"):
        res = manager.manage(inputs(), lambda p, f, a=forbidden: say(a))
        assert res.model_failed is True
        assert res.action is ExitAction.HOLD


# --- prompt handling --------------------------------------------------------


def test_position_state_reaches_the_prompt(manager):
    seen = {}

    def capture(prompt, feedback):
        seen["p"] = prompt
        return say("hold")

    manager.manage(inputs(pnl_pct=-12.5, sessions_held=4), capture)
    assert "-12.50%" in seen["p"]
    assert "Held 4 of 5" in seen["p"]
    assert "$pnl_pct" not in seen["p"]


def test_the_prompt_never_reaches_the_log(manager):
    manager.manage(inputs(), lambda p, f: say("hold"))
    for rec in manager.recording.of_kind("agent_call"):
        assert "Position SPY261016C" not in rec["payload"].model_dump_json()
