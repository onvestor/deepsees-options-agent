"""Runner. Every test is a way the model call goes wrong.

The five deliberately broken responses: malformed JSON, valid JSON failing the
schema, a timeout, an empty response, and a response that quotes its own
prompt back.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest

from src.agents.runner import (
    AgentPath,
    AgentRunner,
    FailureTracker,
    prompt_overlap,
    scrub_response,
)


@pytest.fixture(scope="module")
def config():
    from src.config import load_config

    return load_config()


class RecordingLog:
    """Stands in for DecisionLog. Keeps what would have been written."""

    def __init__(self):
        self.records = []

    def write(self, payload, action, **kw):
        self.records.append({"payload": payload, "action": action, **kw})
        return None

    def of_kind(self, kind):
        return [r for r in self.records if r["payload"].kind == kind]


@pytest.fixture
def runner(config):
    log = RecordingLog()
    r = AgentRunner(config, decision_log=log)
    r.recording = log
    yield r
    r.close()


GOOD = {
    "symbol": "NVDA", "regime": "trending_up", "confidence": 0.9,
    "signal_profile": {"ema_fast": 9, "confirmation_bars": 2,
                       "require_vwap_alignment": False, "min_atr_multiple": 0.6,
                       "allowed_direction": "long_calls"},
    "rationale": "trend intact",
}
PROMPT = "You are a regime classifier. Return JSON with symbol, regime, confidence."


# --- the five broken responses ---------------------------------------------


def test_malformed_json_fails_and_does_not_trade(runner):
    res = runner.run("a1", AgentPath.ENTRY, PROMPT, lambda fb: "{not json")
    assert not res.ok
    assert res.decision is None
    assert res.blocks_action is True


def test_valid_json_failing_the_schema_fails(runner):
    res = runner.run("a1", AgentPath.ENTRY, PROMPT,
                     lambda fb: {"symbol": "NVDA", "regime": "moon"})
    assert not res.ok and res.blocks_action is True
    [call] = runner.recording.of_kind("agent_call")
    assert call["payload"].validation.status == "failed"


def test_a_timeout_is_recorded_as_a_timeout_not_a_parse_failure(config):
    log = RecordingLog()
    runner = AgentRunner(config, decision_log=log)
    runner.timeout = 0.05

    def slow(feedback):
        time.sleep(0.6)
        return GOOD

    res = runner.run("a1", AgentPath.ENTRY, PROMPT, slow)
    assert res.timed_out is True and not res.ok
    [call] = log.of_kind("agent_call")
    assert call["payload"].validation.status == "timeout"
    runner.close()


def test_an_empty_response_is_a_failure_not_an_empty_object(runner):
    for empty in ("", "   ", None):
        res = runner.run("a1", AgentPath.ENTRY, PROMPT, lambda fb, e=empty: e)
        assert not res.ok
        assert "empty" in (res.error or "").lower()


def test_a_response_quoting_its_own_prompt_is_withheld(runner):
    """The last path by which prompt text could reach a published log."""
    leaky = PROMPT + " " + PROMPT      # a model restating its instructions
    res = runner.run("a1", AgentPath.ENTRY, PROMPT, lambda fb: leaky)
    assert res.response_withheld is True
    [call] = runner.recording.of_kind("agent_call")
    raw = call["payload"].response_raw
    assert "withheld" in raw
    assert PROMPT not in raw           # the point of the whole exercise
    assert call["payload"].response_truncated is True


# --- timeout semantics differ by path --------------------------------------


def test_entry_path_timeout_blocks_the_trade(config):
    runner = AgentRunner(config, decision_log=RecordingLog())
    runner.timeout = 0.05
    res = runner.run("a1", AgentPath.ENTRY, PROMPT,
                     lambda fb: (time.sleep(0.5), GOOD)[1])
    assert res.timed_out and res.blocks_action is True
    runner.close()


def test_exit_path_timeout_does_not_block(config):
    """Safe by construction: the deterministic exits are always armed."""
    runner = AgentRunner(config, decision_log=RecordingLog())
    runner.timeout = 0.05
    res = runner.run("a5", AgentPath.EXIT, PROMPT,
                     lambda fb: (time.sleep(0.5), {"action": "hold", "reason": "r"})[1],
                     current_stop_pct=-40.0)
    assert res.timed_out is True
    assert res.ok is False
    assert res.blocks_action is False      # the loop carries on
    runner.close()


def test_exit_path_failure_logs_continue_not_skip(config):
    log = RecordingLog()
    runner = AgentRunner(config, decision_log=log)
    runner.run("a5", AgentPath.EXIT, PROMPT, lambda fb: "garbage", current_stop_pct=-40.0)
    [call] = log.of_kind("agent_call")
    assert call["action"] == "continue"
    runner.close()


def test_entry_path_failure_logs_skip(runner):
    runner.run("a1", AgentPath.ENTRY, PROMPT, lambda fb: "garbage")
    [call] = runner.recording.of_kind("agent_call")
    assert call["action"] == "skip"


def test_review_path_never_blocks(config):
    runner = AgentRunner(config, decision_log=RecordingLog())
    res = runner.run("a6", AgentPath.REVIEW, PROMPT, lambda fb: "garbage")
    assert not res.ok and res.blocks_action is False
    runner.close()


# --- prompts are hashed, never logged --------------------------------------


def test_the_prompt_never_appears_in_the_record(runner):
    secret = "SECRET PROMPT TEXT that must never be committed anywhere at all"
    runner.run("a1", AgentPath.ENTRY, secret, lambda fb: GOOD)
    [call] = runner.recording.of_kind("agent_call")
    dumped = call["payload"].model_dump_json()
    assert secret not in dumped
    assert call["payload"].prompt_hash
    assert call["payload"].prompt_chars == len(secret)


def test_the_hash_distinguishes_prompts(runner):
    runner.run("a1", AgentPath.ENTRY, "prompt one", lambda fb: GOOD)
    runner.run("a1", AgentPath.ENTRY, "prompt two", lambda fb: GOOD)
    a, b = runner.recording.of_kind("agent_call")
    assert a["payload"].prompt_hash != b["payload"].prompt_hash


# --- n-gram overlap --------------------------------------------------------


def test_overlap_is_zero_for_an_unrelated_response():
    assert prompt_overlap("the quick brown fox jumps over the lazy dog now",
                          "entirely different words appear in this instruction text here",
                          8) == 0.0


def test_overlap_is_total_for_an_exact_restatement():
    text = " ".join(f"word{i}" for i in range(40))
    assert prompt_overlap(text, text, 8) == 1.0


def test_a_short_response_cannot_leak():
    """Fewer words than the n-gram width yields no n-grams. Correctly 0."""
    assert prompt_overlap("yes", "a long prompt with many words in it indeed", 8) == 0.0


def test_scrub_keeps_a_clean_response():
    out = scrub_response("a b c d e f g h i j", "totally unrelated prompt text here", 8, 0.25)
    assert out.truncated is False and out.text == "a b c d e f g h i j"


def test_the_marker_records_the_measurement_not_the_text():
    text = " ".join(f"w{i}" for i in range(40))
    out = scrub_response(text, text, 8, 0.25)
    assert out.truncated is True
    assert "100%" in out.text and "prompt_hash" in out.text
    assert "w1 w2" not in out.text


# --- failure tracking for the kill switch ----------------------------------


def test_the_rate_needs_a_minimum_sample():
    """1 failure in 1 call is 100% and means nothing."""
    t = FailureTracker(window_minutes=60, rate_threshold=0.2, min_calls=5)
    t.record(failed=True)
    snap = t.snapshot()
    assert snap.rate == 1.0
    assert snap.halts_new_entries is False
    assert "need 5" in snap.reason


def test_a_sustained_failure_rate_halts_new_entries():
    t = FailureTracker(window_minutes=60, rate_threshold=0.2, min_calls=5)
    for _ in range(3):
        t.record(failed=True)
    for _ in range(7):
        t.record(failed=False)
    snap = t.snapshot()
    assert snap.calls == 10 and snap.failures == 3
    assert snap.rate == pytest.approx(0.3)
    assert snap.halts_new_entries is True
    assert "above 20%" in snap.reason


def test_a_healthy_rate_does_not_halt():
    t = FailureTracker(window_minutes=60, rate_threshold=0.2, min_calls=5)
    t.record(failed=True)
    for _ in range(9):
        t.record(failed=False)
    assert t.snapshot().halts_new_entries is False


def test_old_failures_leave_the_window():
    t = FailureTracker(window_minutes=60, rate_threshold=0.2, min_calls=5)
    old = datetime.now(tz=timezone.utc) - timedelta(hours=2)
    for _ in range(10):
        t.record(failed=True, at=old)
    for _ in range(5):
        t.record(failed=False)
    snap = t.snapshot()
    assert snap.calls == 5 and snap.failures == 0
    assert snap.halts_new_entries is False


def test_the_runner_counts_failures_but_not_clamps(runner):
    """A clamped response succeeded. It is not an agent failure."""
    clamped = dict(GOOD)
    clamped["signal_profile"] = {**GOOD["signal_profile"], "ema_fast": 7}
    for _ in range(5):
        runner.run("a1", AgentPath.ENTRY, PROMPT, lambda fb: clamped)
    snap = runner.failure_snapshot()
    assert snap.calls == 5 and snap.failures == 0


def test_the_runner_counts_real_failures(runner):
    for _ in range(5):
        runner.run("a1", AgentPath.ENTRY, PROMPT, lambda fb: "garbage")
    snap = runner.failure_snapshot()
    assert snap.calls == 5 and snap.failures == 5
    assert snap.halts_new_entries is True


# --- overrides reach the log as their own records --------------------------


def test_each_override_is_its_own_record(runner):
    payload = dict(GOOD)
    payload["confidence"] = 0.1                                    # force
    payload["signal_profile"] = {**GOOD["signal_profile"], "ema_fast": 7}   # clamp
    runner.run("a1", AgentPath.ENTRY, PROMPT, lambda fb: payload)
    overrides = runner.recording.of_kind("agent_override")
    assert len(overrides) == 2
    kinds = [o["payload"].override for o in overrides]
    assert kinds == ["clamp", "force"]
    assert [o["action"] for o in overrides] == ["agent_clamp", "agent_force"]


def test_an_override_record_carries_both_values(runner):
    payload = dict(GOOD)
    payload["signal_profile"] = {**GOOD["signal_profile"], "ema_fast": 7}
    runner.run("a1", AgentPath.ENTRY, PROMPT, lambda fb: payload)
    [ov] = runner.recording.of_kind("agent_override")
    assert ov["payload"].model_value == 7
    assert ov["payload"].applied_value == 8


def test_retry_is_visible_in_the_record(runner):
    calls = []

    def flaky(feedback):
        calls.append(feedback)
        return GOOD if feedback else "garbage"

    res = runner.run("a1", AgentPath.ENTRY, PROMPT, flaky)
    assert res.ok
    [call] = runner.recording.of_kind("agent_call")
    assert call["payload"].validation.attempt == 2
    assert len(calls) == 2
