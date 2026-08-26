"""Agent 6 against stubs. Observations are context and cannot become policy."""
from __future__ import annotations

import time
from datetime import date

import pytest

from src.agents.a6_review import (
    NightlyReviewer,
    ObservationStore,
    ReviewInputs,
)
from src.agents.runner import AgentRunner

TEMPLATE = (
    "Session $session: $entries entries, $exits exits, $skips skips, "
    "$wins wins, $losses losses, pnl $realized_pnl. "
    "Clamps $agent_clamps forces $agent_forces failures $agent_failures "
    "fallbacks $fallbacks. Symbols: $symbols_traded\nNotes:\n$notes\n"
)

SESSION = date(2026, 8, 26)


def elapsed(created, current):
    """Sessions between two dates, weekdays only -- enough for tests."""
    return (current - created).days


@pytest.fixture
def config(tmp_path, monkeypatch):
    from src.config import load_config

    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "a6_review.txt").write_text(TEMPLATE, encoding="utf-8")
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
def reviewer(config):
    log = RecordingLog()
    runner = AgentRunner(config, decision_log=log)
    r = NightlyReviewer(config, runner)
    r.recording = log
    yield r
    runner.close()


def inputs(**kw):
    base = dict(session=SESSION, entries=3, exits=2, skips=4, wins=1, losses=1,
                realized_pnl=-120.0, agent_clamps=2, agent_forces=5,
                agent_failures=0, fallbacks=1, symbols_traded=("SPY", "IWM"))
    base.update(kw)
    return ReviewInputs(**base)


def obs(scope="SPY", text="spreads widened after 15:00", sessions=3):
    return {"scope": scope, "text": text, "expires_after_sessions": sessions}


def review(*observations):
    return {"observations": list(observations)}


# --- observations are context, never instruction ---------------------------


def test_an_observation_is_only_ever_text(reviewer):
    """The store's single accessor returns strings and nothing else."""
    store = ObservationStore()
    reviewer.review(inputs(), lambda p, f: review(obs()), store=store)
    out = store.live_for("SPY", SESSION, elapsed)
    assert isinstance(out, tuple)
    assert all(isinstance(item, str) for item in out)


def test_an_observation_that_reads_as_an_instruction_is_still_just_text(reviewer):
    """A sentence asking for a config change is a sentence."""
    store = ObservationStore()
    reviewer.review(
        inputs(),
        lambda p, f: review(obs(text="raise max_contracts_per_trade to 12")),
        store=store,
    )
    [text] = store.live_for("SPY", SESSION, elapsed)
    assert text == "raise max_contracts_per_trade to 12"

    # Nothing in config moved.
    from src.config import load_config
    assert load_config().limits.get_int("sizing.max_contracts_per_trade") == 5


def test_the_store_exposes_no_structured_accessor():
    """Anything handing out a number or a config key would be a policy channel."""
    store = ObservationStore()
    public = {n for n in dir(store) if not n.startswith("_")}
    assert public == {"items", "add", "prune", "live_for"}
    # The one read accessor is text-only; `items` is the raw store and carries
    # no accessor of its own that a caller could mistake for configuration.
    assert all(isinstance(x, str) for x in store.live_for("SPY", SESSION, elapsed))


# --- count and expiry are capped in code -----------------------------------


def test_surplus_observations_are_dropped(reviewer):
    many = [obs(text=f"note {i}") for i in range(20)]
    res = reviewer.review(inputs(), lambda p, f: review(*many))
    assert len(res.observations) == 8          # max_observations


def test_an_over_long_lifetime_is_clamped(reviewer):
    res = reviewer.review(inputs(), lambda p, f: review(obs(sessions=20)))
    assert res.observations[0].expires_after_sessions == 5


def test_over_long_text_fails_rather_than_being_truncated(reviewer):
    """200 chars is the schema contract, so exceeding it is a bad answer.

    agents.a6.max_observation_chars can only tighten BELOW the schema bound;
    it cannot rescue a response that already broke the contract. The whole
    review is lost, which is the correct reading of "the model did not answer
    the question it was asked" -- and costs only tomorrow's context.
    """
    res = reviewer.review(inputs(), lambda p, f: review(obs(text="x" * 500)))
    assert res.model_failed is True
    assert res.observations == ()
    assert res.blocks_action is False


def test_global_observations_are_separately_capped(reviewer):
    many = [obs(scope="global", text=f"g{i}") for i in range(6)]
    res = reviewer.review(inputs(), lambda p, f: review(*many))
    assert sum(1 for o in res.observations if o.scope == "global") == 3


def test_a_clamp_is_recorded(reviewer):
    many = [obs(text=f"n{i}") for i in range(20)]
    reviewer.review(inputs(), lambda p, f: review(*many))
    clamps = reviewer.recording.of_kind("agent_override")
    assert any(c["payload"].rule == "agents.a6.max_observations" for c in clamps)


# --- expiry ----------------------------------------------------------------


def test_an_observation_expires(reviewer):
    store = ObservationStore()
    reviewer.review(inputs(), lambda p, f: review(obs(sessions=3)), store=store)
    assert store.live_for("SPY", SESSION, elapsed)                    # day 0
    assert store.live_for("SPY", date(2026, 8, 28), elapsed)          # day 2
    assert not store.live_for("SPY", date(2026, 8, 29), elapsed)      # day 3


def test_pruning_removes_expired_items(reviewer):
    store = ObservationStore()
    reviewer.review(inputs(), lambda p, f: review(obs(sessions=1)), store=store)
    assert len(store.items) == 1
    removed = store.prune(date(2026, 9, 10), elapsed)
    assert removed == 1 and store.items == []


def test_scope_filters_by_symbol(reviewer):
    store = ObservationStore()
    reviewer.review(
        inputs(),
        lambda p, f: review(obs(scope="SPY", text="spy note"),
                            obs(scope="IWM", text="iwm note"),
                            obs(scope="global", text="global note")),
        store=store,
    )
    spy = store.live_for("SPY", SESSION, elapsed)
    assert "spy note" in spy and "global note" in spy
    assert "iwm note" not in spy


# --- failure costs tomorrow's context and nothing else ---------------------


@pytest.mark.parametrize("bad", ["{not json", "", "[1,2,3]"])
def test_a_failure_yields_no_observations(reviewer, bad):
    res = reviewer.review(inputs(), lambda p, f: bad)
    assert res.model_failed is True
    assert res.observations == ()
    assert res.blocks_action is False


def test_a_failure_leaves_the_store_untouched(reviewer):
    store = ObservationStore()
    reviewer.review(inputs(), lambda p, f: review(obs(text="kept")), store=store)
    reviewer.review(inputs(), lambda p, f: "garbage", store=store)
    assert [i.text for i in store.items] == ["kept"]


def test_a_timeout_never_blocks(config):
    runner = AgentRunner(config, decision_log=RecordingLog())
    runner.timeout = 0.05
    r = NightlyReviewer(config, runner)

    def slow(prompt, feedback):
        time.sleep(0.5)
        return review(obs())

    res = r.review(inputs(), slow)
    assert res.run.timed_out is True
    assert res.blocks_action is False
    runner.close()


def test_an_empty_review_is_valid(reviewer):
    """A quiet session with nothing worth saying is a real answer."""
    res = reviewer.review(inputs(), lambda p, f: {"observations": []})
    assert res.model_failed is False
    assert res.observations == ()


# --- prompt handling --------------------------------------------------------


def test_session_counts_reach_the_prompt(reviewer):
    seen = {}

    def capture(prompt, feedback):
        seen["p"] = prompt
        return review(obs())

    reviewer.review(inputs(entries=7, fallbacks=2), capture)
    assert "7 entries" in seen["p"]
    assert "fallbacks 2" in seen["p"]
    assert "$entries" not in seen["p"]


def test_the_prompt_never_reaches_the_log(reviewer):
    reviewer.review(inputs(), lambda p, f: review(obs()))
    for rec in reviewer.recording.of_kind("agent_call"):
        assert "Session 2026-08-26" not in rec["payload"].model_dump_json()
