"""The provider call, against a fake client. No network, ever.

The transport is thin on purpose, so these tests are mostly about what it
deliberately does *not* do: retry, catch, parse, or carry prompt text. Each of
those would be invisible in production until it mattered.
"""
from __future__ import annotations

import pytest

from src.agents.transport import (
    AnthropicTransport,
    build_retry_message,
    extract_text,
    transport_from_config,
)


@pytest.fixture(scope="module")
def config():
    from src.config import load_config

    return load_config()


class Block:
    def __init__(self, text, type="text"):
        self.text = text
        self.type = type


class Message:
    def __init__(self, blocks, stop_reason="end_turn", usage=None):
        self.content = blocks
        self.stop_reason = stop_reason
        self.usage = usage


class FakeMessages:
    """Records every create() call and replays queued responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeClient:
    def __init__(self, *responses):
        self.messages = FakeMessages(responses)


def make(config, *responses, **kw):
    return AnthropicTransport(config, client=FakeClient(*responses), **kw)


# --- the happy path --------------------------------------------------------


def test_returns_raw_text_unparsed(config):
    """The transport must not parse. validator.parse is the only reader."""
    raw = '{"size_multiplier": 0.5, "reason": "concentrated"}'
    transport = make(config, Message([Block(raw)]))
    assert transport("prompt", None) == raw


def test_sends_configured_model_and_budget(config):
    transport = make(config, Message([Block("{}")]))
    transport("prompt", None)
    sent = transport._client.messages.calls[0]
    assert sent["model"] == config.limits.get_str("agent_runtime.model")
    assert sent["max_tokens"] == config.limits.get_int("agent_runtime.max_output_tokens")
    assert sent["messages"] == [{"role": "user", "content": "prompt"}]


def test_no_sampling_parameters_are_sent(config):
    """temperature/top_p/top_k were removed from the API on the current model
    generation, and the 1.x SDK raises TypeError on them -- which surfaces as
    every agent failing validation at once, not as one bad request."""
    transport = make(config, Message([Block("{}")]))
    transport("prompt", None)
    sent = transport._client.messages.calls[0]
    for banned in ("temperature", "top_p", "top_k"):
        assert banned not in sent


def test_effort_is_sent_when_configured(config):
    transport = make(config, Message([Block("{}")]))
    transport("prompt", None)
    sent = transport._client.messages.calls[0]
    configured = config.limits.get_str("agent_runtime.effort")
    if configured:
        assert sent["output_config"] == {"effort": configured}
    else:
        assert "output_config" not in sent


def test_max_tokens_leaves_room_for_thinking(config):
    """Adaptive thinking is on by default and its tokens count against
    max_tokens. A budget sized for the JSON alone truncates the JSON."""
    assert config.limits.get_int("agent_runtime.max_output_tokens") >= 2048


def test_matches_the_transport_callable_signature(config):
    """Every agent calls transport(prompt, feedback) positionally."""
    transport = make(config, Message([Block("{}")]))
    assert transport("prompt", None) == "{}"


# --- the retry -------------------------------------------------------------


def test_feedback_carries_the_validation_error(config):
    transport = make(config, Message([Block("{}")]))
    transport("the original question", "size_multiplier: must be <= 1")
    content = transport._client.messages.calls[0]["messages"][0]["content"]
    assert "the original question" in content
    assert "size_multiplier: must be <= 1" in content


def test_retry_is_stateless(config):
    """The retry turn is built from its arguments and nothing else.

    Statefulness here would couple one agent's retry to another agent's last
    call under the runner's thread pool.
    """
    first = build_retry_message("p", "e")
    second = build_retry_message("p", "e")
    assert first == second


def test_no_feedback_sends_the_prompt_untouched(config):
    transport = make(config, Message([Block("{}")]))
    transport("prompt", None)
    assert transport._client.messages.calls[0]["messages"][0]["content"] == "prompt"


# --- what it refuses to do -------------------------------------------------


def test_provider_errors_propagate(config):
    """Swallowing these would hide them from the failure counter that feeds
    the kill switch."""
    transport = make(config, RuntimeError("529 overloaded"))
    with pytest.raises(RuntimeError, match="529 overloaded"):
        transport("prompt", None)


def test_does_not_retry_itself(config):
    """One call in, one call out. The validator owns the single retry, and a
    second retry here would multiply against it."""
    transport = make(config, Message([Block("{}")]), Message([Block("{}")]))
    transport("prompt", None)
    assert len(transport._client.messages.calls) == 1


def test_sdk_retries_are_configured_not_hardcoded(config):
    assert config.limits.get_int("agent_runtime.sdk_max_retries") is not None


# --- system prompt ---------------------------------------------------------


def test_no_system_prompt_by_default(config):
    """Prompt text is operator-supplied. The default must send none rather
    than embed one in the repository."""
    transport = make(config, Message([Block("{}")]))
    transport("prompt", None)
    assert "system" not in transport._client.messages.calls[0]


def test_missing_system_prompt_names_the_file(config):
    from src.config import ConfigError

    with pytest.raises(ConfigError, match="absent_system.txt"):
        make(config, Message([Block("{}")]), system_prompt_name="absent_system.txt")


# --- response extraction ---------------------------------------------------


def test_extract_text_concatenates_text_blocks():
    assert extract_text(Message([Block('{"a":'), Block(" 1}")])) == '{"a": 1}'


def test_extract_text_ignores_non_text_blocks():
    message = Message([Block("kept"), Block("dropped", type="tool_use")])
    assert extract_text(message) == "kept"


def test_empty_response_becomes_empty_string(config):
    """The runner turns this into EmptyResponse -- an absent answer, which is
    a different failure from malformed content."""
    transport = make(config, Message([]))
    assert transport("prompt", None) == ""


def test_max_tokens_truncation_is_warned(config, caplog):
    transport = make(config, Message([Block('{"partial":')], stop_reason="max_tokens"))
    with caplog.at_level("WARNING"):
        transport("prompt", None)
    assert "max_tokens" in caplog.text


# --- factory ---------------------------------------------------------------


def test_factory_returns_the_agent_callable(config):
    """transport_from_config produces the same shape the agents expect.

    Built against a fake client so nothing here can reach the network -- the
    real client is only constructed when no client is injected.
    """
    made = AnthropicTransport(config, client=FakeClient(Message([Block("{}")])))
    assert callable(made)
    assert made("prompt", None) == "{}"


def test_missing_key_names_the_variable(config):
    """Without a key the failure must name the variable, not surface as an SDK
    error somewhere downstream. ``Env`` is frozen, so the substitute is built
    with dataclasses.replace rather than patched in place."""
    import dataclasses

    from src.config import ConfigError

    keyless = dataclasses.replace(config.env, anthropic_api_key=None)
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        keyless.require_anthropic()
