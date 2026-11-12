"""Agent loop tests with a faked stream — no network, no API key.

These pin the tool round-trip against what Groq's docs specify:
  * the tool is sent in OpenAI function shape,
  * a streamed tool call is accumulated correctly even when split across deltas,
  * the result goes back as role="tool" with a matching tool_call_id, and with
    no `name` field (Groq 400s on messages[].name),
  * the token cap uses max_completion_tokens, not max_tokens,
  * and the loop stops instead of running forever.
"""
from types import SimpleNamespace as NS

import pytest

from chatagent import agent as agent_mod
from chatagent import config
from chatagent.agent import ChatAgent, _parse_query
from chatagent.memory import Memory
from chatagent.search import WEB_SEARCH_TOOL


def _chunk(content=None, tool_calls=None, finish=None):
    delta = NS(content=content, tool_calls=tool_calls, reasoning_content=None)
    return NS(choices=[NS(delta=delta, finish_reason=finish, index=0)], usage=None)


def _tc(index=0, call_id=None, name=None, args=None):
    return NS(index=index, id=call_id, function=NS(name=name, arguments=args))


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "gsk-dummy")
    a = ChatAgent(memory=Memory(str(tmp_path / "t.db")))
    monkeypatch.setattr(agent_mod, "web_search", lambda q, m=None: f"RESULTS FOR {q}")
    return a


def _script(agent, monkeypatch, rounds):
    """Make the client return each canned stream in turn, recording requests."""
    sent = []

    def fake_create(**kwargs):
        sent.append(kwargs)
        return iter(rounds[len(sent) - 1])

    monkeypatch.setattr(agent.client.chat.completions, "create", fake_create)
    return sent


# --- wiring ---------------------------------------------------------------

def test_tool_is_openai_function_shape():
    assert WEB_SEARCH_TOOL["type"] == "function"
    fn = WEB_SEARCH_TOOL["function"]
    assert fn["name"] == "web_search"
    assert fn["parameters"]["required"] == ["query"]
    assert "input_schema" not in WEB_SEARCH_TOOL  # the Anthropic shape, not this one


def test_client_points_at_groq(agent):
    assert "api.groq.com" in str(agent.client.base_url)


def test_uses_groq_parameter_name_for_token_cap(agent, monkeypatch):
    """Groq's OpenAI-compatible endpoint takes max_completion_tokens."""
    sent = _script(agent, monkeypatch, [[_chunk(content="hi"), _chunk(finish="stop")]])
    agent.ask("hello")
    assert "max_completion_tokens" in sent[0]
    assert "max_tokens" not in sent[0]


def test_tool_result_omits_name_field(agent, monkeypatch):
    """Groq rejects messages[].name with a 400, so the tool turn must not send it."""
    rounds = [
        [_chunk(tool_calls=[_tc(0, "c1", "web_search", '{"query":"x"}')]),
         _chunk(finish="tool_calls")],
        [_chunk(content="done"), _chunk(finish="stop")],
    ]
    _script(agent, monkeypatch, rounds)
    agent.ask("search please")
    tool_msg = next(m for m in agent.messages if m["role"] == "tool")
    assert set(tool_msg) == {"role", "tool_call_id", "content"}


def test_system_message_carries_memory(agent):
    agent.memory.add_turn("user", "explain LoRA fine-tuning")
    msg = agent._system_message("remind me about LoRA")
    assert msg["role"] == "system"
    assert "LoRA" in msg["content"]
    assert "one tool: web_search" in msg["content"]


def test_parse_query_survives_bad_json():
    assert _parse_query('{"query": "delhi aqi"}') == "delhi aqi"
    assert _parse_query("") == ""
    assert _parse_query("not json at all") == ""
    assert _parse_query("{}") == ""


# --- the tool round-trip --------------------------------------------------

def test_streamed_tool_call_round_trip(agent, monkeypatch):
    """A call split across deltas must be reassembled and answered correctly."""
    rounds = [
        # round 1: the model asks for a search, arguments arriving in pieces
        [
            _chunk(tool_calls=[_tc(0, "call_abc", "web_search", '{"que')]),
            _chunk(tool_calls=[_tc(0, None, None, 'ry": "f1 brazil"}')]),
            _chunk(finish="tool_calls"),
        ],
        # round 2: the model answers from the results
        [_chunk(content="Verstappen"), _chunk(content=" won."), _chunk(finish="stop")],
    ]
    sent = _script(agent, monkeypatch, rounds)
    searched = []

    answer = agent.ask("who won in brazil?", on_search=searched.append)

    assert answer == "Verstappen won."
    assert searched == ["f1 brazil"], "fragmented arguments were not reassembled"

    # The assistant turn carries the call in OpenAI shape...
    assistant = next(m for m in agent.messages if m.get("tool_calls"))
    call = assistant["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "web_search"

    # ...and the result goes back linked by tool_call_id.
    tool_msg = next(m for m in agent.messages if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == call["id"] == "call_abc"
    assert tool_msg["content"] == "RESULTS FOR f1 brazil"

    # Two requests, and the tool was offered on both.
    assert len(sent) == 2
    assert sent[0]["tools"] == [WEB_SEARCH_TOOL]
    assert sent[0]["stream"] is True
    assert sent[0]["model"] == config.MODEL


def test_no_tool_call_means_one_request(agent, monkeypatch):
    rounds = [[_chunk(content="Lisbon."), _chunk(finish="stop")]]
    sent = _script(agent, monkeypatch, rounds)
    assert agent.ask("capital of portugal?") == "Lisbon."
    assert len(sent) == 1, "answered without searching, so only one round trip"
    assert not any(m["role"] == "tool" for m in agent.messages)


def test_reasoning_only_chunks_are_not_answer_text(agent, monkeypatch):
    """Reasoning chunks carry no delta.content; they aren't answer text."""
    rounds = [[
        _chunk(content=None),            # reasoning-only chunk
        _chunk(content="42"),
        _chunk(finish="stop"),
    ]]
    _script(agent, monkeypatch, rounds)
    assert agent.ask("answer?") == "42"


def test_tool_loop_is_bounded(agent, monkeypatch):
    """A model that keeps searching must not spin forever at the user's cost."""
    forever = [
        _chunk(tool_calls=[_tc(0, "c1", "web_search", '{"query":"x"}')]),
        _chunk(finish="tool_calls"),
    ]
    sent = _script(agent, monkeypatch, [list(forever) for _ in range(20)])
    agent.ask("loop please")
    assert len(sent) == config.MAX_TOOL_ROUNDS


def test_turn_is_persisted_to_memory(agent, monkeypatch):
    _script(agent, monkeypatch, [[_chunk(content="Hi."), _chunk(finish="stop")]])
    agent.ask("hello")
    roles = [t["role"] for t in agent.memory.recent_turns(10)]
    assert roles == ["user", "assistant"]
