"""Wire proof: the real OpenAI-compatible adapter sends the X-Agent-Hint header.

Ports the capture-stub evidence from the original harness e2e into pytest:
a real ``langchain-openai`` chat model against a local OpenAI-compatible
capture stub, asserting the captured HTTP requests carry the exact protocol
shape (decoded independently of the encode path under test). No API key.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from conftest import HINT_HEADER, SpyChatModel, independent_decode
from langchain_agent_hints.middleware import AgentHintsMiddleware


def _records(capture_path: Path) -> list[dict[str, Any]]:
    lines = capture_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _assert_hint_shape(record: dict[str, Any], *, thread_id: str, agent_type: str) -> None:
    headers = record["headers"]
    assert HINT_HEADER.lower() in headers, f"no X-Agent-Hint header; got {sorted(headers)}"

    hint = independent_decode(headers[HINT_HEADER.lower()])
    assert hint["session"]["session_id"] == thread_id
    assert hint["session"]["agent_type"] == agent_type
    assert hint["session"]["call_index"] == 1
    assert hint["session"]["epoch"] == 0
    assert "parent_session_id" not in hint["session"]
    assert "derivation" not in hint["session"]
    assert hint["agent_status"]["state"] == "running"
    assert hint["priority"] == {"level": 3, "reason": "user_waiting"}
    assert hint["cache_control"] == [{"type": "pin", "range": {"end": 1}}]
    assert "prediction" not in hint and "strict_priority" not in hint

    # Header-only channel: the hint never leaks into the request body.
    body = record["body"] or {}
    assert "agent_hints" not in body and "nvext" not in body


def _build(stub_url: str):
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(model="stub-model", api_key="agent-hints-test-dummy", base_url=stub_url)
    return create_agent(
        model=model,
        middleware=[AgentHintsMiddleware(agent_type="test-agent")],
    )


def test_sync_request_carries_the_hint_header(stub_server) -> None:
    stub_url, capture = stub_server
    agent = _build(stub_url)

    result = agent.invoke(
        {"messages": [("user", "Agent hints wire probe.")]},
        config={"configurable": {"thread_id": "wire-1"}},
    )
    assert result["messages"][-1].content == "AGENT_HINTS_WIRE_OK"

    records = _records(capture)
    assert len(records) == 1, f"expected exactly 1 model request, got {len(records)}"
    _assert_hint_shape(records[0], thread_id="wire-1", agent_type="test-agent")


def test_async_request_carries_the_hint_header(stub_server) -> None:
    stub_url, capture = stub_server
    agent = _build(stub_url)

    result = asyncio.run(
        agent.ainvoke(
            {"messages": [("user", "Agent hints wire probe (async).")]},
            config={"configurable": {"thread_id": "wire-async"}},
        )
    )
    assert result["messages"][-1].content == "AGENT_HINTS_WIRE_OK"

    records = _records(capture)
    assert len(records) == 1
    _assert_hint_shape(records[0], thread_id="wire-async", agent_type="test-agent")


def test_fake_model_path_ignores_unknown_bind_kwargs() -> None:
    """The degradation contract on non-OpenAI models: extra bind kwargs are inert."""
    spy = SpyChatModel(messages=iter([AIMessage(content="ok")]))
    agent = create_agent(
        model=spy,
        middleware=[AgentHintsMiddleware(agent_type="test-agent")],
    )
    result = agent.invoke(
        {"messages": [("user", "hi")]}, config={"configurable": {"thread_id": "t-fake"}}
    )
    assert result["messages"][-1].content == "ok"
