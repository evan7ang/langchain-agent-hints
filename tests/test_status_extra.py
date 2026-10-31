"""Field completion on the status channel: join_barrier, suspend, HTTP sink."""

from __future__ import annotations

import json
import time
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from conftest import SpyChatModel, read_status_events
from langchain_agent_hints.middleware import AgentHintsMiddleware
from langchain_agent_hints.status import (
    AgentStatusEventsMiddleware,
    HttpStatusEventSink,
    JsonlStatusEventSink,
)


@tool
def task(description: str, subagent_type: str) -> str:
    """Delegate to a subagent (deepagents-compatible shape)."""

    return f"{subagent_type} done: {description}"


@tool
def gated_write(payload: str) -> str:
    """An external write gated behind human approval."""

    return f"wrote {payload}"


def test_subagent_delegation_reports_join_barrier(tmp_path: Path) -> None:
    sink_path = tmp_path / "events.jsonl"
    spy = SpyChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "task",
                            "args": {"description": "research x", "subagent_type": "researcher"},
                            "id": "call_task",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="finished"),
            ]
        )
    )
    agent = create_agent(
        model=spy,
        tools=[task],
        middleware=[
            AgentHintsMiddleware(agent_type="orchestrator"),
            AgentStatusEventsMiddleware(sink=JsonlStatusEventSink(sink_path)),
        ],
    )
    agent.invoke({"messages": [("user", "go")]}, config={"configurable": {"thread_id": "t"}})

    events = read_status_events(sink_path, "t")
    tool_events = [event for event in events if event["agent_status"]["state"] == "tool"]
    assert len(tool_events) == 1
    status = tool_events[0]["agent_status"]
    assert status["reason"] == "join_barrier"
    assert status["tools"] == [
        {
            "name": "agent:researcher",
            "call_id": "call_task",
            "child_session_id": "t:researcher",
        }
    ]


def test_pause_carries_expected_resume_and_turns_into_suspend(
    tmp_path: Path,
) -> None:
    sink_path = tmp_path / "events.jsonl"
    spy = SpyChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "gated_write",
                            "args": {"payload": "x"},
                            "id": "call_gated",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="final answer"),
            ]
        )
    )
    status = AgentStatusEventsMiddleware(
        sink=JsonlStatusEventSink(sink_path),
        interrupt_tools={"gated_write"},
        suspend_after_s=0.05,
        expected_resume_s=600,
    )
    agent = create_agent(
        model=spy,
        tools=[gated_write],
        middleware=[
            AgentHintsMiddleware(agent_type="approval-agent"),
            status,
            HumanInTheLoopMiddleware(interrupt_on={"gated_write": True}),
        ],
        checkpointer=MemorySaver(),
    )
    try:
        agent.invoke({"messages": [("user", "go")]}, config={"configurable": {"thread_id": "a"}})
        time.sleep(0.25)
    finally:
        status.close()

    events = read_status_events(sink_path, "a")
    pause = [event for event in events if event["agent_status"]["state"] == "pause"]
    assert len(pause) == 1
    assert pause[0]["agent_status"]["reason"] == "human_approval"
    assert pause[0]["prediction"] == {"expected_resume_s": 600}

    suspend = [event for event in events if event["agent_status"]["state"] == "suspend"]
    assert len(suspend) == 1
    assert suspend[0]["agent_status"]["reason"] == "checkpoint"


def test_http_sink_posts_the_event(stub_server: tuple[str, Path]) -> None:
    base_url, capture = stub_server
    sink = HttpStatusEventSink(base_url + "/agent-status", attempts=2)
    sink.send(
        {
            "session": {"session_id": "h1"},
            "agent_status": {"state": "running", "state_since_ms": 1},
        }
    )
    records = [json.loads(line) for line in capture.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["path"].endswith("/agent-status")
    assert records[0]["body"]["seq"] == 1
    assert records[0]["body"]["session"]["session_id"] == "h1"
