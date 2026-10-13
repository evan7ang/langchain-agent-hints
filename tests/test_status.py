"""Out-of-band status event tests: lifecycle, approval pause/resume, degradation."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from conftest import SpyChatModel, read_status_events
from langchain_agent_hints.middleware import AgentHintsMiddleware
from langchain_agent_hints.status import AgentStatusEventsMiddleware, JsonlStatusEventSink

GATED_TOOL = "gated_action"


def _make_tools(workspace: Path) -> list[Any]:
    @tool
    def gated_action(payload: str) -> str:
        """An external write gated behind human approval."""

        (workspace / "gated.txt").write_text(payload, encoding="utf-8")
        return f"approved write: {payload}"

    return [gated_action]


def _gated_turns() -> tuple[AIMessage, AIMessage]:
    return (
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": GATED_TOOL,
                    "args": {"payload": "status probe write"},
                    "id": "call_gated",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(content="final answer after approval"),
    )


def _build(
    tmp_path: Path,
    spy: SpyChatModel,
    *,
    interrupt_tools: frozenset[str] | set[str] = frozenset(),
    hitl: bool = True,
    checkpointer: bool = True,
):
    sink_path = tmp_path / "status-events.jsonl"
    middlewares: list[Any] = [
        AgentHintsMiddleware(agent_type="test-agent"),
        AgentStatusEventsMiddleware(
            sink=JsonlStatusEventSink(sink_path),
            interrupt_tools=interrupt_tools,
        ),
    ]
    if hitl:
        # HITL sits innermost (stack tail, mirroring deepagents), so its
        # after_model node interrupts before the user middleware's runs.
        middlewares.append(
            HumanInTheLoopMiddleware(
                interrupt_on={GATED_TOOL: {"allowed_decisions": ["approve", "reject"]}}
            )
        )
    agent = create_agent(
        model=spy,
        tools=_make_tools(tmp_path),
        middleware=middlewares,
        checkpointer=MemorySaver() if checkpointer else None,
    )
    return agent, sink_path


def _config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def _states(events: list[dict[str, Any]]) -> list[str]:
    return [e["agent_status"]["state"] for e in events]


def test_gated_round_pauses_without_finalize_then_resumes(tmp_path: Path) -> None:
    spy = SpyChatModel(messages=iter(list(_gated_turns())))
    agent, sink_path = _build(tmp_path, spy, interrupt_tools={GATED_TOOL})

    paused = agent.invoke(
        {"messages": [("user", "export the final report")]}, config=_config("gate-1")
    )
    assert "__interrupt__" in paused, sorted(paused.keys())

    paused_events = read_status_events(sink_path, "gate-1")
    assert _states(paused_events) == ["running", "pause"], paused_events
    pause_status = paused_events[1]["agent_status"]
    assert pause_status["reason"] == "human_approval"
    assert pause_status["tools"] == [{"name": GATED_TOOL, "call_id": "call_gated"}]

    resumed = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}), config=_config("gate-1")
    )
    assert "__interrupt__" not in resumed, "approved run must not pause again"
    assert (tmp_path / "gated.txt").read_text(encoding="utf-8") == "status probe write"

    assert _states(read_status_events(sink_path, "gate-1")) == [
        "running",
        "pause",
        "running",
        "tool",
        "running",
        "finalize",
    ]
    # Per-session ordering: the sink's seq counter is contiguous.
    all_events = read_status_events(sink_path, "gate-1")
    assert [e["seq"] for e in all_events] == [1, 2, 3, 4, 5, 6]


def test_ungated_tool_never_pauses(tmp_path: Path) -> None:
    spy = SpyChatModel(messages=iter(list(_gated_turns())))
    agent, sink_path = _build(tmp_path, spy, interrupt_tools=frozenset(), hitl=False)

    result = agent.invoke(
        {"messages": [("user", "export the final report")]}, config=_config("ungated")
    )
    assert "__interrupt__" not in result
    assert _states(read_status_events(sink_path, "ungated")) == [
        "running",
        "tool",
        "running",
        "finalize",
    ]


def test_no_thread_id_emits_no_events(tmp_path: Path) -> None:
    spy = SpyChatModel(messages=iter(list(_gated_turns())))
    agent, sink_path = _build(tmp_path, spy, hitl=False, checkpointer=False)

    agent.invoke({"messages": [("user", "export the final report")]})
    assert read_status_events(sink_path) == []


def test_async_gated_round_matches_sync_semantics(tmp_path: Path) -> None:
    spy = SpyChatModel(messages=iter(list(_gated_turns())))
    agent, sink_path = _build(tmp_path, spy, interrupt_tools={GATED_TOOL})

    paused = asyncio.run(
        agent.ainvoke({"messages": [("user", "export the final report")]}, config=_config("gate-a"))
    )
    assert "__interrupt__" in paused, sorted(paused.keys())
    assert _states(read_status_events(sink_path, "gate-a")) == ["running", "pause"]

    asyncio.run(
        agent.ainvoke(
            Command(resume={"decisions": [{"type": "approve"}]}), config=_config("gate-a")
        )
    )
    assert _states(read_status_events(sink_path, "gate-a")) == [
        "running",
        "pause",
        "running",
        "tool",
        "running",
        "finalize",
    ]
