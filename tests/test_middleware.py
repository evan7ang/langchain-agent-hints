"""In-process middleware tests: emission rules over a real agent loop (keyless)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from conftest import SpyChatModel, hinted_headers, independent_decode, read_status_events
from langchain_agent_hints.middleware import AgentHintsMiddleware
from langchain_agent_hints.status import AgentStatusEventsMiddleware, JsonlStatusEventSink


def _make_tools(workspace: Path) -> list[Any]:
    @tool
    def write_note(file_path: str, content: str) -> str:
        """Write a note file into the workspace."""

        target = workspace / file_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"wrote {file_path}"

    return [write_note]


def _plain_turns() -> tuple[AIMessage, AIMessage]:
    return (
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "write_note",
                    "args": {"file_path": "note.txt", "content": "hint probe\n"},
                    "id": "call_probe",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(content="hint probe finished"),
    )


def _build(workspace: Path, spy: SpyChatModel, *, checkpointer: bool = True, **hints_kwargs: Any):
    sink_path = workspace / "status-events.jsonl"
    agent = create_agent(
        model=spy,
        tools=_make_tools(workspace),
        middleware=[
            AgentHintsMiddleware(agent_type="test-agent", **hints_kwargs),
            AgentStatusEventsMiddleware(sink=JsonlStatusEventSink(sink_path)),
        ],
        checkpointer=MemorySaver() if checkpointer else None,
    )
    return agent, sink_path


def _config(thread_id: str | None) -> dict[str, Any] | None:
    return {"configurable": {"thread_id": thread_id}} if thread_id else None


def test_two_calls_hinted_with_state_derived_call_index(tmp_path: Path) -> None:
    spy = SpyChatModel(messages=iter(list(_plain_turns())))
    agent, sink_path = _build(tmp_path, spy)

    agent.invoke({"messages": [("user", "hint probe")]}, config=_config("t1"))

    hints = [independent_decode(encoded) for encoded in hinted_headers(spy)]
    assert len(hints) == 2, f"expected 2 hinted model calls, got {len(hints)}"
    first, second = hints
    assert first["session"] == {
        "session_id": "t1",
        "agent_type": "test-agent",
        "call_index": 1,
        "epoch": 0,
    }
    assert second["session"]["call_index"] == 2
    assert first["agent_status"]["state"] == "running"
    assert first["priority"] == {"level": 3, "reason": "user_waiting"}
    assert first["cache_control"] == [{"type": "pin", "range": {"end": 1}}]
    assert "prediction" not in first and "strict_priority" not in first
    assert "parent_session_id" not in first["session"]
    assert "derivation" not in first["session"]

    # The full status lifecycle of the plain round rode along.
    states = [e["agent_status"]["state"] for e in read_status_events(sink_path, "t1")]
    assert states == ["running", "tool", "running", "finalize"]


def test_no_thread_id_degrades_to_no_hint_no_events(tmp_path: Path) -> None:
    spy = SpyChatModel(messages=iter(list(_plain_turns())))
    agent, sink_path = _build(tmp_path, spy, checkpointer=False)

    agent.invoke({"messages": [("user", "hint probe")]}, config=_config(None))

    assert not list(hinted_headers(spy)), "no-thread-id run must not carry a hint"
    assert read_status_events(sink_path) == []


def test_custom_options_reach_the_wire(tmp_path: Path) -> None:
    from langchain_agent_hints.schema import CacheControlHint, PriorityHint

    spy = SpyChatModel(messages=iter([AIMessage(content="no tools, one call, final answer")]))
    agent, _ = _build(
        tmp_path,
        spy,
        priority=PriorityHint(level=1, reason="deadline"),
        cache_control=(CacheControlHint(type="ephemeral", ttl_s=600),),
    )
    agent.invoke({"messages": [("user", "hi")]}, config=_config("t-custom"))

    hints = [independent_decode(encoded) for encoded in hinted_headers(spy)]
    assert hints[0]["session"]["agent_type"] == "test-agent"
    assert hints[0]["priority"] == {"level": 1, "reason": "deadline"}
    assert hints[0]["cache_control"] == [{"type": "ephemeral", "ttl_s": 600}]


def test_async_path_hints_identically(tmp_path: Path) -> None:
    spy = SpyChatModel(messages=iter(list(_plain_turns())))
    agent, sink_path = _build(tmp_path, spy)

    asyncio.run(agent.ainvoke({"messages": [("user", "hint probe")]}, config=_config("t-async")))

    hints = [independent_decode(encoded) for encoded in hinted_headers(spy)]
    assert [h["session"]["call_index"] for h in hints] == [1, 2]
    assert hints[0]["session"]["session_id"] == "t-async"
    states = [e["agent_status"]["state"] for e in read_status_events(sink_path, "t-async")]
    assert states == ["running", "tool", "running", "finalize"]


def test_epoch_bumps_on_prefix_rewrite() -> None:
    mw = AgentHintsMiddleware(agent_type="t")
    assert mw._take_epoch_slot("s", (3, "aaa")) == 0
    assert mw._take_epoch_slot("s", (3, "aaa")) == 0  # same prefix: no bump
    assert mw._take_epoch_slot("s", (2, "aaa")) == 1  # shrink: summarization signal
    assert mw._take_epoch_slot("s", (2, "bbb")) == 2  # first message replaced
    assert mw._take_epoch_slot("other", (1, "aaa")) == 0  # per-session isolation
