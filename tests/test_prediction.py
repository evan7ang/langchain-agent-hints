"""Field completion: measured prediction, priority folding, static overrides."""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from conftest import SpyChatModel, hinted_headers, independent_decode
from langchain_agent_hints.middleware import AgentHintsMiddleware
from langchain_agent_hints.observation import ObservationOptions
from langchain_agent_hints.schema import (
    NextActionHint,
    PredictionHint,
    PriorityHint,
    normalize_priority_level,
)


@tool
def web_search(query: str) -> str:
    """Search the web."""

    return f"results for {query}"


def _turns() -> list[AIMessage]:
    return [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "web_search", "args": {"query": "x"}, "id": "c1", "type": "tool_call"}
            ],
            usage_metadata={"input_tokens": 10, "output_tokens": 40, "total_tokens": 50},
        ),
        AIMessage(
            content="done",
            usage_metadata={"input_tokens": 20, "output_tokens": 60, "total_tokens": 80},
        ),
    ]


def _session_agent(hints: AgentHintsMiddleware, sessions: int) -> tuple[Any, SpyChatModel]:
    messages: list[AIMessage] = []
    for _ in range(sessions):
        messages.extend(_turns())
    spy = SpyChatModel(messages=iter(messages))
    return create_agent(model=spy, tools=[web_search], middleware=[hints]), spy


def _hints_by_session(spy: SpyChatModel) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for encoded in hinted_headers(spy):
        wire = independent_decode(encoded)
        grouped.setdefault(wire["session"]["session_id"], []).append(wire)
    return grouped


def test_measured_prediction_appears_after_enough_sessions() -> None:
    hints = AgentHintsMiddleware(
        agent_type="pred-agent", observation=ObservationOptions(min_samples=2)
    )
    agent, spy = _session_agent(hints, sessions=4)
    for index in range(4):
        agent.invoke(
            {"messages": [("user", "go")]},
            config={"configurable": {"thread_id": f"p{index}"}},
        )

    grouped = _hints_by_session(spy)
    # The first three sessions seed the rolling window; the fourth carries
    # measured estimates on its very first request.
    assert "prediction" not in grouped["p0"][0]
    prediction = grouped["p3"][0]["prediction"]
    assert prediction["latency_ms"] is not None
    assert prediction["output_tokens"] == 60
    assert prediction["remaining_calls"] == 1
    assert prediction["next_action"]["state"] == "running"
    assert prediction["next_action"]["tools"] == [{"name": "web_search"}]
    assert prediction["next_action"]["prob"] == 0.5


def test_priority_levels_are_folded_to_the_two_value_domain() -> None:
    assert normalize_priority_level(0) == 1
    assert normalize_priority_level(1) == 1
    assert normalize_priority_level(2) == 1
    assert normalize_priority_level(3) == 3
    assert normalize_priority_level(4) == 3
    assert normalize_priority_level(7) == 3

    hints = AgentHintsMiddleware(
        agent_type="fold-agent",
        priority=PriorityHint(level=7, reason="user_waiting"),
        observation=ObservationOptions(enabled=False),
    )
    spy = SpyChatModel(messages=iter([AIMessage(content="ok")]))
    agent = create_agent(model=spy, middleware=[hints])
    agent.invoke({"messages": [("user", "hi")]}, config={"configurable": {"thread_id": "f1"}})
    wire = independent_decode(next(hinted_headers(spy)))
    assert wire["priority"] == {"level": 3, "reason": "user_waiting"}


def test_strict_priority_is_control_plane_injected_only() -> None:
    hints = AgentHintsMiddleware(
        agent_type="control-agent",
        strict_priority=PriorityHint(level=3, reason="critical_path"),
        observation=ObservationOptions(enabled=False),
    )
    spy = SpyChatModel(messages=iter([AIMessage(content="ok")]))
    agent = create_agent(model=spy, middleware=[hints])
    agent.invoke({"messages": [("user", "hi")]}, config={"configurable": {"thread_id": "c1"}})
    wire = independent_decode(next(hinted_headers(spy)))
    assert wire["strict_priority"] == {"level": 3, "reason": "critical_path"}


def test_static_prediction_overrides_measured_fields() -> None:
    hints = AgentHintsMiddleware(
        agent_type="override-agent",
        prediction=PredictionHint(
            output_tokens=999,
            next_action=NextActionHint(state="finalize", prob=0.9),
        ),
        observation=ObservationOptions(min_samples=1),
    )
    spy = SpyChatModel(
        messages=iter(
            [
                AIMessage(
                    content="one",
                    usage_metadata={"input_tokens": 1, "output_tokens": 5, "total_tokens": 6},
                ),
                AIMessage(
                    content="two",
                    usage_metadata={"input_tokens": 1, "output_tokens": 6, "total_tokens": 7},
                ),
            ]
        )
    )
    agent = create_agent(model=spy, middleware=[hints])
    for index in range(2):
        agent.invoke(
            {"messages": [("user", "hi")]},
            config={"configurable": {"thread_id": f"o{index}"}},
        )
    wire = list(hinted_headers(spy))[-1]
    decoded = independent_decode(wire)
    assert decoded["prediction"]["output_tokens"] == 999
    assert decoded["prediction"]["next_action"]["state"] == "finalize"


def test_prediction_is_omitted_when_observation_is_disabled() -> None:
    hints = AgentHintsMiddleware(
        agent_type="off-agent", observation=ObservationOptions(enabled=False)
    )
    spy = SpyChatModel(messages=iter([AIMessage(content="ok")]))
    agent = create_agent(model=spy, middleware=[hints])
    agent.invoke({"messages": [("user", "hi")]}, config={"configurable": {"thread_id": "x"}})
    wire = independent_decode(next(hinted_headers(spy)))
    assert "prediction" not in wire and "strict_priority" not in wire
