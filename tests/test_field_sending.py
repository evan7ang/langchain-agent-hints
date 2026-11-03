"""Field-sending evidence for LangGraph: lineage, epoch, state, prediction, priority, cache.

Mirrors AGENT_HINT_FIELD_TEST_REPORT_20261102.md per-field checks against the
real middleware wire output (header decode), using the same fake-clock/request
techniques as the behavior suites.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import langchain_agent_hints.middleware as middleware_mod
from langchain_agent_hints.middleware import AgentHintsMiddleware
from langchain_agent_hints.schema import decode_agent_hint


class _Request:
    """Minimal ModelRequest stand-in for direct middleware-hook tests."""

    def __init__(self, messages: list[Any]) -> None:
        self.messages = messages
        self.model_settings: dict[str, Any] = {}
        self.model = SimpleNamespace(model_name="m")
        self.tools: list[Any] = []
        self.system_message = None

    def override(self, **kwargs: Any) -> _Request:
        clone = copy.copy(self)
        clone.model_settings = kwargs.get("model_settings", self.model_settings)
        return clone


def _wire(request: Any, middleware: AgentHintsMiddleware) -> dict[str, Any]:
    hinted = middleware._hinted_request(request)
    assert hinted is not None
    encoded = hinted.model_settings["extra_headers"]["X-Agent-Hint"]
    return decode_agent_hint(encoded)


def _config(*, thread_id: str = "t", subagent: bool = False, agent_name: str | None = None):
    configurable: dict[str, Any] = {"thread_id": thread_id}
    if subagent:
        configurable["ls_agent_type"] = "subagent"
    metadata: dict[str, Any] = {}
    if agent_name:
        metadata["lc_agent_name"] = agent_name
    return {"configurable": configurable, "metadata": metadata}


def test_subagent_lineage_is_real(monkeypatch) -> None:
    """spawn + sync + parent_session_id + child agent_type from runtime config."""
    monkeypatch.setattr(
        middleware_mod,
        "ensure_config",
        lambda: _config(thread_id="t", subagent=True, agent_name="researcher"),
    )
    mw = AgentHintsMiddleware(agent_type="orchestrator")
    wire = _wire(_Request([HumanMessage(content="go")]), mw)

    assert wire["session"]["session_id"] == "t:researcher"
    assert wire["session"]["agent_type"] == "researcher"
    assert wire["session"]["parent_session_id"] == "t"
    assert wire["session"]["derivation"] == {"relation": "spawn", "join": "sync"}
    # A subagent is self-driven: binarization keeps 3 for user intervention only.
    assert wire["priority"] == {"level": 1, "reason": "critical_path"}


def test_root_session_omits_lineage(monkeypatch) -> None:
    monkeypatch.setattr(middleware_mod, "ensure_config", lambda: _config(thread_id="t"))
    mw = AgentHintsMiddleware(agent_type="office-agent")
    wire = _wire(_Request([HumanMessage(content="go")]), mw)
    assert "parent_session_id" not in wire["session"]
    assert "derivation" not in wire["session"]
    assert wire["priority"] == {"level": 3, "reason": "user_waiting"}


def test_epoch_bumps_on_prefix_rewrite_at_the_wire(monkeypatch) -> None:
    monkeypatch.setattr(middleware_mod, "ensure_config", lambda: _config(thread_id="epoch"))
    mw = AgentHintsMiddleware(agent_type="epoch-agent")
    first = _wire(_Request([HumanMessage(content="a"), HumanMessage(content="b")]), mw)
    second = _wire(_Request([HumanMessage(content="a")]), mw)  # history shrank
    assert first["session"]["epoch"] == 0
    assert second["session"]["epoch"] == 1


def test_state_and_call_index_on_the_hint(monkeypatch) -> None:
    monkeypatch.setattr(middleware_mod, "ensure_config", lambda: _config(thread_id="t"))
    mw = AgentHintsMiddleware(agent_type="a")
    request = _Request(
        [HumanMessage(content="q"), AIMessage(content="done"), HumanMessage(content="q2")]
    )
    wire = _wire(request, mw)
    # In-band state is always running; call_index counts prior assistant turns.
    assert wire["agent_status"]["state"] == "running"
    assert wire["session"]["call_index"] == 2


def test_prediction_priority_cache_are_emitted_from_real_facts(monkeypatch) -> None:
    """A long system prefix + a tool continuation -> ephemeral cache + level 1."""
    monkeypatch.setattr(middleware_mod, "ensure_config", lambda: _config(thread_id="facts"))
    mw = AgentHintsMiddleware(agent_type="facts-agent")
    request = _Request(
        [
            HumanMessage(content="q"),
            ToolMessage(content="tool-result", tool_call_id="c1"),
        ]
    )
    request.system_message = SimpleNamespace(content="cache block " * 120)
    wire = _wire(request, mw)

    assert wire["agent_status"]["state"] == "running"
    assert wire["priority"] == {"level": 1, "reason": "critical_path"}  # tool continuation
    assert wire["cache_control"][0]["type"] == "ephemeral"
    assert 60 <= wire["cache_control"][0]["ttl_s"] <= 3600
    assert wire["cache_control"][0]["range"]["end"] == 3  # system + 2 messages
    assert wire["cache_control"][0]["range"]["unit"] == "message"


def test_fork_is_not_emitted_by_this_harness(monkeypatch) -> None:
    """Documented gap: deepagents only produces spawn lineage (no fork source)."""
    monkeypatch.setattr(middleware_mod, "ensure_config", lambda: _config(thread_id="t"))
    mw = AgentHintsMiddleware(agent_type="a")
    wire = _wire(_Request([HumanMessage(content="go")]), mw)
    assert "derivation" not in wire["session"]
