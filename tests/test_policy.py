"""Cache-control policy: TTL, token estimate, logical count, declaration gates."""

from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_agent_hints.policy import (
    MIN_CACHE_TOKENS,
    build_cache_control,
    estimate_tokens,
    estimate_ttl_s,
    logical_message_count,
)


def test_ttl_priority_and_clamp() -> None:
    # Tool estimate wins over the interval (guide §4.2).
    assert estimate_ttl_s(tool_expected_s=120, interval_ms=60_000) == 240
    # Interval fallback, doubled.
    assert estimate_ttl_s(interval_ms=120_000) == 240
    # No signal: 300 s fallback, doubled.
    assert estimate_ttl_s() == 600
    # Clamp: below 60 -> 60; above 3600 -> 3600 (truncate, never drop).
    assert estimate_ttl_s(tool_expected_s=10) == 60
    assert estimate_ttl_s(tool_expected_s=7200) == 3600


def test_token_estimate_cjk_and_ascii() -> None:
    assert estimate_tokens("x" * 400) == 100
    assert estimate_tokens("中" * 100) == 100
    assert estimate_tokens(None) == 0


def test_logical_count_counts_system_and_skips_empty_assistant() -> None:
    messages = [HumanMessage(content="hi"), AIMessage(content="")]
    assert logical_message_count(messages, system_present=True) == 2
    assert logical_message_count(messages, system_present=False) == 1
    assert logical_message_count([], system_present=True) == 1
    # A real assistant turn is part of the logical input.
    assert (
        logical_message_count(
            [HumanMessage(content="a"), AIMessage(content="b")], system_present=False
        )
        == 2
    )


def test_declaration_gates() -> None:
    assert build_cache_control(logical_count=0, prefix_tokens=500, ttl_s=600) is None
    assert (
        build_cache_control(logical_count=2, prefix_tokens=MIN_CACHE_TOKENS - 1, ttl_s=600) is None
    )
    assert build_cache_control(logical_count=2, prefix_tokens=500, ttl_s=0) is None
    decl = build_cache_control(logical_count=2, prefix_tokens=500, ttl_s=10)
    assert decl is not None
    assert decl[0].type == "ephemeral"
    assert decl[0].ttl_s == 60
    assert decl[0].range is not None
    assert (decl[0].range.end, decl[0].range.unit) == (2, "message")


def _request(*, system: str, messages: list, tool_names=()):  # noqa: ANN001, ANN202
    tools = [SimpleNamespace(name=name) for name in tool_names]
    return SimpleNamespace(
        model=SimpleNamespace(model_name="m"),
        tools=tools,
        system_message=SimpleNamespace(content=system) if system else None,
        messages=messages,
        model_settings={},
    )


def test_cache_derivation_and_skip_rules() -> None:
    from langchain_agent_hints.middleware import AgentHintsMiddleware, SessionIdentity

    long_system = "cache block " * 120
    identity = SessionIdentity(thread_id="t", session_id="t", is_subagent=False, agent_name=None)
    mw = AgentHintsMiddleware(agent_type="a")
    long_prefix = _request(system=long_system, messages=[HumanMessage(content="go")])

    decl = mw._resolve_cache_control(identity, long_prefix, prefix_rewritten=False, agent_type="a")
    assert decl is not None
    assert decl[0].ttl_s == 600  # cold start: 300 s fallback * 2
    assert decl[0].range is not None and decl[0].range.end == 2

    # A changed model/tool/system key invalidates the prefix for that round.
    changed = _request(system=long_system + " v2", messages=[HumanMessage(content="go")])
    assert (
        mw._resolve_cache_control(identity, changed, prefix_rewritten=False, agent_type="a") is None
    )

    # A rewritten prefix (compaction) is never declared.
    fresh = AgentHintsMiddleware(agent_type="a")
    assert (
        fresh._resolve_cache_control(identity, long_prefix, prefix_rewritten=True, agent_type="a")
        is None
    )

    # Short prefixes stay below the engine block floor.
    short = _request(system="short", messages=[HumanMessage(content="hi")])
    assert (
        fresh._resolve_cache_control(identity, short, prefix_rewritten=False, agent_type="a")
        is None
    )


def test_priority_auto_classification() -> None:
    from langchain_agent_hints.middleware import AgentHintsMiddleware, SessionIdentity
    from langchain_agent_hints.schema import PriorityHint
    from langchain_agent_hints.signals import RunSignals

    root = SessionIdentity(thread_id="t", session_id="t", is_subagent=False, agent_name=None)
    child = SessionIdentity(thread_id="t", session_id="t:r", is_subagent=True, agent_name="r")
    mw = AgentHintsMiddleware(agent_type="a")
    human = SimpleNamespace(messages=[HumanMessage(content="go")])
    continuation = SimpleNamespace(
        messages=[HumanMessage(content="go"), ToolMessage(content="ok", tool_call_id="c")]
    )

    assert mw._resolve_priority(root, human) == PriorityHint(level=3, reason="user_waiting")
    assert mw._resolve_priority(root, continuation) == PriorityHint(level=1, reason="critical_path")
    assert mw._resolve_priority(child, human) == PriorityHint(level=1, reason="critical_path")

    batch = AgentHintsMiddleware(agent_type="a", run_mode="batch")
    assert batch._resolve_priority(root, human) == PriorityHint(level=1, reason="background")

    # Approval resume: the shared signal upgrades the next call to user-waiting.
    signals = RunSignals()
    resumed = AgentHintsMiddleware(agent_type="a", signals=signals)
    signals.observe_transition("t", "pause", "running", 42)
    assert resumed._resolve_priority(root, continuation) == PriorityHint(
        level=3, reason="user_waiting"
    )
    assert resumed._resolve_priority(root, continuation) == PriorityHint(
        level=1, reason="critical_path"
    )  # one-shot

    # Explicit policy still wins; None omits the field.
    pinned = AgentHintsMiddleware(
        agent_type="a", priority=PriorityHint(level=1, reason="background")
    )
    assert pinned._resolve_priority(root, human) == PriorityHint(level=1, reason="background")
    omitted = AgentHintsMiddleware(agent_type="a", priority=None)
    assert omitted._resolve_priority(root, human) is None
