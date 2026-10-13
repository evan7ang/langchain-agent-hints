"""Pure-layer tests: wire codec, None-dropping, and hint construction rules."""

from __future__ import annotations

import base64
import json

import pytest

from conftest import independent_decode
from langchain_agent_hints.schema import (
    DEFAULT_CACHE_CONTROL,
    DEFAULT_PRIORITY,
    DEFAULT_SUBAGENT_PRIORITY,
    CacheControlHint,
    CacheControlRange,
    HintOptions,
    HintRequestFacts,
    PriorityHint,
    build_agent_hint,
    encode_agent_hint,
    hint_to_dict,
)


def _root_facts() -> HintRequestFacts:
    return HintRequestFacts(session_id="thread-1", call_index=1, epoch=0, state_since_ms=42)


def test_encode_decode_roundtrip_is_none_free_wire() -> None:
    hint = build_agent_hint(_root_facts(), HintOptions(agent_type="my-agent"), now_ms=100)
    wire = independent_decode(encode_agent_hint(hint))

    assert wire == {
        "session": {
            "session_id": "thread-1",
            "agent_type": "my-agent",
            "call_index": 1,
            "epoch": 0,
        },
        "agent_status": {"state": "running", "state_since_ms": 42},
        "priority": {"level": 3, "reason": "user_waiting"},
        "cache_control": [{"type": "pin", "range": {"end": 1}}],
    }


def test_encode_is_unpadded_base64url() -> None:
    encoded = encode_agent_hint(build_agent_hint(_root_facts(), HintOptions(agent_type="a"), 1))
    assert "=" not in encoded
    assert "+" not in encoded and "/" not in encoded
    # Round-trips through the raw codec (not the package decoder).
    padded = encoded + "=" * (-len(encoded) % 4)
    raw = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
    assert json.loads(raw)["session"]["agent_type"] == "a"


def test_root_requests_omit_lineage_fields() -> None:
    hint = build_agent_hint(_root_facts(), HintOptions(agent_type="a"), now_ms=1)
    wire = hint_to_dict(hint)
    assert "parent_session_id" not in wire["session"]
    assert "derivation" not in wire["session"]


def test_child_requests_pin_sync_join() -> None:
    facts = HintRequestFacts(
        session_id="t:researcher",
        call_index=1,
        epoch=0,
        parent_session_id="t",
        parent_relation="spawn",
        parent_join="sync",
    )
    wire = hint_to_dict(build_agent_hint(facts, HintOptions(agent_type="researcher"), now_ms=5))
    assert wire["session"]["parent_session_id"] == "t"
    assert wire["session"]["derivation"] == {"relation": "spawn", "join": "sync"}


def test_state_since_ms_falls_back_to_now() -> None:
    facts = HintRequestFacts(session_id="s", call_index=1, epoch=0, state_since_ms=None)
    wire = hint_to_dict(build_agent_hint(facts, HintOptions(agent_type="a"), now_ms=1234))
    assert wire["agent_status"]["state_since_ms"] == 1234


def test_v1_never_emits_prediction_or_strict_priority() -> None:
    wire = hint_to_dict(build_agent_hint(_root_facts(), HintOptions(agent_type="a"), now_ms=1))
    assert "prediction" not in wire
    assert "strict_priority" not in wire


def test_option_overrides_reach_the_wire() -> None:
    options = HintOptions(
        agent_type="crit",
        priority=PriorityHint(level=1, reason="deadline"),
        cache_control=(CacheControlHint(type="ephemeral", ttl_s=600),),
    )
    wire = hint_to_dict(build_agent_hint(_root_facts(), options, now_ms=1))
    assert wire["priority"] == {"level": 1, "reason": "deadline"}
    assert wire["cache_control"] == [{"type": "ephemeral", "ttl_s": 600}]


def test_options_can_disable_priority_and_cache_control() -> None:
    options = HintOptions(agent_type="a", priority=None, cache_control=None)
    wire = hint_to_dict(build_agent_hint(_root_facts(), options, now_ms=1))
    assert "priority" not in wire
    assert "cache_control" not in wire


def test_agent_type_is_required() -> None:
    with pytest.raises(TypeError):
        HintOptions()  # type: ignore[call-arg]


def test_protocol_defaults() -> None:
    assert DEFAULT_PRIORITY.level == 3
    assert DEFAULT_SUBAGENT_PRIORITY.reason == "critical_path"
    assert (CacheControlHint(type="pin", range=CacheControlRange(end=1)),) == DEFAULT_CACHE_CONTROL


def test_range_fields_are_dropped_when_unset() -> None:
    hint_cache = (CacheControlHint(type="pin"),)
    wire = hint_to_dict(
        build_agent_hint(_root_facts(), HintOptions(agent_type="a", cache_control=hint_cache), 1)
    )
    assert wire["cache_control"] == [{"type": "pin"}]
