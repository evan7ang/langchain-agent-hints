"""langchain-agent-hints — per-request serving-intent hints for LangChain agents.

One AgentHint per main-line model request, mounted by the composition as
user middleware (``AgentMiddleware.wrap_model_call``) and carried on the
wire as the base64url-encoded ``X-Agent-Hint`` HTTP header, plus an
out-of-band agent status event channel. Split into a pure protocol layer
(:mod:`langchain_agent_hints.schema`, stdlib only) and the harness mounts
(:mod:`langchain_agent_hints.middleware`, :mod:`langchain_agent_hints.status`,
langchain public API only) so the wire shape is testable without langchain.
"""

from __future__ import annotations

from langchain_agent_hints.middleware import (
    AgentHintsMiddleware,
    SessionIdentity,
    resolve_session_identity,
)
from langchain_agent_hints.observation import ObservationOptions, ObservationStore
from langchain_agent_hints.policy import (
    MIN_CACHE_TOKENS,
    TTL_MAX_S,
    TTL_MIN_S,
    build_cache_control,
    estimate_tokens,
    estimate_ttl_s,
    logical_message_count,
)
from langchain_agent_hints.schema import (
    DEFAULT_CACHE_CONTROL,
    DEFAULT_PRIORITY,
    DEFAULT_SUBAGENT_PRIORITY,
    HINT_HEADER,
    AgentHint,
    AgentStatus,
    CacheControlHint,
    CacheControlRange,
    DerivationHint,
    HintOptions,
    HintRequestFacts,
    NextActionHint,
    PredictedTool,
    PredictionHint,
    PriorityHint,
    SessionHint,
    StatusTool,
    build_agent_hint,
    decode_agent_hint,
    encode_agent_hint,
    hint_to_dict,
    normalize_priority_level,
)
from langchain_agent_hints.signals import RunSignals
from langchain_agent_hints.status import (
    AgentStatusEventsMiddleware,
    HttpStatusEventSink,
    JsonlStatusEventSink,
    StatusEventSink,
    status_event,
)

__version__ = "0.4.0"

__all__ = [
    "HINT_HEADER",
    "__version__",
    "AgentHint",
    "AgentHintsMiddleware",
    "AgentStatus",
    "AgentStatusEventsMiddleware",
    "CacheControlHint",
    "CacheControlRange",
    "DEFAULT_CACHE_CONTROL",
    "DEFAULT_PRIORITY",
    "DEFAULT_SUBAGENT_PRIORITY",
    "DerivationHint",
    "HintOptions",
    "HintRequestFacts",
    "HttpStatusEventSink",
    "JsonlStatusEventSink",
    "NextActionHint",
    "ObservationOptions",
    "ObservationStore",
    "PredictedTool",
    "RunSignals",
    "MIN_CACHE_TOKENS",
    "TTL_MAX_S",
    "TTL_MIN_S",
    "build_cache_control",
    "estimate_ttl_s",
    "estimate_tokens",
    "logical_message_count",
    "PredictionHint",
    "PriorityHint",
    "SessionHint",
    "SessionIdentity",
    "StatusEventSink",
    "StatusTool",
    "build_agent_hint",
    "decode_agent_hint",
    "encode_agent_hint",
    "hint_to_dict",
    "normalize_priority_level",
    "resolve_session_identity",
    "status_event",
]
