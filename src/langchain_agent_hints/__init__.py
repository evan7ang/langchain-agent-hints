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
    PredictionHint,
    PriorityHint,
    SessionHint,
    StatusTool,
    build_agent_hint,
    decode_agent_hint,
    encode_agent_hint,
    hint_to_dict,
)
from langchain_agent_hints.status import (
    AgentStatusEventsMiddleware,
    JsonlStatusEventSink,
    StatusEventSink,
    status_event,
)

__version__ = "0.1.0"

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
    "JsonlStatusEventSink",
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
    "resolve_session_identity",
    "status_event",
]
