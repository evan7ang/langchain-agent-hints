"""AgentHintsMiddleware — mount the AgentHint as the ``X-Agent-Hint`` header.

Per-request serving-intent hints for downstream inference scheduling,
injected through the standard LangChain middleware passthrough channel —
``ModelRequest.model_settings`` — which reaches OpenAI-compatible chat models
as per-request ``extra_headers`` (``model.bind(**model_settings)`` → SDK
``create(extra_headers=…)``).

Session facts come from the LangGraph runtime config (verified against
deepagents 0.7.15 / langchain 1.4.1):

- ``configurable.thread_id`` — session identity (``ensure_config()``);
- ``metadata.lc_agent_name`` — agent instance name;
- ``configurable["ls_agent_type"] == "subagent"`` (``subagents.py`` run
  tagging) — child-request detection; the child session then self-identifies
  as ``{thread_id}:{agent_name}`` pointing back at the parent thread.

Emission rules (docs/PROTOCOL.md §4/§5):

- requests without a ``thread_id`` carry no hint (identity is mandatory);
- auxiliary model calls that bypass the middleware chain (e.g.
  SummarizationMiddleware's own summarization request) are naturally never
  hinted — only requests that flow through ``wrap_model_call`` are;
- ``call_index`` is derived from the persistent message history
  (``1 + count(AIMessage in request.messages)``): the history lives in the
  checkpointer, so the counter survives process restarts with zero state —
  LangGraph has no per-call event stream to derive it from;
- ``epoch`` stays a per-session in-process fact: bumped when the message
  prefix is rewritten (count shrinks or the first message changes — the
  summarization signal).

Failure semantics: hint construction is deterministic over config facts, so
``wrap_model_call`` has no swallow-and-continue path — a bug here fails
loudly rather than silently degrading every request. Ops opt-out: drop this
middleware from the composition (see README "Disabling").
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.runnables import ensure_config

from langchain_agent_hints.schema import (
    DEFAULT_CACHE_CONTROL,
    DEFAULT_PRIORITY,
    DEFAULT_SUBAGENT_PRIORITY,
    HINT_HEADER,
    CacheControlHint,
    DerivationJoin,
    DerivationRelation,
    HintOptions,
    HintRequestFacts,
    PriorityHint,
    build_agent_hint,
    encode_agent_hint,
)


@dataclass(frozen=True)
class SessionIdentity:
    """Resolved session identity for the run surrounding one hook call."""

    thread_id: str
    session_id: str
    is_subagent: bool
    agent_name: str | None


def resolve_session_identity() -> SessionIdentity | None:
    """Resolve the session identity of the current run (``None`` without one).

    Root runs report the bare ``thread_id``; subagent runs (deepagents tags
    them via ``ls_agent_type``) report ``{thread_id}:{agent_name}`` with the
    parent back-reference carried separately by the hint builder. Same-name
    concurrent subagents would collide on this id (documented v1 limitation).
    """
    config = ensure_config()
    thread_id = (config.get("configurable") or {}).get("thread_id")
    if not thread_id:
        return None
    configurable = config.get("configurable") or {}
    metadata = config.get("metadata") or {}
    is_subagent = (
        configurable.get("ls_agent_type") == "subagent"
        or metadata.get("ls_agent_type") == "subagent"
    )
    agent_name = metadata.get("lc_agent_name")
    if is_subagent:
        return SessionIdentity(
            thread_id=str(thread_id),
            session_id=f"{thread_id}:{agent_name or 'subagent'}",
            is_subagent=True,
            agent_name=agent_name,
        )
    return SessionIdentity(
        thread_id=str(thread_id),
        session_id=str(thread_id),
        is_subagent=False,
        agent_name=agent_name,
    )


def _fingerprint(messages: list[AnyMessage]) -> tuple[int, str]:
    """Prefix fingerprint for epoch detection: message count + first content.

    A context rewrite (summarization) both shrinks the history and replaces
    the leading message; either change on an existing session bumps ``epoch``.
    """
    if not messages:
        return (0, "")
    first = messages[0].content
    text = (
        first
        if isinstance(first, str)
        else json.dumps(first, ensure_ascii=False, default=str, sort_keys=True)
    )
    return (len(messages), hashlib.sha256(text.encode("utf-8")).hexdigest())


def _call_index(messages: list[AnyMessage]) -> int:
    """State-derived ``call_index``: one prior AIMessage per completed call.

    The message history is checkpointer-persisted, so this survives process
    restarts (no in-process counter to lose). A summarization rewrite can
    compact past AIMessages away and lower the count — accepted, the epoch
    bump signals the rewrite.
    """
    return 1 + sum(1 for message in messages if isinstance(message, AIMessage))


class _SessionTrack:
    """Per-session in-process serving facts (``epoch`` only since v1.1)."""

    __slots__ = ("epoch", "fingerprint")

    def __init__(self) -> None:
        self.epoch = 0
        self.fingerprint: tuple[int, str] | None = None


class AgentHintsMiddleware(AgentMiddleware):
    """Attach one AgentHint per hinted model request as ``X-Agent-Hint``.

    Read-only with respect to the model-visible request: it never touches
    messages or the system prompt — it only extends ``model_settings`` with
    the transport header, merging into any ``extra_headers`` an upstream
    middleware already set.
    """

    name = "agent_hints"

    def __init__(
        self,
        *,
        agent_type: str,
        priority: PriorityHint | None = DEFAULT_PRIORITY,
        subagent_priority: PriorityHint | None = DEFAULT_SUBAGENT_PRIORITY,
        cache_control: tuple[CacheControlHint, ...] | None = DEFAULT_CACHE_CONTROL,
        clock: Callable[[], int] | None = None,
    ) -> None:
        """Fix the composition-level hint policy.

        ``agent_type`` is the serving-side name of the composed agent (no
        default: it is composition-specific). ``subagent_priority`` applies to
        child requests (protocol example: sync-joined children sit on the
        critical path); ``clock`` returns epoch milliseconds and is injectable
        for tests.
        """
        self._options = HintOptions(
            agent_type=agent_type, priority=priority, cache_control=cache_control
        )
        self._subagent_priority = subagent_priority
        self._clock = clock or (lambda: time.time_ns() // 1_000_000)
        self._lock = threading.Lock()
        self._tracks: dict[str, _SessionTrack] = {}

    def _take_epoch_slot(self, session_id: str, fingerprint: tuple[int, str]) -> int:
        """Bump the session epoch when the message prefix was rewritten.

        Serialized so concurrent subagents cannot interleave fingerprint
        updates for the same session id.
        """
        with self._lock:
            track = self._tracks.setdefault(session_id, _SessionTrack())
            previous = track.fingerprint
            if previous is not None and (
                fingerprint[0] < previous[0] or fingerprint[1] != previous[1]
            ):
                track.epoch += 1
            track.fingerprint = fingerprint
            return track.epoch

    def _hinted_request(self, request: Any) -> Any | None:
        """Build the hint for one request and return the overridden request.

        ``None`` when the run has no session identity — the degradation rule
        (no identity, no hint), matching the protocol's mandatory ``session``
        block.
        """
        identity = resolve_session_identity()
        if identity is None:
            return None

        relation: DerivationRelation | None
        join: DerivationJoin | None
        if identity.is_subagent:
            parent_session_id = identity.thread_id
            relation = "spawn"
            # deepagents task subagents are foreground-blocking: the parent
            # waits for the child result, so the join semantics are exactly
            # the protocol's "sync".
            join = "sync"
            agent_type = identity.agent_name or "subagent"
            priority = self._subagent_priority
        else:
            parent_session_id = None
            relation = None
            join = None
            agent_type = self._options.agent_type
            priority = self._options.priority

        epoch = self._take_epoch_slot(identity.session_id, _fingerprint(request.messages))
        hint = build_agent_hint(
            HintRequestFacts(
                session_id=identity.session_id,
                # Derived from the persistent message history (restart-safe).
                call_index=_call_index(request.messages),
                epoch=epoch,
                # wrap_model_call runs right after the tools returned, so the
                # wall clock is the closest available "running since" instant.
                state_since_ms=self._clock(),
                parent_session_id=parent_session_id,
                parent_relation=relation,
                parent_join=join,
            ),
            HintOptions(
                agent_type=agent_type,
                priority=priority,
                cache_control=self._options.cache_control,
            ),
            now_ms=self._clock(),
        )

        model_settings = dict(request.model_settings)
        existing_headers = model_settings.get("extra_headers") or {}
        model_settings["extra_headers"] = {
            **existing_headers,
            HINT_HEADER: encode_agent_hint(hint),
        }
        return request.override(model_settings=model_settings)

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        hinted = self._hinted_request(request)
        return handler(hinted if hinted is not None else request)

    async def awrap_model_call(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        hinted = self._hinted_request(request)
        return await handler(hinted if hinted is not None else request)
