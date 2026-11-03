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

Serving estimates (``prediction``) are measured in-plugin by
:class:`langchain_agent_hints.observation.ObservationStore`: this middleware
records model-call latency/output tokens, tool durations and model→tool/
finalize transitions through its own hooks, and reads the rolling aggregates
when it builds the next hint. ``strict_priority`` is only emitted when a
trusted control plane injects it (constructor argument); agent facts never
produce it. Priority levels are folded into the contracted ``{1, 3}`` domain
at the protocol layer.

Failure semantics: hint construction is deterministic over config facts, so
``wrap_model_call`` has no swallow-and-continue path — a bug here fails
loudly rather than silently degrading every request. Ops opt-out: drop this
middleware from the composition (see README "Disabling") or pass an
``ObservationOptions(enabled=False)`` store to disable measurement only.
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

from langchain_agent_hints.observation import ObservationOptions, ObservationStore
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
    PredictionHint,
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


def _response_messages(response: Any) -> list[Any]:
    """Model messages inside one ``wrap_model_call`` handler result.

    The handler contract allows ``ModelResponse`` (``.result`` messages),
    a bare ``AIMessage``, or an extended response; only the message list is
    of interest here, everything else degrades to an empty list.
    """
    if isinstance(response, AIMessage):
        return [response]
    result = getattr(response, "result", None)
    if isinstance(result, list):
        return result
    return []


def _response_facts(response: Any) -> tuple[int | None, bool]:
    """``(output_tokens, had_tool_calls)`` from one handler result."""
    output_tokens: int | None = None
    had_tool_calls = False
    for message in _response_messages(response):
        if not isinstance(message, AIMessage):
            continue
        usage = getattr(message, "usage_metadata", None)
        if isinstance(usage, dict):
            value = usage.get("output_tokens")
            if isinstance(value, int) and not isinstance(value, bool):
                output_tokens = value
        if message.tool_calls:
            had_tool_calls = True
    return output_tokens, had_tool_calls


def _merge_prediction(
    measured: PredictionHint | None, configured: PredictionHint | None
) -> PredictionHint | None:
    """Measured estimates merged with explicit caller overrides (override wins)."""
    if measured is None:
        return configured
    if configured is None:
        return measured
    return PredictionHint(
        latency_ms=(
            configured.latency_ms if configured.latency_ms is not None else measured.latency_ms
        ),
        remaining_calls=(
            configured.remaining_calls
            if configured.remaining_calls is not None
            else measured.remaining_calls
        ),
        output_tokens=(
            configured.output_tokens
            if configured.output_tokens is not None
            else measured.output_tokens
        ),
        interval_ms=(
            configured.interval_ms if configured.interval_ms is not None else measured.interval_ms
        ),
        remaining_time_ms=(
            configured.remaining_time_ms
            if configured.remaining_time_ms is not None
            else measured.remaining_time_ms
        ),
        expected_resume_s=(
            configured.expected_resume_s
            if configured.expected_resume_s is not None
            else measured.expected_resume_s
        ),
        next_action=(
            configured.next_action if configured.next_action is not None else measured.next_action
        ),
    )


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
        strict_priority: PriorityHint | None = None,
        prediction: PredictionHint | None = None,
        observation: ObservationOptions | ObservationStore | None = None,
        clock: Callable[[], int] | None = None,
    ) -> None:
        """Fix the composition-level hint policy.

        ``agent_type`` is the serving-side name of the composed agent (no
        default: it is composition-specific). ``subagent_priority`` applies to
        child requests (protocol example: sync-joined children sit on the
        critical path); ``clock`` returns epoch milliseconds and is injectable
        for tests.

        ``strict_priority`` is control-plane-only: ordinary agents never set
        it, so it is emitted only when the composition explicitly injects it.
        ``prediction`` supplies static overrides for the measured estimates
        (per-field: an explicit value wins over the observation engine).
        ``observation`` is an ``ObservationOptions`` (engine config) or a
        shared ``ObservationStore``; measurement is on by default.
        """
        self._options = HintOptions(
            agent_type=agent_type, priority=priority, cache_control=cache_control
        )
        self._subagent_priority = subagent_priority
        self._strict_priority = strict_priority
        self._configured_prediction = prediction
        if isinstance(observation, ObservationStore):
            self._observation = observation
        else:
            self._observation = ObservationStore(observation or ObservationOptions())
        self._clock = clock or (lambda: time.time_ns() // 1_000_000)
        self._lock = threading.Lock()
        self._tracks: dict[str, _SessionTrack] = {}

    @property
    def observation(self) -> ObservationStore:
        """The observation engine this middleware records into / reads from."""
        return self._observation

    def _agent_type_for(self, identity: SessionIdentity) -> str:
        if identity.is_subagent:
            return identity.agent_name or "subagent"
        return self._options.agent_type

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

        now_ms = self._clock()
        call_index = _call_index(request.messages)
        # Record the call start first: the measured inter-call gap feeds the
        # interval estimate for this very request.
        self._observation.model_call_started(identity.session_id, now_ms=now_ms)
        measured = self._observation.prediction(
            identity.session_id, agent_type, call_index=call_index, now_ms=now_ms
        )
        prediction = _merge_prediction(measured, self._configured_prediction)

        epoch = self._take_epoch_slot(identity.session_id, _fingerprint(request.messages))
        hint = build_agent_hint(
            HintRequestFacts(
                session_id=identity.session_id,
                # Derived from the persistent message history (restart-safe).
                call_index=call_index,
                epoch=epoch,
                # wrap_model_call runs right after the tools returned, so the
                # wall clock is the closest available "running since" instant.
                state_since_ms=now_ms,
                parent_session_id=parent_session_id,
                parent_relation=relation,
                parent_join=join,
            ),
            HintOptions(
                agent_type=agent_type,
                priority=priority,
                cache_control=self._options.cache_control,
            ),
            now_ms=now_ms,
            prediction=prediction,
            strict_priority=self._strict_priority,
        )

        model_settings = dict(request.model_settings)
        existing_headers = model_settings.get("extra_headers") or {}
        model_settings["extra_headers"] = {
            **existing_headers,
            HINT_HEADER: encode_agent_hint(hint),
        }
        return request.override(model_settings=model_settings)

    # -- observation of the surrounding run (feeds the next hint) ------------

    def _record_model_response(self, response: Any, started_at: int) -> None:
        identity = resolve_session_identity()
        if identity is None or not self._observation.enabled:
            return
        output_tokens, had_tool_calls = _response_facts(response)
        self._observation.model_call_finished(
            identity.session_id,
            self._agent_type_for(identity),
            latency_ms=max(0, self._clock() - started_at),
            output_tokens=output_tokens,
            had_tool_calls=had_tool_calls,
            now_ms=self._clock(),
        )

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        hinted = self._hinted_request(request)
        started_at = self._clock()
        response = handler(hinted if hinted is not None else request)
        self._record_model_response(response, started_at)
        return response

    async def awrap_model_call(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        hinted = self._hinted_request(request)
        started_at = self._clock()
        response = await handler(hinted if hinted is not None else request)
        self._record_model_response(response, started_at)
        return response

    def _tool_started(self, request: Any) -> tuple[SessionIdentity | None, str | None]:
        identity = resolve_session_identity()
        if identity is None or not self._observation.enabled:
            return None, None
        call = getattr(request, "tool_call", {}) or {}
        name = str(call.get("name") or "")
        self._observation.tool_started(identity.session_id, self._agent_type_for(identity), name)
        return identity, name

    def _tool_finished(self, identity: Any, name: str | None, started_at: int) -> None:
        if identity is None or name is None or not self._observation.enabled:
            return
        self._observation.tool_finished(
            self._agent_type_for(identity), name, max(0, self._clock() - started_at)
        )

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        identity, name = self._tool_started(request)
        started_at = self._clock()
        try:
            return handler(request)
        finally:
            self._tool_finished(identity, name, started_at)

    async def awrap_tool_call(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        identity, name = self._tool_started(request)
        started_at = self._clock()
        try:
            return await handler(request)
        finally:
            self._tool_finished(identity, name, started_at)

    def _run_finished(self) -> None:
        identity = resolve_session_identity()
        if identity is None or not self._observation.enabled:
            return None
        self._observation.run_finished(
            identity.session_id, self._agent_type_for(identity), now_ms=self._clock()
        )
        return None

    def after_agent(self, state: Any, runtime: Any) -> None:
        return self._run_finished()

    async def aafter_agent(self, state: Any, runtime: Any) -> None:
        return self._run_finished()
