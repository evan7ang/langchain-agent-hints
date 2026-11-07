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
- ``call_index`` is a session-local high-water counter seeded from the
  persistent message history (``1 + count(AIMessage)``) on first sight and
  incremented per hinted request; a compaction rewrite must not rewind it;
- ``epoch`` stays a per-session in-process fact: bumped when the message
  prefix is rewritten (count shrinks or the first message changes).

Real value derivation (v0.3.0):

- ``priority`` — ``"auto"`` classifies the request trigger: a human message is
  user-intervened (``3/user_waiting``), an approval resume (shared
  ``RunSignals``) is the same window, a tool-result continuation and a
  subagent run are ``1/critical_path`` (self-driven), and ``run_mode="batch"``
  reports ``1/background``. Explicit ``PriorityHint`` values still win.
- ``cache_control`` — ``"auto"`` rebuilds the declaration per the
  cache-control guide: one ``ephemeral`` entry, explicit TTL
  (``clamp(gap*2, 60, 3600)`` from the next-tool estimate / last interval /
  300 s fallback), ``range.end`` = logical input count (``[system?] +
  messages``), and no declaration for a <128-token prefix, a rewritten
  prefix (epoch bump), or a changed model/tool set.
- ``agent_status.state_since_ms`` — the status middleware's tracked
  transition instant via ``RunSignals`` (request wall clock as the fallback).

Observation (``prediction``) stays in-plugin: this middleware records
model-call latency/output tokens, tool durations and model→tool/finalize
transitions through its own hooks. ``strict_priority`` is only emitted when a
trusted control plane injects it.

Failure semantics: hint construction is deterministic over config facts, so
``wrap_model_call`` has no swallow-and-continue path — a bug here fails
loudly rather than silently degrading every request. Ops opt-out: drop this
middleware from the composition (see README "Disabling") or pass
``priority=None`` / ``cache_control=None`` / an ``ObservationOptions`` store
with measurement disabled.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import ensure_config

from langchain_agent_hints.observation import ObservationOptions, ObservationStore
from langchain_agent_hints.policy import (
    build_cache_control,
    estimate_tokens,
    estimate_ttl_s,
    logical_message_count,
)
from langchain_agent_hints.schema import (
    HINT_HEADER,
    AgentState,
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
from langchain_agent_hints.signals import RunSignals

#: Policy sentinel: derive the value from the request/observations.
AUTO = "auto"

#: Context flag for auxiliary model calls (compaction summaries, titles, ...).
#: The RFC requires them to carry no hint and no lifecycle transition; an outer
#: middleware sets this around its auxiliary handler call.
_SUPPRESS_HINT: ContextVar[bool] = ContextVar("langchain_agent_hints_suppress", default=False)


def hint_suppressed() -> bool:
    """Whether the current model call is an auxiliary request."""
    return _SUPPRESS_HINT.get()


@contextmanager
def suppress_hints() -> Any:
    """Mark one model call as auxiliary (no hint, no status transition).

    Composition usage::

        from langchain_agent_hints import suppress_hints

        class CompactionMiddleware(AgentMiddleware):
            def wrap_model_call(self, request, handler):
                with suppress_hints():
                    summary = handler(request.override(messages=history))
                ...
    """
    token = _SUPPRESS_HINT.set(True)
    try:
        yield
    finally:
        _SUPPRESS_HINT.reset(token)


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
    """Model messages inside one ``wrap_model_call`` handler result."""
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


def _system_content(request: Any) -> str | None:
    system = getattr(request, "system_message", None)
    if system is None:
        return None
    content = getattr(system, "content", None)
    return content if isinstance(content, str) else str(content) if content is not None else ""


def _prefix_key(request: Any) -> str:
    """Model + tool-set + system-prompt identity (cache-prefix invalidation key)."""
    model = getattr(request.model, "model_name", None) or getattr(request.model, "model", None)
    model_id = str(model or type(request.model).__name__)
    tools = []
    for tool in getattr(request, "tools", None) or []:
        name = getattr(tool, "name", None)
        if name is None and isinstance(tool, dict):
            name = tool.get("name")
        tools.append(str(name or type(tool).__name__))
    system = _system_content(request) or ""
    payload = json.dumps([model_id, sorted(tools), system], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _prefix_tokens(request: Any) -> int:
    contents: list[Any] = [
        getattr(message, "content", None) for message in getattr(request, "messages", None) or []
    ]
    contents.append(_system_content(request))
    return estimate_tokens(contents)


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
    """Per-session in-process facts (epoch, call ordinal, cache-prefix key)."""

    __slots__ = ("call_index", "epoch", "fingerprint", "prefix_key")

    def __init__(self) -> None:
        self.call_index: int | None = None
        self.epoch = 0
        self.fingerprint: tuple[int, str] | None = None
        self.prefix_key: str | None = None


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
        priority: PriorityHint | None | str = AUTO,
        subagent_priority: PriorityHint | None | str = AUTO,
        cache_control: tuple[CacheControlHint, ...] | None | str = AUTO,
        strict_priority: PriorityHint | None = None,
        prediction: PredictionHint | None = None,
        observation: ObservationOptions | ObservationStore | None = None,
        signals: RunSignals | None = None,
        run_mode: str = "interactive",
        clock: Callable[[], int] | None = None,
    ) -> None:
        """Fix the composition-level hint policy.

        ``priority`` / ``subagent_priority``: ``"auto"`` derives the value
        from the request trigger (human message, approval resume, tool
        continuation, subagent run, ``run_mode="batch"``); an explicit
        ``PriorityHint`` pins it; ``None`` omits the field.

        ``cache_control``: ``"auto"`` builds the guide-compliant declaration
        per request; an explicit tuple pins it; ``None`` omits the field.

        ``signals``: share one ``RunSignals`` with ``AgentStatusEventsMiddleware``
        to get the tracked ``state_since_ms`` and approval-resume priority.

        ``strict_priority`` is control-plane-only (emitted only when injected).
        ``prediction`` supplies static per-field overrides for the measured
        estimates. ``observation`` is an ``ObservationOptions`` (engine config)
        or a shared ``ObservationStore``; measurement is on by default.
        """
        if run_mode not in ("interactive", "batch"):
            msg = f"run_mode must be 'interactive' or 'batch', got {run_mode!r}"
            raise ValueError(msg)
        for name, value in (
            ("priority", priority),
            ("subagent_priority", subagent_priority),
            ("cache_control", cache_control),
        ):
            if isinstance(value, str) and value != AUTO:
                msg = f"{name} must be {AUTO!r}, an explicit value, or None; got {value!r}"
                raise ValueError(msg)
        self._agent_type = agent_type
        self._priority_policy = priority
        self._subagent_priority_policy = subagent_priority
        self._cache_policy = cache_control
        self._strict_priority = strict_priority
        self._configured_prediction = prediction
        self._run_mode = run_mode
        if isinstance(observation, ObservationStore):
            self._observation = observation
        else:
            self._observation = ObservationStore(observation or ObservationOptions())
        self._signals = signals or RunSignals()
        self._clock = clock or (lambda: time.time_ns() // 1_000_000)
        self._lock = threading.Lock()
        self._tracks: dict[str, _SessionTrack] = {}

    @property
    def observation(self) -> ObservationStore:
        """The observation engine this middleware records into / reads from."""
        return self._observation

    @property
    def signals(self) -> RunSignals:
        """The shared run signals (mount the status middleware with the same)."""
        return self._signals

    def _agent_type_for(self, identity: SessionIdentity) -> str:
        if identity.is_subagent:
            return identity.agent_name or "subagent"
        return self._agent_type

    def _next_call_index(self, session_id: str, messages: list[AnyMessage]) -> int:
        """Session-local, monotonic call ordinal.

        Seeded from the persistent history on first sight, then incremented per
        hinted request. A summarization rewrite shrinks the history but must
        not rewind the ordinal observed by consumers (RFC: call_index only
        increases); the session track carries the high-water mark across the
        epoch bump.
        """
        with self._lock:
            track = self._tracks.setdefault(session_id, _SessionTrack())
            if track.call_index is None:
                track.call_index = _call_index(messages)
            else:
                track.call_index += 1
            return track.call_index

    def _take_epoch_slot(self, session_id: str, fingerprint: tuple[int, str]) -> tuple[int, bool]:
        """Update the session epoch; return ``(epoch, prefix_rewritten)``."""
        with self._lock:
            track = self._tracks.setdefault(session_id, _SessionTrack())
            previous = track.fingerprint
            rewritten = previous is not None and (
                fingerprint[0] < previous[0] or fingerprint[1] != previous[1]
            )
            if rewritten:
                track.epoch += 1
            track.fingerprint = fingerprint
            return track.epoch, rewritten

    def _take_prefix_key(self, session_id: str, key: str) -> bool:
        """Record the model/tool/system key; ``True`` when it changed."""
        with self._lock:
            track = self._tracks.setdefault(session_id, _SessionTrack())
            previous = track.prefix_key
            track.prefix_key = key
            return previous is not None and previous != key

    def _resolve_priority(
        self, identity: SessionIdentity, request: Any, *, resumed: bool
    ) -> PriorityHint | None:
        policy = self._subagent_priority_policy if identity.is_subagent else self._priority_policy
        if policy is None:
            return None
        if not isinstance(policy, str):  # explicit PriorityHint
            return policy
        if resumed:
            return PriorityHint(level=3, reason="user_waiting")
        if self._run_mode == "batch":
            return PriorityHint(level=1, reason="background")
        if identity.is_subagent:
            return PriorityHint(level=1, reason="critical_path")
        messages = getattr(request, "messages", None) or []
        if messages and isinstance(messages[-1], HumanMessage):
            return PriorityHint(level=3, reason="user_waiting")
        return PriorityHint(level=1, reason="critical_path")

    def _resolve_cache_control(
        self,
        identity: SessionIdentity,
        request: Any,
        *,
        prefix_rewritten: bool,
        agent_type: str,
    ) -> tuple[CacheControlHint, ...] | None:
        policy = self._cache_policy
        if policy is None:
            return None
        if not isinstance(policy, str):  # explicit tuple
            return policy
        if prefix_rewritten:
            return None
        if self._take_prefix_key(identity.session_id, _prefix_key(request)):
            return None
        logical = logical_message_count(
            list(getattr(request, "messages", None) or []),
            system_present=getattr(request, "system_message", None) is not None,
        )
        next_tool = self._observation.predicted_next_tool(agent_type)
        tool_expected = self._observation.tool_expected_s(agent_type, next_tool)
        ttl = estimate_ttl_s(
            tool_expected_s=tool_expected,
            interval_ms=self._observation.last_interval_ms(identity.session_id),
        )
        return build_cache_control(
            logical_count=logical,
            prefix_tokens=_prefix_tokens(request),
            ttl_s=ttl,
        )

    def _hinted_request(self, request: Any) -> Any | None:
        """Build the hint for one request and return the overridden request.

        ``None`` when the run has no session identity — the degradation rule
        (no identity, no hint), matching the protocol's mandatory ``session``.
        """
        if hint_suppressed():
            return None
        identity = resolve_session_identity()
        if identity is None:
            return None

        relation: DerivationRelation | None
        join: DerivationJoin | None
        if identity.is_subagent:
            parent_session_id = identity.thread_id
            relation = "spawn"
            join = "sync"
        else:
            parent_session_id = None
            relation = None
            join = None
        agent_type = self._agent_type_for(identity)

        now_ms = self._clock()
        call_index = self._next_call_index(identity.session_id, request.messages)
        self._observation.model_call_started(identity.session_id, now_ms=now_ms)
        measured = self._observation.prediction(
            identity.session_id, agent_type, call_index=call_index, now_ms=now_ms
        )
        prediction = _merge_prediction(measured, self._configured_prediction)
        resumed = self._signals.take_resumed(identity.session_id)
        priority = self._resolve_priority(identity, request, resumed=resumed)
        current = self._signals.current_state(identity.session_id)
        status_state: AgentState
        if resumed:
            status_state = "resume"
        elif current in ("pause", "suspend", "tool"):
            # A request made while the agent is parked (including a tool wait)
            # still reports the wait.
            status_state = "pause"
        else:
            status_state = "running"
        epoch, rewritten = self._take_epoch_slot(
            identity.session_id, _fingerprint(request.messages)
        )
        cache_control = self._resolve_cache_control(
            identity,
            request,
            prefix_rewritten=rewritten,
            agent_type=agent_type,
        )
        state_since_ms = self._signals.state_since_ms(identity.session_id) or now_ms

        hint = build_agent_hint(
            HintRequestFacts(
                session_id=identity.session_id,
                call_index=call_index,
                epoch=epoch,
                state_since_ms=state_since_ms,
                parent_session_id=parent_session_id,
                parent_relation=relation,
                parent_join=join,
            ),
            HintOptions(
                agent_type=agent_type,
                priority=priority,
                cache_control=cache_control,
            ),
            now_ms=now_ms,
            prediction=prediction,
            strict_priority=self._strict_priority,
            state=status_state,
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
        if identity is None:
            return None
        if self._observation.enabled:
            self._observation.run_finished(
                identity.session_id, self._agent_type_for(identity), now_ms=self._clock()
            )
        self._signals.finish(identity.session_id)
        return None

    def after_agent(self, state: Any, runtime: Any) -> None:
        return self._run_finished()

    async def aafter_agent(self, state: Any, runtime: Any) -> None:
        return self._run_finished()
