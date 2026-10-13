"""Agent status events — the out-of-band ``agent_status`` reporting channel.

The protocol's two channels (docs/PROTOCOL.md §4): LLM requests carry the
in-band ``agent_status`` snapshot (the hint header, ``middleware.py``), while
state *transitions* are reported as they happen. This module implements the
transition channel with public-API middleware hooks only — hook behavior
verified against deepagents 0.7.15 + langchain 1.4.1:

- ``wrap_model_call`` entry → ``running`` (run start, or a resume that did not
  go through a tool: reject/respond decisions re-enter the model directly);
- ``wrap_model_call`` response inspection → ``pause`` (``human_approval``):
  the model produced a tool call gated by human-in-the-loop approval.
  langchain's ``HumanInTheLoopMiddleware`` interrupts from its own dedicated
  ``after_model`` graph node — placed ahead of user middleware in deepagents'
  stack, so no user hook fires at the suspension instant itself; the gated
  tool call passes through this wrap frame first, making response inspection
  the closest observable point (emitted just before the pause begins). A
  ``when`` predicate on an interrupt config could still decline the
  interrupt — a false-positive pause is possible in that (optional) setup;
- ``after_model`` → ``running``: fires after HITL's node on the resumed run,
  so approve-resume transitions surface before the tool executes;
- ``wrap_tool_call`` → ``tool`` (``await_tool``, with ``name``/``call_id``)
  on entry, and ``running`` once the last outstanding tool of the session
  completes (protocol §5: parallel tools report running only after the last
  result). Gated tools never reach this hook while paused — no spurious
  ``tool`` event — and reach it after approval: the natural resume signal;
- ``after_agent`` → ``finalize``: LangGraph fires it only when the graph
  actually completes. An approval pause suspends mid-run without firing it,
  which is exactly the protocol's "the task is really over" semantics; after
  a resume the run completes and reports finalize once.

Session identity comes from ``resolve_session_identity()`` (same rules as the
hint header). Events are protocol §4 objects — ``{"session": {...},
"agent_status": {...}}`` — handed to the configured sink; ``prediction`` is
never attached (unchanged v1 boundary). Delivery is best-effort at the
transport boundary: the sink retries transient failures and then drops with
a warning — a broken status channel must not take the agent run down (the
events are advisory scheduling signals). Order per session is guaranteed by
serialized per-session sequence numbers in the sink; cross-process resume
tolerance comes from the ``state_since_ms`` timestamps and idempotent
``running`` transitions. The sink interface is synchronous; the async hooks
call it inline (file/network writes are short) — use a buffered sink if the
transport is slow.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage

from langchain_agent_hints.middleware import resolve_session_identity
from langchain_agent_hints.schema import AgentState, StatusTool


class StatusEventSink(Protocol):
    """Transport for one status event object (protocol §4 shape)."""

    def send(self, event: dict[str, Any]) -> None: ...


class _DiscardSink:
    """Default no-op sink: events are built but not persisted."""

    def send(self, event: dict[str, Any]) -> None:  # noqa: ARG002
        return None


class JsonlStatusEventSink:
    """Append status events to a JSONL file (the local "gateway absent" sink).

    One JSON object per line — the protocol event plus a per-session ``seq``
    (assigned under the sink lock, so the file is the ordered per-session
    record; concurrent tool hooks and parallel subagents serialize here).
    Writes retry a few times before dropping the event with a warning (see
    module docstring for the delivery contract).
    """

    def __init__(self, path: Path | str, *, attempts: int = 3) -> None:
        self._path = Path(path)
        self._attempts = attempts
        self._lock = threading.Lock()
        self._seqs: dict[str, int] = {}
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def send(self, event: dict[str, Any]) -> None:
        session_id = str((event.get("session") or {}).get("session_id", ""))
        with self._lock:
            seq = self._seqs.get(session_id, 0) + 1
            self._seqs[session_id] = seq
            record = {"seq": seq, **event}
            line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
            last_error: Exception | None = None
            for _ in range(self._attempts):
                try:
                    with self._path.open("a", encoding="utf-8") as fh:
                        fh.write(line)
                    return
                except OSError as error:  # transient (fs latency, rotation, ...)
                    last_error = error
            print(
                f"langchain-agent-hints: status event dropped after {self._attempts} "
                f"write attempts to {self._path}: {last_error}",
                file=sys.stderr,
            )


def status_event(
    session_id: str,
    state: AgentState,
    *,
    reason: str | None = None,
    tools: tuple[StatusTool, ...] | None = None,
    now_ms: int,
) -> dict[str, Any]:
    """Build one protocol §4 status event object (``session`` + ``agent_status``)."""
    event: dict[str, Any] = {"session": {"session_id": session_id}}
    agent_status: dict[str, Any] = {"state": state, "state_since_ms": now_ms}
    if reason is not None:
        agent_status["reason"] = reason
    if tools:
        agent_status["tools"] = _status_tool_entries(tools)
    event["agent_status"] = agent_status
    return event


def _status_tool_entries(tools: tuple[StatusTool, ...]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for tool in tools:
        entry: dict[str, Any] = {"name": tool.name}
        if tool.call_id is not None:
            entry["call_id"] = tool.call_id
        if tool.child_session_id is not None:
            entry["child_session_id"] = tool.child_session_id
        entries.append(entry)
    return entries


@dataclass(frozen=True)
class _SessionState:
    """Per-session in-process transition bookkeeping."""

    last_state: AgentState | None = None
    outstanding_tools: int = 0


_UNSET = _SessionState()


class AgentStatusEventsMiddleware(AgentMiddleware):
    """Report agent state transitions to the configured sink.

    Mounted after the hint middleware in a composition; with the default
    discard sink it only maintains the transition bookkeeping (no observable
    effect). Pass a ``JsonlStatusEventSink`` (or your own ``StatusEventSink``)
    to persist the event stream.
    """

    name = "agent_status_events"

    def __init__(
        self,
        *,
        sink: StatusEventSink | None = None,
        interrupt_tools: frozenset[str] | set[str] = frozenset(),
        clock: Callable[[], int] | None = None,
    ) -> None:
        """``interrupt_tools`` are the human-approval gate keys (composition
        passes them); a model response calling one of them is reported as the
        ``human_approval`` pause. ``clock`` is injectable for tests."""
        self._sink = sink or _DiscardSink()
        self._interrupt_tools = frozenset(interrupt_tools)
        self._clock = clock or (lambda: time.time_ns() // 1_000_000)
        self._lock = threading.Lock()
        self._sessions: dict[str, _SessionState] = {}

    # -- transition machinery ------------------------------------------------

    def _transition(
        self,
        session_id: str,
        state: AgentState,
        *,
        reason: str | None = None,
        tools: tuple[StatusTool, ...] | None = None,
        dedupe_running: bool = False,
    ) -> None:
        """Emit one transition unless it is a redundant ``running`` repeat.

        ``running`` repeats carry no transition information (every model call
        and tool completion would otherwise re-report it); every other state
        always emits — state changes are the point of this channel.
        """
        with self._lock:
            track = self._sessions.get(session_id, _UNSET)
            if dedupe_running and state == "running" and track.last_state == "running":
                return
            self._sessions[session_id] = _SessionState(
                last_state=state,
                outstanding_tools=track.outstanding_tools,
            )
        self._sink.send(
            status_event(
                session_id,
                state,
                reason=reason,
                tools=tools,
                now_ms=self._clock(),
            )
        )

    def _begin_tool(self, session_id: str, tool: StatusTool) -> None:
        """Record one in-flight tool and emit its ``await_tool`` event atomically."""
        with self._lock:
            track = self._sessions.get(session_id, _UNSET)
            self._sessions[session_id] = _SessionState(
                last_state="tool",
                outstanding_tools=track.outstanding_tools + 1,
            )
        self._sink.send(
            status_event(
                session_id,
                "tool",
                reason="await_tool",
                tools=(tool,),
                now_ms=self._clock(),
            )
        )

    def _end_tool(self, session_id: str) -> bool:
        """Close one in-flight tool; ``True`` when the session has none left."""
        with self._lock:
            track = self._sessions.get(session_id, _UNSET)
            outstanding = track.outstanding_tools - 1
            self._sessions[session_id] = _SessionState(
                last_state=track.last_state,
                outstanding_tools=outstanding,
            )
        return outstanding <= 0

    def _gated_tool_calls(self, response: Any) -> tuple[StatusTool, ...]:
        """Gated tool calls (name + call_id) inside one model response."""
        if not self._interrupt_tools:
            return ()
        tools: list[StatusTool] = []
        for message in _response_messages(response):
            if not isinstance(message, AIMessage):
                continue
            for call in message.tool_calls or []:
                if call.get("name") in self._interrupt_tools:
                    tools.append(
                        StatusTool(name=str(call.get("name")), call_id=str(call.get("id")))
                    )
        return tuple(tools)

    # -- sync hooks ----------------------------------------------------------

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        identity = resolve_session_identity()
        if identity is None:
            return handler(request)
        self._transition(identity.session_id, "running", dedupe_running=True)
        response = handler(request)
        gated = self._gated_tool_calls(response)
        if gated:
            self._transition(
                identity.session_id,
                "pause",
                reason="human_approval",
                tools=gated,
            )
        return response

    def after_model(self, state: Any, runtime: Any) -> None:
        identity = resolve_session_identity()
        if identity is None:
            return None
        # Runs right after HITL's after_model node: on an approval resume this
        # is where "back to running" surfaces (before the tool executes).
        self._transition(identity.session_id, "running", dedupe_running=True)
        return None

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        identity = resolve_session_identity()
        if identity is None:
            return handler(request)
        call = request.tool_call
        self._begin_tool(
            identity.session_id,
            StatusTool(name=str(call["name"]), call_id=str(call.get("id"))),
        )
        try:
            return handler(request)
        finally:
            # Protocol §5: report running only once the last outstanding tool
            # of the session completed (covers parallel tool batches).
            if self._end_tool(identity.session_id):
                self._transition(identity.session_id, "running")

    def after_agent(self, state: Any, runtime: Any) -> None:
        identity = resolve_session_identity()
        if identity is None:
            return None
        # Fires only on real completion: an approval pause suspends the graph
        # without reaching this hook, so no finalize is reported while an
        # approval is pending (and the resumed run finalizes exactly once).
        self._transition(identity.session_id, "finalize")
        return None

    # -- async hooks (same semantics, shared machinery) ----------------------

    async def awrap_model_call(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        identity = resolve_session_identity()
        if identity is None:
            return await handler(request)
        self._transition(identity.session_id, "running", dedupe_running=True)
        response = await handler(request)
        gated = self._gated_tool_calls(response)
        if gated:
            self._transition(
                identity.session_id,
                "pause",
                reason="human_approval",
                tools=gated,
            )
        return response

    async def aafter_model(self, state: Any, runtime: Any) -> None:
        identity = resolve_session_identity()
        if identity is None:
            return None
        self._transition(identity.session_id, "running", dedupe_running=True)
        return None

    async def awrap_tool_call(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        identity = resolve_session_identity()
        if identity is None:
            return await handler(request)
        call = request.tool_call
        self._begin_tool(
            identity.session_id,
            StatusTool(name=str(call["name"]), call_id=str(call.get("id"))),
        )
        try:
            return await handler(request)
        finally:
            if self._end_tool(identity.session_id):
                self._transition(identity.session_id, "running")

    async def aafter_agent(self, state: Any, runtime: Any) -> None:
        identity = resolve_session_identity()
        if identity is None:
            return None
        self._transition(identity.session_id, "finalize")
        return None


def _response_messages(response: Any) -> list[Any]:
    """Model messages inside one ``wrap_model_call`` handler result.

    The handler contract allows ``ModelResponse`` (``.result`` messages),
    a bare ``AIMessage``, or an extended response; only the message list is
    of interest here, everything else degrades to "no gated calls".
    """
    if isinstance(response, AIMessage):
        return [response]
    result = getattr(response, "result", None)
    if isinstance(result, list):
        return result
    return []
