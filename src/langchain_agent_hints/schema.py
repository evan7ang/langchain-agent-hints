"""AgentHint protocol schema — the pure layer of langchain-agent-hints.

Implements the Agent Hint wire contract (see ``docs/PROTOCOL.md``): the typed
schema, the pure ``build_agent_hint`` constructor, and the base64url
``X-Agent-Hint`` header codec. Stdlib only, no harness imports, so the wire
shape is testable without langchain.

Emission boundaries (docs/PROTOCOL.md §6):

- ``prediction`` is emitted when the caller supplies one (the observation
  engine in :mod:`langchain_agent_hints.observation` produces measured
  estimates); partial objects are valid — omission is the encoding for
  "no estimate".
- ``strict_priority`` is never derived from agent facts; it is only emitted
  when a trusted control plane injects it explicitly.
- ``derivation.fork_point`` is defined but never emitted (needs a message
  index contract).
"""

from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

#: Wire header carrying the base64url-encoded AgentHint object.
HINT_HEADER = "X-Agent-Hint"

#: Lifecycle states shared by ``agent_status.state`` and
#: ``prediction.next_action.state``.
#: Wire value set (AGENT_STATUS_STATE_DESIGN.md v1.0): ``tool`` and ``suspend``
#: are folded at the harness boundary, and ``resume`` marks the first request
#: after an approval pause.
AgentState = Literal["running", "pause", "resume", "finalize"]

#: ``session.state`` vocabulary: the per-request lifecycle summary. ``stop`` is
#: reserved for the out-of-band event channel (a finished turn has no closing
#: request) and is never emitted on the header today.
SessionState = Literal["start", "pause", "resume", "stop", "compact", "append"]

#: Internal harness states before the wire fold (``tool`` -> running,
#: ``suspend`` -> pause). Used by the status middleware only.
InternalAgentState = Literal["running", "tool", "pause", "suspend", "finalize"]

DerivationRelation = Literal["fork", "spawn", "handoff"]
DerivationJoin = Literal["sync", "async"]


@dataclass(frozen=True)
class DerivationHint:
    """``session.derivation``: how this context came to exist.

    ``join`` defaults to ``sync`` at protocol level and is omitted on the wire
    when unset; ``fork_point`` is v1-never-emitted (see module docstring).
    """

    relation: DerivationRelation
    join: DerivationJoin | None = None
    fork_point: int | None = None


@dataclass(frozen=True)
class SessionHint:
    """``session`` block: who is calling, from which lineage, at which call."""

    session_id: str
    call_index: int
    epoch: int
    #: Serving-side agent name. Optional: manage requests carry no agent type.
    agent_type: str | None = None
    state: SessionState = "append"
    parent_session_id: str | None = None
    derivation: DerivationHint | None = None


@dataclass(frozen=True)
class StatusTool:
    """One tool a ``tool``-state agent is waiting on (or a child agent)."""

    name: str
    call_id: str | None = None
    child_session_id: str | None = None


@dataclass(frozen=True)
class AgentStatus:
    """``agent_status`` block: the framework-reported lifecycle state."""

    state: AgentState
    state_since_ms: int
    reason: str | None = None
    tools: tuple[StatusTool, ...] | None = None


@dataclass(frozen=True)
class PriorityHint:
    """``priority`` / ``strict_priority`` block: level + audit reason."""

    level: int
    reason: str


@dataclass(frozen=True)
class PriorityControl:
    """``priority_control`` block: the canonical home of ``priority``."""

    priority: PriorityHint


@dataclass(frozen=True)
class ContextEdit:
    """One ``context_management.edits`` entry."""

    type: Literal["offload", "prefetch", "evict"]
    target: Literal["session"] = "session"


@dataclass(frozen=True)
class ContextManagement:
    """``context_management`` block — **manage requests only** (protocol §5.3).

    ``manage_request`` is the JSON boolean ``true`` and v1 carries exactly one
    explicit edit. Ordinary requests must omit the whole block.
    """

    manage_request: Literal[True] = True
    edits: tuple[ContextEdit, ...] = ()

    def __post_init__(self) -> None:
        if len(self.edits) != 1:
            msg = "context_management.edits must carry exactly one edit in v1"
            raise ValueError(msg)


@dataclass(frozen=True)
class CacheControlRange:
    """``cache_control.range``: effective scope of one cache directive."""

    start: int | None = None
    end: int | None = None
    unit: Literal["message", "token"] | None = None


@dataclass(frozen=True)
class CacheControlHint:
    """``cache_control`` block: one ephemeral declaration over the prefix.

    ``msg_offset`` is the 1-based effect point (equal to ``range.end``); the
    legacy ``range`` form stays embedded for consumers that still read it.
    """

    type: Literal["ephemeral"]
    ttl_s: int | None = None
    msg_offset: int | None = None
    range: CacheControlRange | None = None


#: Reference behavior from the protocol definition's own examples: interactive
#: work waiting on the user.
DEFAULT_PRIORITY = PriorityHint(level=3, reason="user_waiting")

#: Sync-joined children sit on the parent's critical path (protocol §4.2
#: example).
DEFAULT_SUBAGENT_PRIORITY = PriorityHint(level=3, reason="critical_path")

#: ``pin`` is retired: every declaration is an ``ephemeral`` retention window.
DEFAULT_CACHE_CONTROL: tuple[CacheControlHint, ...] = (
    CacheControlHint(type="ephemeral", range=CacheControlRange(end=1)),
)

#: The contracted priority domain (v2): ``3`` = user-intervened
#: (admission-exempt), ``1`` = everything else. The values are contract
#: anchors against the inference-side admission line, not tiers.
PRIORITY_LEVELS: tuple[int, ...] = (1, 3)


def normalize_priority_level(level: int) -> int:
    """Fold a priority level into the two-value domain ``{1, 3}``.

    Cross-harness contract: the Motor admission line is ``3``. v1 legacy
    ``{2, 4}`` map to the nearest value; any other integer folds by the
    admission line (``>= 3 -> 3``, else ``-> 1``) — admission outcomes are
    unchanged. Non-integers are returned untouched so a caller bug surfaces
    instead of being silently rewritten.
    """
    if not isinstance(level, int) or isinstance(level, bool):
        return level
    if level in PRIORITY_LEVELS:
        return level
    if level == 2:
        return 1
    if level == 4:
        return 3
    return 3 if level >= 3 else 1


_PRIORITY_REASONS = ("user_waiting", "critical_path", "background")


def _fold_priority(priority: PriorityHint | None) -> PriorityHint | None:
    """Return the priority folded to the contracted domain.

    Simplified contract: only ``user_waiting`` is level 3; every other reason
    is level 1. Audit reasons may carry a suffix (e.g.
    ``background: context_compaction``); the wire reason is the base word. A
    bare level still folds by the admission line so legacy callers keep their
    admission outcome.
    """
    if priority is None:
        return None
    raw_reason = str(priority.reason or "").strip()
    reason = raw_reason
    for known in _PRIORITY_REASONS:
        if raw_reason.startswith(known):
            reason = known
            break
    if reason:
        level = 3 if reason == "user_waiting" else 1
        if level == priority.level and reason == priority.reason:
            return priority
        return PriorityHint(level=level, reason=reason)
    level = normalize_priority_level(priority.level)
    if level == priority.level:
        return priority
    return PriorityHint(level=level, reason=priority.reason)


@dataclass(frozen=True)
class PredictedTool:
    """One tool predicted by ``prediction.next_action``."""

    name: str
    expected_duration_s: float | None = None


@dataclass(frozen=True)
class NextActionHint:
    """``prediction.next_action``: predicted next state of the agent."""

    state: AgentState
    tools: tuple[PredictedTool, ...] | None = None
    prob: float | None = None


@dataclass(frozen=True)
class PredictionHint:
    """``prediction`` block: serving-side estimates (v1: never emitted)."""

    latency_ms: int | None = None
    remaining_calls: int | None = None
    output_tokens: int | None = None
    interval_ms: int | None = None
    remaining_time_ms: int | None = None
    expected_resume_s: float | None = None
    next_action: NextActionHint | None = None


@dataclass(frozen=True)
class AgentHint:
    """The complete AgentHint object sent (encoded) as ``X-Agent-Hint``."""

    session: SessionHint
    # Manage requests only; ordinary requests omit the block (protocol §5.3).
    context_management: ContextManagement | None = None
    agent_status: AgentStatus | None = None
    priority: PriorityHint | None = None
    priority_control: PriorityControl | None = None
    strict_priority: PriorityHint | None = None
    cache_control: CacheControlHint | None = None
    prediction: PredictionHint | None = None


@dataclass(frozen=True)
class HintRequestFacts:
    """Runtime facts for one request, gathered by the middleware.

    Pure data: the middleware owns the observation; this layer only shapes
    the wire object.
    """

    session_id: str
    call_index: int
    epoch: int
    state_since_ms: int | None = None
    parent_session_id: str | None = None
    parent_relation: DerivationRelation | None = None
    parent_join: DerivationJoin | None = None


@dataclass(frozen=True)
class HintOptions:
    """Static per-composition behavior.

    ``agent_type`` is the serving-side name of the composed agent (no default:
    the identity is composition-specific by nature). The other fields have
    protocol defaults.
    """

    agent_type: str
    priority: PriorityHint | None = DEFAULT_PRIORITY
    cache_control: tuple[CacheControlHint, ...] | None = DEFAULT_CACHE_CONTROL


def _drop_none(value: Any) -> Any:
    """Recursively drop ``None`` fields from dataclass-derived structures.

    Omission is the protocol's encoding for "not applicable" (e.g. root
    sessions carry no ``parent_session_id``), so unset optionals never reach
    the wire. ``asdict`` keeps tuples as tuples while recursing into them —
    both are rendered as JSON arrays.
    """
    if isinstance(value, dict):
        return {key: _drop_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, (list, tuple)):
        return [_drop_none(item) for item in value]
    return value


def _session_state(call_index: int, has_parent: bool) -> SessionState:
    """Ordinary requests only carry ``start`` or ``append`` (protocol §5.2).

    ``start`` means a brand-new root context on its first accepted call; every
    other main-line request (tool continuations, compaction follow-ups, adopted
    sessions) is ``append``. Lifecycle verbs live on manage requests.
    """
    if call_index == 1 and not has_parent:
        return "start"
    return "append"


#: Fixed verb -> edit mapping (protocol §8.2).
MANAGE_EDITS: dict[str, Literal["offload", "prefetch", "evict"]] = {
    "pause": "offload",
    "resume": "prefetch",
    "stop": "evict",
    "compact": "evict",
}


def build_manage_hint(
    *,
    session_id: str,
    verb: Literal["pause", "resume", "stop", "compact"],
    call_index: int,
    epoch: int,
    parent_session_id: str | None = None,
) -> AgentHint:
    """Build one manage request hint (``session`` + ``context_management``).

    ``epoch`` is the epoch the edit applies to: a ``compact`` request carries
    the **old** epoch of the rewritten context (protocol §6.3); the following
    ordinary request carries ``epoch + 1``.
    """
    return AgentHint(
        session=SessionHint(
            session_id=session_id,
            call_index=call_index,
            epoch=epoch,
            state=verb,
            parent_session_id=parent_session_id,
        ),
        context_management=ContextManagement(edits=(ContextEdit(type=MANAGE_EDITS[verb]),)),
    )


def _fold_strict_priority(priority: PriorityHint | None) -> PriorityHint | None:
    """Control-plane ``strict_priority``: fold the level only, never the reason."""
    if priority is None:
        return None
    level = normalize_priority_level(priority.level)
    if level == priority.level:
        return priority
    return PriorityHint(level=level, reason=priority.reason)


def build_agent_hint(
    facts: HintRequestFacts,
    options: HintOptions,
    now_ms: int,
    *,
    prediction: PredictionHint | None = None,
    strict_priority: PriorityHint | None = None,
    state: AgentState = "running",
) -> AgentHint:
    """Build one AgentHint for a main-line (ordinary) model request.

    ``session.state`` is only ``start`` or ``append`` (protocol §5.2); the
    block ``context_management`` is omitted entirely and lifecycle verbs travel
    on manage requests (see :func:`build_manage_hint`).

    ``agent_status`` still records the request-time snapshot (a tolerated
    superset field): ``state`` defaults to ``running`` and the hint carries the
    tracked transition instant — the middleware passes the instant the previous
    tool result landed; ``now_ms`` is the fallback for calls without an
    observed instant.

    Root requests (no ``parent_session_id``) omit the lineage fields entirely;
    child requests carry ``parent_session_id`` + ``derivation`` — foreground
    subagents (e.g. deepagents ``task``) are blocking, so children report
    ``join="sync"`` explicitly (the protocol default is sync; making it
    explicit costs nothing and pins the semantics).

    ``prediction`` is the serving-side estimate block (measured by the
    observation engine or supplied by the caller); ``strict_priority`` is
    control-plane injected only. Both are omitted from the wire when ``None``.
    Priority levels are folded into the contracted ``{1, 3}`` domain here so
    every emission path shares one normalization point.
    """
    has_parent = facts.parent_session_id is not None
    derivation = (
        DerivationHint(
            relation=facts.parent_relation or "spawn",
            join=facts.parent_join,
        )
        if has_parent and facts.parent_relation is not None
        else None
    )
    session_state = _session_state(facts.call_index, has_parent)
    priority = _fold_priority(options.priority)
    cache_control = None
    if options.cache_control:
        first = options.cache_control[0]
        offset = first.msg_offset
        if offset is None and first.range is not None:
            offset = first.range.end
        cache_control = CacheControlHint(
            type=first.type,
            ttl_s=first.ttl_s,
            msg_offset=offset,
            range=first.range,
        )
    return AgentHint(
        session=SessionHint(
            session_id=facts.session_id,
            agent_type=options.agent_type,
            call_index=facts.call_index,
            epoch=facts.epoch,
            state=session_state,
            parent_session_id=facts.parent_session_id if has_parent else None,
            derivation=derivation,
        ),
        context_management=None,
        agent_status=AgentStatus(
            state=state,
            state_since_ms=facts.state_since_ms if facts.state_since_ms is not None else now_ms,
        ),
        priority=priority,
        priority_control=PriorityControl(priority=priority) if priority is not None else None,
        strict_priority=_fold_strict_priority(strict_priority),
        cache_control=cache_control,
        prediction=prediction,
    )


def hint_to_dict(hint: AgentHint) -> dict[str, Any]:
    """Render an AgentHint as the plain JSON-compatible dict (None-free).

    ``session.parent_session_id`` is the one key that survives as an explicit
    ``null``: the contract requires the key to exist on root sessions too.
    """
    rendered: dict[str, Any] = _drop_none(asdict(hint))
    session = rendered.get("session")
    if isinstance(session, dict) and "parent_session_id" not in session:
        session["parent_session_id"] = None
    return rendered


def encode_agent_hint(hint: AgentHint) -> str:
    """Serialize to compact UTF-8 JSON and encode as unpadded base64url.

    Matches the transport rule in docs/PROTOCOL.md §2 and the JS
    ``Buffer.toString("base64url")`` shape.
    """
    raw = json.dumps(hint_to_dict(hint), separators=(",", ":"), ensure_ascii=False)
    return base64.urlsafe_b64encode(raw.encode("utf-8")).rstrip(b"=").decode("ascii")


def decode_agent_hint(encoded: str) -> dict[str, Any]:
    """Inverse of :func:`encode_agent_hint` (verification/test helper)."""
    padded = encoded + "=" * (-len(encoded) % 4)
    raw = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
    decoded: dict[str, Any] = json.loads(raw)
    return decoded
