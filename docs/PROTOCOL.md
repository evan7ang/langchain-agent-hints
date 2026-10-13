# Agent Hint protocol — wire specification

This document specifies the Agent Hint wire contract implemented by
`langchain-agent-hints`. Section references (`§N`) in the code and tests point
here.

The protocol implements an "agent signals, engine decides" pattern: the agent
framework describes **how it is running** on every LLM request; the serving
layer (gateway, router, inference engine) consumes these signals as advisory
scheduling inputs. Nothing in this protocol is model-visible: hints are
transport-side metadata and agents must function without a consuming
endpoint.

## 1. Overview

The protocol defines two channels:

1. **In-band** — one `AgentHint` object per main-line LLM request, carried as
   the `X-Agent-Hint` HTTP header (`schema.py` + `middleware.py`).
2. **Out-of-band** — `agent_status` **transition** events, reported as they
   happen to a pluggable sink (`status.py`).

## 2. Transport (in-band channel)

- Header name: `X-Agent-Hint`.
- Value: the AgentHint object serialized as compact UTF-8 JSON
  (`separators=(",", ":")`, `ensure_ascii=False`) and encoded as **unpadded
  base64url** (URL-safe alphabet, no `=`, no `+`/`/`), matching JS
  `Buffer.toString("base64url")`.
- The model request body is untouched. The channel is header-only: the hint
  must never appear in the request body (no `agent_hints` / `nvext` body
  fields).
- Receivers must treat the header as optional and unknown-header-tolerant;
  absence of the header never changes request semantics.

### Worked example

```json
{"session":{"session_id":"s-1","agent_type":"my-agent","call_index":1,"epoch":0},
 "agent_status":{"state":"running","state_since_ms":1730000000000},
 "priority":{"level":3,"reason":"user_waiting"},
 "cache_control":[{"type":"pin","range":{"end":1}}]}
```

Compact JSON → UTF-8 bytes → unpadded base64url → `X-Agent-Hint` header value.

## 3. Schema

JSON encoding drops `null`/unset fields: **omission is the encoding for "not
applicable"** (e.g. root sessions carry no `parent_session_id`).

### 3.1 `session` (mandatory)

| Field | Type | Meaning |
| --- | --- | --- |
| `session_id` | string | Identity of the serving session. Root runs: the LangGraph `thread_id`. Subagent runs: `{thread_id}:{agent_name}`. |
| `agent_type` | string | Serving-side name of the agent (composition-specific; subagents report their own name). |
| `call_index` | int ≥ 1 | 1-based index of this model call within the session. Derived from the persistent message history (`1 + count(AIMessage)`), so it survives process restarts. |
| `epoch` | int ≥ 0 | Context-generation counter of the session. Bumped when the message prefix is rewritten (history shrinks or the first message changes — the summarization signal). In-process fact (restarts reset it). |
| `parent_session_id` | string | Present only on child (subagent) requests; the parent's session id. |
| `derivation` | object | Present only with `parent_session_id`. See 3.2. |

### 3.2 `session.derivation`

| Field | Type | Meaning |
| --- | --- | --- |
| `relation` | `fork` \| `spawn` \| `handoff` | How the child context came to exist. Subagent delegation reports `spawn`. |
| `join` | `sync` \| `async` | Whether the parent blocks on the child. Protocol default: `sync`. Foreground-blocking subagents (deepagents `task`) report `sync` explicitly. |
| `fork_point` | int | Message index the child forked from. **v1: never emitted** (requires a message-index contract). |

### 3.3 `agent_status` (mandatory on the hint)

| Field | Type | Meaning |
| --- | --- | --- |
| `state` | `running` \| `tool` \| `pause` \| `suspend` \| `finalize` | Lifecycle state snapshot. On the in-band hint this is always `running` (a request exists only while the agent generates). |
| `state_since_ms` | epoch ms | Instant the current state began. On the hint: the instant the previous tool result landed (wall clock at request preparation as the fallback). |
| `reason` | string | Optional reason code (see §4 events). |
| `tools` | array | Optional `[{name, call_id?, child_session_id?}]` — what a `tool`-state agent is waiting on. |

### 3.4 `priority` / `strict_priority`

`{level: int, reason: string}`. Levels follow the protocol examples: `3` =
interactive work waiting on the user (`user_waiting`); sync-joined children
sit on the parent's critical path (`critical_path`).

- `priority` — the agent's own serving priority.
- `strict_priority` — **reserved for a trusted control plane; ordinary agents
  never emit it.**

### 3.5 `cache_control`

Array (or single object) of `{type: "pin" | "ephemeral", ttl_s?: int,
range?: {start?: int, end?: int, unit?: "message" | "token"}}`, declaring
cache intent over message ranges. The library default pins the system prompt
(range `[0, 1)`) — a declaration of the byte-stable-prefix guarantee a
composition already makes.

### 3.6 `prediction` (v1: never emitted)

Serving-side estimates: `latency_ms`, `remaining_calls`, `output_tokens`,
`interval_ms`, `remaining_time_ms`, `expected_resume_s`,
`next_action: {state, tools?: [{name, expected_duration_s?}], prob?}`.
Defined for forward compatibility; filling it requires an
observation-statistics engine.

## 4. Emission rules (in-band)

1. **Identity is mandatory**: requests without a `thread_id` carry no hint.
2. **Main-line only**: only requests that flow through the middleware chain
   (`wrap_model_call` / `awrap_model_call`) are hinted. Auxiliary model calls
   that bypass the chain (e.g. a summarization middleware's own request) are
   never hinted.
3. `agent_status.state` on the hint is always `running`; `state_since_ms` is
   the tracked transition instant (fallback: request-preparation wall clock).
4. Root requests omit lineage fields entirely; child requests carry
   `parent_session_id` + `derivation` with `join` explicit per §3.2.
5. The header merge is additive: existing `extra_headers` set by upstream
   middleware are preserved; `X-Agent-Hint` is added/updated last.

## 5. Out-of-band status events

Event shape: `{"seq": N, "session": {"session_id": ...}, "agent_status":
{state, state_since_ms, reason?, tools?}}` (§3.3 field semantics; `state` here
takes all lifecycle values).

Transition rules:

| Transition | Emitted when |
| --- | --- |
| `running` | A model request starts (also: a reject/respond resume re-enters the model directly). Repeats are suppressed — every model call and tool completion would otherwise re-report it. |
| `pause` (`reason: human_approval`) | A model response contains a tool call gated for human approval, just before the suspension begins. |
| `tool` (`reason: await_tool`) | A tool call begins; `tools` lists `{name, call_id}`. |
| `running` (from `tool`) | The session's **last** outstanding tool completes (parallel tool batches report running only after the last result). |
| `finalize` | The graph run actually completes. An approval pause suspends mid-run without finalize; the resumed run finalizes exactly once. |

Delivery contract: **best-effort**. The sink retries transient failures and
then drops the event with a warning — a broken status channel must not take
the agent run down. Per-session ordering is guaranteed by the sink's
serialized per-session `seq`; cross-process resume tolerance comes from
`state_since_ms` timestamps and idempotent `running` transitions.

## 6. v1 boundaries (never emitted)

- `prediction` (§3.6) — requires an observation-statistics engine.
- `strict_priority` (§3.4) — reserved for a trusted control plane.
- `derivation.fork_point` (§3.2) — requires a message-index contract.
