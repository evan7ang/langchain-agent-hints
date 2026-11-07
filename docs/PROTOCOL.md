# Agent Hint protocol — wire specification

**English** | [简体中文](PROTOCOL.zh-CN.md)

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
 "cache_control":[{"type":"ephemeral","ttl_s":240,"range":{"end":1,"unit":"message"}}]}
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
| `call_index` | int ≥ 1 | 1-based index of this model call within the session. Seeded from the persistent message history (`1 + count(AIMessage)`) and kept as a session high-water counter; a compaction rewrite does not rewind it. |
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
| `state` | `running` \| `pause` \| `resume` \| `finalize` | Wire lifecycle snapshot (4 values, `AGENT_STATUS_STATE_DESIGN.md` v1.0). Internal `tool`/`suspend` fold to `pause` at the harness boundary; `resume` marks the first request after a pause (approval **or** tool result, one-shot). |
| `state_since_ms` | epoch ms | Instant the current state began. On the hint: the tracked transition instant (request wall clock as the fallback). |
| `reason` | string | **Event channel only** — not emitted in the request header. |
| `tools` | array | **Event channel only** — `[{name, call_id?, child_session_id?}]` on the out-of-band event. |

### 3.4 `priority` / `strict_priority`

`{level: int, reason: string}`. Levels follow the protocol examples: `3` =
interactive work waiting on the user (`user_waiting`); sync-joined children
sit on the parent's critical path (`critical_path`).

- `priority` — the agent's own serving priority. The simplified contract is
  reason-driven: only `user_waiting` is `3`; `critical_path` and `background`
  are `1`. Audit reasons may carry a suffix (e.g.
  `background: context_compaction`); the wire reason is the base word. The
  reference implementation derives the value from the request trigger: a
  human message or an approval resume is `3/user_waiting`; a tool-result
  continuation or a subagent run is `1/critical_path`; batch runs are
  `1/background`. Explicit composition/service values are normalized by the
  same reason rule.
- `strict_priority` — **reserved for a trusted control plane; ordinary agents
  never emit it.** It is only serialized when the composition injects it
  explicitly (constructor argument), never derived from agent facts.

### 3.5 `cache_control`

Implements `CACHE_CONTROL_IMPLEMENTATION_GUIDE.md`. The field carries **one**
declaration:

```json
{"type": "ephemeral", "ttl_s": 240, "range": {"end": 12, "unit": "message"}}
```

- `type` — `ephemeral` only (`pin` is retired; consumers normalize it and warn).
- `ttl_s` — mandatory; the harness computes `clamp(gap * 2, 60, 3600)` where
  `gap` is the next-tool expected duration (p90), else the last inter-call
  interval, else a 300 s fallback. The consumer never renews it, so every
  request re-declares.
- `range.end` — the **logical input count** `[system?] + messages` (1-based,
  inclusive), never the append-only history length; `range.unit` is
  `message`.

The implementation omits the declaration when the prefix is below one engine
cache block (128 tokens), when the message prefix was rewritten (compaction or
branch: the epoch bump), or when the model/tool set changed that round.

### 3.6 `prediction` (measured in-process)

Serving-side estimates: `latency_ms`, `remaining_calls`, `output_tokens`,
`interval_ms`, `remaining_time_ms`, `expected_resume_s`,
`next_action: {state, tools?: [{name, expected_duration_s?}], prob?}`.

The library fills this block from an in-plugin observation engine (rolling
per-agent-type statistics, one value per metric — nearest-rank quantile,
default p90). The object is partial by design: a metric is omitted until it has
enough samples, and omission is the encoding for "no estimate". Static
per-field overrides supplied to the middleware win over measured values.

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
{state, state_since_ms, reason?, tools?}}` (§3.3 field semantics; `state` is
the same 4-value wire set; `reason`/`tools` are event-only).

Transition rules (wire states; internal `tool`/`suspend` are folded):

| Transition | Emitted when |
| --- | --- |
| `running` | A model request starts. Repeats are suppressed — every model call and tool completion would otherwise re-report it. |
| `pause` (`reason: await_tool`, `tools`) | A tool call begins (internal `tool` folds to `pause`; the tool name/call_id stay on the event). |
| `pause` (`reason: join_barrier`, `tools`) | A delegation tool (deepagents `task`) starts; `tools` names the child (`agent:<type>`) and carries `child_session_id`. |
| `resume` (from a tool) | The session's **last** outstanding tool completes (parallel batches report resume only after the last result). |
| `pause` (`reason: human_approval`) | A model response contains a tool call gated for human approval, just before the suspension begins. |
| `pause` (`reason: checkpoint`) | An approval pause outlives `suspend_after_s` (optional; off by default; internal `suspend` folds to `pause`). |
| `resume` | The transition out of `pause`/`suspend`/`tool` — the one-shot release edge. The next request header also reports `resume`. |
| `finalize` | The graph run actually completes. An approval pause suspends mid-run without finalize; the resumed run finalizes exactly once. |

A `pause` event may carry an estimate block: `"prediction":
{"expected_resume_s": N}` when the composition configures the expected wait.

Delivery contract: **best-effort**. The sink retries transient failures and
then drops the event with a warning — a broken status channel must not take
the agent run down. Per-session ordering is guaranteed by the sink's
serialized per-session `seq`; cross-process resume tolerance comes from
`state_since_ms` timestamps and idempotent `running` transitions.

## 6. Emission boundaries

- `strict_priority` (§3.4) — reserved for a trusted control plane; only
  emitted when explicitly injected.
- `derivation.fork_point` (§3.2) — requires a message-index contract.
- `prediction` measurements are process-local: a restart starts a fresh
  rolling window unless `persist_path` carries the aggregates.
