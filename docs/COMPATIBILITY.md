# Compatibility notes

**English** | [简体中文](COMPATIBILITY.zh-CN.md)

Measured runtime facts the implementation relies on, plus known limitations.
Verified against `langchain 1.4.1` / `langchain-core 1.6.4` / `deepagents 0.7.15`
(Python), LangGraph memory checkpointing (`MemorySaver`), Python 3.11–3.12;
real-endpoint smoke additionally verified against `langchain 1.4.5` /
`deepagents 0.7.24` / `langchain-openai 1.7.1` with a GLM reasoning model
behind an OpenAI-compatible gateway.

## Session identity sources

- `configurable.thread_id` — session identity (`ensure_config()` inside
  middleware hooks).
- `metadata.lc_agent_name` — agent instance name.
- `configurable["ls_agent_type"] == "subagent"` (also checked in `metadata`)
  — deepagents `subagents.py` tags child runs this way; the child session
  then self-identifies as `{thread_id}:{agent_name}`.

## Header passthrough chain (measured)

`wrap_model_call` → `ModelRequest.model_settings` →
`model.bind(**model_settings)` / `bind_tools(..., **model_settings)`
(`langchain.agents` factory) → ChatOpenAI request → OpenAI SDK
`create(extra_headers=…)`. The `X-Agent-Hint` header therefore reaches the
wire on OpenAI-compatible adapters; fake models simply ignore the extra bind
kwargs (keyless paths are unaffected).

The channel is SDK-level and API-shape-agnostic: with langchain-openai 1.7.x
the request may go through the OpenAI **Responses API** (`POST /v1/responses`)
instead of chat completions — verified end-to-end against a real
OpenAI-compatible gateway that the header rides both paths and is accepted
(HTTP 200) by the upstream.

## Hook timing (measured)

- `after_model` hooks are dedicated graph nodes
  (`graph.add_edge("model", "<mw>.after_model")`). Middleware listed later in
  the `middleware=` sequence runs innermost, and its `after_model` node fires
  first. deepagents places `HumanInTheLoopMiddleware` at the stack tail, so
  HITL's node interrupts **before** user middleware `after_model` hooks run.
- `wrap_model_call` runs right after the previous tools returned — the wall
  clock inside the hook is the closest available "running since" instant for
  `agent_status.state_since_ms`.
- `after_agent` fires only when the graph run actually completes; an approval
  pause (`interrupt()`) suspends without reaching it, and the resumed run
  finalizes exactly once.
- Gated tools never reach `wrap_tool_call` while paused (no spurious `tool`
  event) and reach it after approval — the natural resume signal.

## In-plugin observation (measured)

- `wrap_model_call` sees the handler result's `AIMessage.usage_metadata`
  (`input_tokens` / `output_tokens` / `total_tokens`) with `langchain-core`
  1.6.x and `langchain-openai` 1.6.x; models that omit usage metadata simply
  contribute no `output_tokens` samples.
- Latency is measured around the handler (queueing included);
  `wrap_tool_call` measures tool duration the same way.
- The interval sample is the gap between the previous call's end and this
  call's start (state-derived, no counter to lose).
- `remaining_calls` / `remaining_time_ms` are derived from completed sessions
  of the same `agent_type`; they are estimates, not orchestrator knowledge.
- `next_action` is learned from model→tool vs model→finalize transitions
  (majority vote, with the most common next tool); a cold process emits none.
- Aggregates are thread-safe and bounded (`max_samples`, default 256); the
  `persist_path` JSON carries them across process restarts.

## Wire state contract (v0.4.0)

`agent_status.state` follows `AGENT_STATUS_STATE_DESIGN.md` v1.0: four values
(`running` / `pause` / `resume` / `finalize`). Internal `tool` and `suspend`
fold to `pause` (tool events keep `reason`/`tools`; `suspend` uses
`reason: checkpoint`). The request header carries only `state` and
`state_since_ms`; reason/tools are event-channel fields. `resume` is emitted
once on the first request (and event) after a release — an approval pause or a
tool wait.

## Real-value derivation (v0.3.0)

- `ModelRequest.messages` **excludes** the system message; the system prompt is
  `request.system_message`. `range.end` therefore counts
  `len(messages) + (1 if system_message else 0)`.
- `priority` classification reads the last message's role: `HumanMessage` →
  `3/user_waiting`; a tool result → `1/critical_path` (self-driven). The
  approval-resume window is signalled by the status middleware through the
  shared `RunSignals` (one-shot), because the resumed request's last message is
  a tool result.
- `cache_control` TTL samples the **most common next tool**'s last-20 p90
  duration (per tool, not per call position — the guide's per-position
  approximation), falling back to the measured inter-call gap.
- A changed model/tool/system fingerprint suppresses the declaration for that
  round (cache key invalidation), as does an epoch bump.

## Known limitations (v1)

- **Same-name concurrent subagents collide** on the session id
  (`{thread_id}:{agent_name}`): events and epochs of two concurrent runs of
  the same named subagent interleave.
- **`epoch` is a heuristic**: it detects context rewrites via the message
  prefix fingerprint (count + first-message hash). A rewrite that preserves
  both is not detected.
- **`call_index` restart seed** is the persistent message history; if a
  summarization rewrite compacted past AIMessages before a process restart, the
  first post-restart seed can be lower than the pre-restart high-water mark
  (the in-process counter itself never rewinds).
- **False-positive `pause`** is possible when an interrupt config uses a
  `when` predicate that declines the interrupt after the response already
  passed the static gate-name check (the library checks names only).
- **Sinks are synchronous**: async hooks call the sink inline. Use a buffered
  sink for slow transports.
- **Non-OpenAI-compatible models**: the hint travels in bind kwargs
  (`extra_headers`); models/SDKs that do not support per-request headers
  simply ignore them — the agent run is never affected.
