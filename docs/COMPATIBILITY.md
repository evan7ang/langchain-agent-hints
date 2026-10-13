# Compatibility notes

Measured runtime facts the implementation relies on, plus known limitations.
Verified against `langchain 1.4.1`, `langchain-core 1.6.4`, `deepagents 0.7.15`
(Python), LangGraph memory checkpointing (`MemorySaver`), Python 3.11–3.12.

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

## Known limitations (v1)

- **Same-name concurrent subagents collide** on the session id
  (`{thread_id}:{agent_name}`): events and epochs of two concurrent runs of
  the same named subagent interleave.
- **`epoch` is a heuristic**: it detects context rewrites via the message
  prefix fingerprint (count + first-message hash). A rewrite that preserves
  both is not detected.
- **`call_index` can dip** after a summarization rewrite compacts past
  AIMessages away; the concurrent `epoch` bump signals the rewrite.
- **False-positive `pause`** is possible when an interrupt config uses a
  `when` predicate that declines the interrupt after the response already
  passed the static gate-name check (the library checks names only).
- **Sinks are synchronous**: async hooks call the sink inline. Use a buffered
  sink for slow transports.
- **Non-OpenAI-compatible models**: the hint travels in bind kwargs
  (`extra_headers`); models/SDKs that do not support per-request headers
  simply ignore them — the agent run is never affected.
