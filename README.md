# langchain-agent-hints

**English** | [简体中文](README.zh-CN.md)

[![CI](https://github.com/evan7ang/langchain-agent-hints/actions/workflows/ci.yml/badge.svg)](https://github.com/evan7ang/langchain-agent-hints/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/langchain-agent-hints)](https://pypi.org/project/langchain-agent-hints/)
[![License: MulanPSL2](https://img.shields.io/badge/License-MulanPSL2-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)

**Per-request serving-intent hints for LangChain agents, as middleware.**

`langchain-agent-hints` makes your agent announce *how it is running* to the
downstream inference layer. On every main-line model request it attaches an
**AgentHint** object — session identity, lifecycle state, priority, cache
control — as the base64url-encoded `X-Agent-Hint` HTTP header, and reports
state *transitions* (`tool` / `running` / `pause` / `finalize`) through an
out-of-band event sink. The serving side (gateway, router, inference engine)
consumes these signals for scheduling: session-aware KV-cache placement,
priority scheduling, approval-pending demotion, and so on.

**The agent signals; the engine decides.** The middleware is read-only with
respect to the model-visible request: it never touches messages or the system
prompt — it only extends `ModelRequest.model_settings` with the transport
header, which reaches OpenAI-compatible chat models as per-request
`extra_headers`.

## How it fits LangChain

Both middlewares subclass the public
[`AgentMiddleware`](https://reference.langchain.com/python/langchain/agents/middleware/types/AgentMiddleware)
base class, so they work anywhere user middleware is accepted: LangChain's
`create_agent`, and deepagents' `create_deep_agent` (where user middleware is
placed after the core stack). No harness is modified and no private APIs are
imported.

- `AgentHintsMiddleware` — builds one hint per `wrap_model_call` and injects
  the `X-Agent-Hint` header (async: `awrap_model_call`).
- `AgentStatusEventsMiddleware` — reports lifecycle transitions to a pluggable
  sink via `wrap_model_call` / `after_model` / `wrap_tool_call` /
  `after_agent` and their async counterparts.

Session identity comes from the LangGraph runtime config:
`configurable.thread_id` is the session id; deepagents subagent runs
(`ls_agent_type == "subagent"`) self-identify as
`{thread_id}:{agent_name}` with a `parent_session_id` back-reference and
`derivation: {relation: spawn, join: sync}` (deepagents task subagents are
foreground-blocking).

## Install

```bash
pip install langchain-agent-hints
```

Requires Python ≥ 3.11 and `langchain>=1.4,<2.0`. Source install also works:
`pip install git+https://github.com/evan7ang/langchain-agent-hints.git`.

## Quickstart

### Plain `create_agent`

```python
from pathlib import Path

from langchain.agents import create_agent

from langchain_agent_hints import (
    AgentHintsMiddleware,
    AgentStatusEventsMiddleware,
    JsonlStatusEventSink,
)

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[...],
    middleware=[
        AgentHintsMiddleware(agent_type="my-agent"),
        AgentStatusEventsMiddleware(
            sink=JsonlStatusEventSink(Path("status-events.jsonl")),
        ),
    ],
)

result = agent.invoke(
    {"messages": [("user", "Summarize this quarter.")]},
    config={"configurable": {"thread_id": "session-42"}},  # identity is mandatory
)
```

Requests then carry a header such as (base64url-decoded for readability):

```json
{
  "session": {"session_id": "session-42", "agent_type": "my-agent", "call_index": 3, "epoch": 0},
  "agent_status": {"state": "running", "state_since_ms": 1730000000000},
  "priority": {"level": 3, "reason": "user_waiting"},
  "cache_control": [
    {"type": "ephemeral", "ttl_s": 240,
     "range": {"end": 2, "unit": "message"}}
  ]
}
```

### deepagents

```python
from deepagents import create_deep_agent

from langchain_agent_hints import AgentHintsMiddleware, AgentStatusEventsMiddleware

agent = create_deep_agent(
    model="openai:gpt-4o-mini",
    tools=[...],
    middleware=[
        AgentHintsMiddleware(agent_type="my-agent"),
        AgentStatusEventsMiddleware(sink=my_sink, interrupt_tools=INTERRUPT_ON.keys()),
    ],
)
```

A runnable version lives in [`examples/deepagents_quickstart.py`](examples/deepagents_quickstart.py) —
it accepts `AGENT_HINTS_EXAMPLE_MODEL` (model string override) and
`AGENT_HINTS_STATUS_SINK` (event file), and works against any OpenAI-compatible
gateway via `OPENAI_BASE_URL`.

## What gets sent

| Block | Content | Source |
| --- | --- | --- |
| `session` | `session_id`, `agent_type`, `call_index`, `epoch`, optional `parent_session_id` + `derivation` | `thread_id`, constructor options, persistent message history |
| `agent_status` | wire snapshot: `running` by default, `resume` on the first request after an approval, `pause` if a request occurs while parked | middleware observation + shared `RunSignals` |
| `priority` | level + audit reason. Auto: human message / approval resume → `3/user_waiting`; tool continuation and subagent runs → `1/critical_path`; `run_mode="batch"` → `1/background` | request trigger + shared `RunSignals`; explicit `PriorityHint` overrides |
| `cache_control` | one `ephemeral` declaration with an explicit TTL (`clamp(gap*2, 60, 3600)`); `range.end` = logical input count (`[system?] + messages`) | per-request derivation from the cache-control guide (next-tool p90 → last interval → 300 s fallback; omitted below 128 tokens, on prefix rewrite, or on a model/tool-set change) |
| `prediction` | measured serving estimates: `latency_ms`, `output_tokens`, `interval_ms`, `remaining_calls`, `remaining_time_ms`, `next_action` | in-plugin observation engine (rolling p90); omitted until enough samples |
| `strict_priority` | reserved for a trusted control plane | constructor injection only; ordinary agents never set it |

Full wire contract: [`docs/PROTOCOL.md`](docs/PROTOCOL.md).

Key emission rules:

- No `thread_id` → no hint, no status events (identity is mandatory).
- `call_index` is derived from the checkpointer-persisted message history
  (`1 + count(AIMessage)`), so it survives process restarts with zero state.
- `epoch` bumps when the message prefix is rewritten (summarization signal).
- Only requests flowing through `wrap_model_call` are hinted; auxiliary
  model calls that bypass the middleware chain are not.
- `priority` and `cache_control` are derived from real run facts by default
  (`"auto"`); pass an explicit `PriorityHint` / tuple to pin them, or `None`
  to omit the field. `cache_control` is re-declared on every request (the
  consumer never renews TTL).

## Status events

`AgentStatusEventsMiddleware` reports transitions as they happen, in protocol
shape `{"seq": N, "session": {...}, "agent_status": {...}}`:

Wire states follow `AGENT_STATUS_STATE_DESIGN.md` v1.0: **four values**
(`running` / `pause` / `resume` / `finalize`). Internal `tool` folds into
`running` (the reason/tools stay on the event) and `suspend` folds into
`pause`; `resume` is the one-shot approval edge.

| Hook | Event (wire state) | Dedup |
| --- | --- | --- |
| `wrap_model_call` entry | `running` | only if last state ≠ running |
| model response with gated tools | `pause` (`human_approval`, gated tools listed) | every gated response |
| approval resume | `resume` (one-shot; the next header also reports `resume`) | once per resume |
| `wrap_tool_call` entry | `running` (`reason: await_tool`/`join_barrier`, tools listed) | every tool call |
| `wrap_tool_call` exit | `running` | only when the last outstanding tool finishes |
| suspend threshold (`suspend_after_s`) | `pause` (`reason: checkpoint`) | once while parked |
| `after_agent` | `finalize` | fires only on real completion (an approval pause suspends without finalize) |

With `subagent_tools={"task"}` (the default) a delegation call is reported as
`tool` with `reason="join_barrier"`, the child agent name (`agent:<type>`) and
`child_session_id` (`{thread_id}:{type}`). Pass `suspend_after_s=N` to report
`suspend` (`reason="checkpoint"`) when an approval pause outlives N seconds,
and `expected_resume_s=N` to attach the wait estimate to the pause event.

Sinks implement the one-method `StatusEventSink` protocol; `JsonlStatusEventSink`
persists to a JSONL file and `HttpStatusEventSink` posts to a gateway endpoint,
both with per-session sequence numbers and retry-then-drop semantics. The sink
interface is synchronous — async hooks call it inline, so use a buffered sink
for slow transports.

## Serving estimates (`prediction`)

Unlike Dify (where platform boundaries force an out-of-band hint service), the
LangChain hooks expose everything needed to measure in-process. The observation
engine records, per agent type, a bounded rolling window of:

- model-call latency (measured around the handler) and output tokens
  (`usage_metadata`),
- the inter-call interval (from the previous call's end to this call's start),
- calls per session and session duration (closed in `after_agent`),
- tool durations, and the model→tool / model→finalize transition counts
  (including which tool follows a model call).

When a hint is built, one value per metric is selected (nearest-rank quantile,
default p90) and emitted as a partial `prediction` object. A metric is omitted
until it has `min_samples` observations (default 3), so a fresh process sends
no estimates rather than invented ones.

```python
from langchain_agent_hints import AgentHintsMiddleware, ObservationOptions

AgentHintsMiddleware(
    agent_type="my-agent",
    observation=ObservationOptions(min_samples=5, quantile=0.9),
    # Optional static overrides (per field, explicit values win):
    # prediction=PredictionHint(output_tokens=512),
)
```

Measurement is advisory and in-process: `ObservationOptions(enabled=False)`
turns it off, and stats reset on restart unless `persist_path` points at a JSON
file for aggregate carry-over.

## Disabling

Drop the middlewares from the composition — hints and events are advisory
signals and are never required for the agent to function. A common
composition-level pattern is an environment kill switch:

```python
import os

from langchain_agent_hints import AgentHintsMiddleware, AgentStatusEventsMiddleware

if os.environ.get("AGENT_HINTS_DISABLE", "").strip().lower() not in {"1", "true", "yes"}:
    middleware = [
        AgentHintsMiddleware(agent_type="my-agent"),
        AgentStatusEventsMiddleware(
            sink=JsonlStatusEventSink(
                os.environ.get("AGENT_HINTS_STATUS_SINK", "status-events.jsonl")
            ),
        ),
    ]
else:
    middleware = []
```

## Compatibility

Verified against `langchain 1.4.1` / `langchain-core 1.6.4` /
`deepagents 0.7.15` / `langgraph` memory checkpointing. Known limitations and
measured hook-timing facts: [`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md).

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[test,lint]'
pytest            # keyless: fake models + a local capture stub
ruff check .
ruff format --check .
mypy              # strict, src only
```

CI runs the same suite on Python 3.11–3.13.

## Roadmap

- Publish to PyPI and list under the LangChain middleware integrations.
- Async-friendly (buffered / async) sink interface.
- Cross-process/cross-replica observation aggregation (the in-plugin engine is
  process-local today; `persist_path` carries aggregates across restarts).

## License

[Mulan Permissive Software License, Version 2 (Mulan PSL v2)](LICENSE)
