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
  "cache_control": [{"type": "pin", "range": {"end": 1}}]
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
| `agent_status` | request-time lifecycle snapshot (always `running`) | middleware observation |
| `priority` | level + audit reason (default `{level: 3, reason: user_waiting}`; subagents: `critical_path`) | constructor options |
| `cache_control` | pin/ephemeral intent over message ranges (default: pin the system prompt `[0, 1)`) | constructor options |
| `prediction`, `strict_priority` | defined by the protocol, **never emitted** in v1 | — |

Full wire contract: [`docs/PROTOCOL.md`](docs/PROTOCOL.md).

Key emission rules:

- No `thread_id` → no hint, no status events (identity is mandatory).
- `call_index` is derived from the checkpointer-persisted message history
  (`1 + count(AIMessage)`), so it survives process restarts with zero state.
- `epoch` bumps when the message prefix is rewritten (summarization signal).
- Only requests flowing through `wrap_model_call` are hinted; auxiliary
  model calls that bypass the middleware chain are not.

## Status events

`AgentStatusEventsMiddleware` reports transitions as they happen, in protocol
shape `{"seq": N, "session": {...}, "agent_status": {...}}`:

| Hook | Event | Dedup |
| --- | --- | --- |
| `wrap_model_call` entry | `running` | only if last state ≠ running |
| `wrap_model_call` response | `pause` (`human_approval`, gated tools listed) | every gated response |
| `after_model` | `running` | only if last state ≠ running |
| `wrap_tool_call` entry | `tool` (`await_tool`, name + call_id) | every tool call |
| `wrap_tool_call` exit | `running` | only when the last outstanding tool finishes |
| `after_agent` | `finalize` | fires only on real completion (an approval pause suspends without finalize) |

Pass `interrupt_tools=INTERRUPT_ON.keys()` (the human-approval gate names) so
gated tool calls surface as `pause` events. Sinks implement the one-method
`StatusEventSink` protocol; `JsonlStatusEventSink` persists to a JSONL file
with per-session sequence numbers and retries transient failures. The sink
interface is synchronous — async hooks call it inline, so use a buffered sink
for slow transports.

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
- `prediction` block: serving-side estimates from observation statistics.

## License

[Mulan Permissive Software License, Version 2 (Mulan PSL v2)](LICENSE)
