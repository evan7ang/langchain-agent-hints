# langchain-agent-hints

[English](README.md) | **简体中文**

[![CI](https://github.com/evan7ang/langchain-agent-hints/actions/workflows/ci.yml/badge.svg)](https://github.com/evan7ang/langchain-agent-hints/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)

**面向 LangChain Agent 的按请求服务意图提示，以中间件形式提供。**

`langchain-agent-hints` 让你的 Agent 向下游推理层"播报自己当前的运行方式"。每次主链模型请求都会附带一个 **AgentHint** 对象——会话身份、生命周期状态、优先级、缓存控制——以 base64url 编码的 `X-Agent-Hint` HTTP 头发出；同时通过带外事件 sink 上报状态**转换**（`tool` / `running` / `pause` / `finalize`）。服务端（网关、路由、推理引擎）消费这些信号用于调度：会话感知的 KV-cache 放置、优先级调度、审批挂起降级等。

**Agent 发信号，引擎做决策。** 该中间件对模型可见的请求是只读的：从不改动 messages 或 system prompt——只向 `ModelRequest.model_settings` 扩展传输头，经 OpenAI 兼容 chat 模型表现为按请求的 `extra_headers`。

## 在 LangChain 中的位置

两个中间件均子类化公开的
[`AgentMiddleware`](https://reference.langchain.com/python/langchain/agents/middleware/types/AgentMiddleware)
基类，因此任何接受用户中间件的地方都能用：LangChain 的 `create_agent`，以及 deepagents 的 `create_deep_agent`（用户中间件位于核心栈之后）。不修改任何 harness、不 import 任何私有 API。

- `AgentHintsMiddleware` — 在每次 `wrap_model_call` 中构造一条 hint，注入 `X-Agent-Hint` 头（异步：`awrap_model_call`）。
- `AgentStatusEventsMiddleware` — 经 `wrap_model_call` / `after_model` / `wrap_tool_call` / `after_agent`（及其异步对应钩子）向可插拔 sink 上报生命周期转换。

会话身份来自 LangGraph 运行时 config：`configurable.thread_id` 是会话 id；deepagents 子代理运行（`ls_agent_type == "subagent"`）自报为 `{thread_id}:{agent_name}`，并带 `parent_session_id` 回指与 `derivation: {relation: spawn, join: sync}`（deepagents task 子代理是前台阻塞语义）。

## 安装

```bash
pip install langchain-agent-hints
# 尚未发布 PyPI——在此之前可从 git 安装：
pip install git+https://github.com/evan7ang/langchain-agent-hints.git
```

要求 Python ≥ 3.11、`langchain>=1.4,<2.0`。

## 快速开始

### 原生 `create_agent`

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
    config={"configurable": {"thread_id": "session-42"}},  # 身份必传
)
```

此后每个请求都会携带类似如下的头（base64url 解码后便于阅读）：

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

可运行版本见 [`examples/deepagents_quickstart.py`](examples/deepagents_quickstart.py)——
支持 `AGENT_HINTS_EXAMPLE_MODEL`（覆盖模型字符串）与 `AGENT_HINTS_STATUS_SINK`（事件落盘路径），
经 `OPENAI_BASE_URL` 可对接任意 OpenAI 兼容网关。

## 发送什么

| 块 | 内容 | 来源 |
| --- | --- | --- |
| `session` | `session_id`、`agent_type`、`call_index`、`epoch`、可选 `parent_session_id` + `derivation` | `thread_id`、构造参数、持久化消息历史 |
| `agent_status` | 请求时刻的生命周期快照（恒为 `running`） | 中间件观测 |
| `priority` | 级别 + 审计原因（默认 `{level: 3, reason: user_waiting}`；子代理为 `critical_path`） | 构造参数 |
| `cache_control` | 消息区间的 pin/ephemeral 缓存意图（默认：pin 系统提示词 `[0, 1)`） | 构造参数 |
| `prediction`、`strict_priority` | 协议已定义，v1 **永不发送** | — |

完整 wire 契约：[`docs/PROTOCOL.md`](docs/PROTOCOL.md)（[中文版](docs/PROTOCOL.zh-CN.md)）。

关键发射规则：

- 无 `thread_id` → 不发 hint、不发状态事件（身份必传）。
- `call_index` 由检查点器持久化的消息历史派生（`1 + count(AIMessage)`），零状态即可在进程重启后存活。
- `epoch` 在消息前缀被重写时递增（summarization 信号）。
- 只有流经 `wrap_model_call` 的请求携带 hint；旁路模型调用（不经中间件链）不会发射。

## 状态事件

`AgentStatusEventsMiddleware` 在转换发生的当下上报，事件形如
`{"seq": N, "session": {...}, "agent_status": {...}}`：

| 钩子 | 事件 | 去重 |
| --- | --- | --- |
| `wrap_model_call` 入口 | `running` | 仅当上一状态 ≠ running |
| `wrap_model_call` 响应 | `pause`（`human_approval`，列出被门控的工具） | 每个门控响应都发 |
| `after_model` | `running` | 仅当上一状态 ≠ running |
| `wrap_tool_call` 入口 | `tool`（`await_tool`，name + call_id） | 每次工具调用 |
| `wrap_tool_call` 退出 | `running` | 仅当会话最后一个在途工具结束 |
| `after_agent` | `finalize` | 仅在真正完成时触发（审批挂起不触发 finalize） |

传入 `interrupt_tools=INTERRUPT_ON.keys()`（人工审批门控的工具名），被门控的工具调用就会以 `pause` 事件呈现。sink 实现单方法 `StatusEventSink` 协议；`JsonlStatusEventSink` 落盘 JSONL 文件（按会话序列号排序、瞬时失败重试）。sink 接口是同步的——异步钩子内联调用它，慢传输请使用带缓冲的 sink。

## 禁用

把中间件从组合中移除即可——hint 与事件只是建议性信号，Agent 功能从不依赖它们。常见的组合层做法是环境变量总开关：

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

## 兼容性

实测验证于 `langchain 1.4.1` / `langchain-core 1.6.4` / `deepagents 0.7.15` / LangGraph 内存检查点。已知限制与实测钩子时序事实：[`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md)（[中文版](docs/COMPATIBILITY.zh-CN.md)）。

## 开发

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[test,lint]'
pytest            # keyless：fake 模型 + 本地捕获 stub
ruff check .
ruff format --check .
mypy              # strict，仅 src
```

CI 在 Python 3.11–3.13 上运行同一套检查。

## 路线图

- 发布 PyPI，并进入 LangChain middleware 集成列表。
- 异步友好（带缓冲/异步）的 sink 接口。
- `prediction` 块：基于观测统计的服务端预估。

## 许可证

[MIT](LICENSE)
