# Agent Hint 协议 — wire 规格

[English](PROTOCOL.md) | **简体中文**

本文档规定 `langchain-agent-hints` 实现的 Agent Hint wire 契约。代码与测试中的 `§N` 引用均指向本文。

该协议实现"Agent 发信号、引擎做决策"模式：agent 框架在每个 LLM 请求上描述**自己当前的运行方式**；服务层（网关、路由、推理引擎）将其作为建议性调度输入消费。协议中的任何内容都对模型不可见：hint 属于传输侧元数据，没有消费端时 Agent 功能完全不受影响。

## 1. 总览

协议定义两个通道：

1. **带内**——每条主链 LLM 请求携带一个 `AgentHint` 对象，经 `X-Agent-Hint` HTTP 头传输（`schema.py` + `middleware.py`）。
2. **带外**——`agent_status` **转换**事件，在发生当下上报给可插拔 sink（`status.py`）。

## 2. 传输（带内通道）

- 头名称：`X-Agent-Hint`。
- 值：AgentHint 对象序列化为紧凑 UTF-8 JSON（`separators=(",", ":")`、`ensure_ascii=False`），再编码为**无 padding 的 base64url**（URL 安全字母表，无 `=`、无 `+`/`/`），对齐 JS `Buffer.toString("base64url")`。
- 模型请求体不做任何改动。通道仅限头部：hint 绝不出现在请求体中（不含 `agent_hints` / `nvext` 体字段）。
- 接收端必须把该头视为可选、对未知头宽容；头缺失不改变请求语义。

### 示例

```json
{"session":{"session_id":"s-1","agent_type":"my-agent","call_index":1,"epoch":0},
 "agent_status":{"state":"running","state_since_ms":1730000000000},
 "priority":{"level":3,"reason":"user_waiting"},
 "cache_control":[{"type":"pin","range":{"end":1}}]}
```

紧凑 JSON → UTF-8 字节 → 无 padding base64url → `X-Agent-Hint` 头值。

## 3. Schema

JSON 编码丢弃 `null`/未设置字段：**"不适用"的编码方式就是省略**（如根会话不携带 `parent_session_id`）。

### 3.1 `session`（必传）

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `session_id` | string | 服务会话身份。根运行：LangGraph `thread_id`。子代理运行：`{thread_id}:{agent_name}`。 |
| `agent_type` | string | Agent 的服务侧名称（随组合而定；子代理上报自己的名称）。 |
| `call_index` | int ≥ 1 | 该会话内模型调用的 1 起始序号。由持久化消息历史派生（`1 + count(AIMessage)`），进程重启后存活。 |
| `epoch` | int ≥ 0 | 会话的上下文代际计数。消息前缀被重写时递增（历史缩短或首消息变化——summarization 信号）。进程内事实（重启清零）。 |
| `parent_session_id` | string | 仅子（子代理）请求携带；指向父会话 id。 |
| `derivation` | object | 与 `parent_session_id` 同时出现。见 3.2。 |

### 3.2 `session.derivation`

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `relation` | `fork` \| `spawn` \| `handoff` | 子上下文的由来。子代理委派上报 `spawn`。 |
| `join` | `sync` \| `async` | 父是否阻塞等待子。协议默认 `sync`。前台阻塞子代理（deepagents `task`）显式上报 `sync`。 |
| `fork_point` | int | 子分叉自的消息序号。**v1：永不发送**（需要消息序号契约）。 |

### 3.3 `agent_status`（hint 上必传）

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `state` | `running` \| `tool` \| `pause` \| `suspend` \| `finalize` | 生命周期状态快照。带内 hint 上恒为 `running`（请求只在生成期存在）。 |
| `state_since_ms` | epoch 毫秒 | 当前状态的起始时刻。hint 上：上一个工具结果落地的时刻（回退为请求准备时刻的墙钟）。 |
| `reason` | string | 可选原因码（见 §4 事件）。 |
| `tools` | array | 可选 `[{name, call_id?, child_session_id?}]`——`tool` 状态的 Agent 正在等待什么。 |

### 3.4 `priority` / `strict_priority`

`{level: int, reason: string}`。级别遵循协议示例：`3` = 等待用户的交互型工作（`user_waiting`）；sync-join 的子代理位于父关键路径上（`critical_path`）。

- `priority`——Agent 自身的服务优先级。
- `strict_priority`——**保留给可信控制面；普通 Agent 永不发射。**

### 3.5 `cache_control`

数组（或单对象）`{type: "pin" | "ephemeral", ttl_s?: int, range?: {start?: int, end?: int, unit?: "message" | "token"}}`，声明消息区间上的缓存意图。库默认 pin 系统提示词（区间 `[0, 1)`）——把组合本已保证的字节稳定前缀显式声明出来。

### 3.6 `prediction`（v1：永不发送）

服务端预估：`latency_ms`、`remaining_calls`、`output_tokens`、`interval_ms`、`remaining_time_ms`、`expected_resume_s`、`next_action: {state, tools?: [{name, expected_duration_s?}], prob?}`。为前向兼容而定义；填充它需要观测统计引擎。

## 4. 发射规则（带内）

1. **身份必传**：无 `thread_id` 的请求不携带 hint。
2. **仅主链**：只有流经中间件链（`wrap_model_call` / `awrap_model_call`）的请求被 hint。旁路模型调用（如 summarization 中间件自身的请求）绝不发射。
3. hint 上 `agent_status.state` 恒为 `running`；`state_since_ms` 是被追踪的转换时刻（回退：请求准备墙钟）。
4. 根请求完全省略世系字段；子请求携带 `parent_session_id` + `derivation`，`join` 按 §3.2 显式给出。
5. 头合并为增量式：上游中间件已设置的 `extra_headers` 全部保留；`X-Agent-Hint` 最后新增/更新。

## 5. 带外状态事件

事件形状：`{"seq": N, "session": {"session_id": ...}, "agent_status": {state, state_since_ms, reason?, tools?}}`（字段语义见 §3.3；此处 `state` 取全部生命周期值）。

转换规则：

| 转换 | 触发时机 |
| --- | --- |
| `running` | 模型请求开始（也覆盖 reject/respond 恢复后直接回到模型）。重复被抑制——否则每次模型调用与工具完成都会重复上报。 |
| `pause`（`reason: human_approval`） | 模型响应包含被人审门控的工具调用，恰在挂起开始之前。 |
| `tool`（`reason: await_tool`） | 工具调用开始；`tools` 列出 `{name, call_id}`。 |
| `running`（自 `tool`） | 会话**最后一个**在途工具完成（并行工具批只在最后一个 result 后报 running）。 |
| `finalize` | 图运行真正完成。审批挂起中途暂停不触发 finalize；恢复后的运行恰好 finalize 一次。 |

投递契约：**尽力而为**。sink 对瞬时失败重试数次后丢弃并告警——状态通道故障不得拖垮 Agent 运行。会话内顺序由 sink 的串行化 per-session `seq` 保证；跨进程恢复容忍来自 `state_since_ms` 时间戳与幂等的 `running` 转换。

## 6. v1 边界（永不发射）

- `prediction`（§3.6）——需要观测统计引擎。
- `strict_priority`（§3.4）——保留给可信控制面。
- `derivation.fork_point`（§3.2）——需要消息序号契约。
