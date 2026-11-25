# Agent Hint 协议 — wire 规格

[English](PROTOCOL.md) | **简体中文**

本文档规定 `langchain-agent-hints` 实现的 Agent Hint wire 契约。代码与测试中的 `§N` 引用均指向本文。

该协议实现"Agent 发信号、引擎做决策"模式：agent 框架在每个 LLM 请求上描述**自己当前的运行方式**；服务层（网关、路由、推理引擎）将其作为建议性调度输入消费。协议中的任何内容都对模型不可见：hint 属于传输侧元数据，没有消费端时 Agent 功能完全不受影响。

## 1. 总览

协议有三种请求类型，外加一条带外状态通道：

1. **带内（普通）**——每条主链 LLM 请求携带一个 `AgentHint` 对象，经 `X-Agent-Hint` HTTP 头传输（`schema.py` + `middleware.py`）：`session` + `priority_control`（+ 合格时的 `cache_control`）。
2. **辅助请求**——压缩摘要、标题、记忆抽取：完全不带 hint。
3. **管理（带外生命周期）**——`pause` / `resume` / `stop` / `compact` 作为**独立 HTTP 请求**发送：`session` + `context_management`，body 为占位符（`manage.py`）。
4. **状态事件**——`agent_status` **转换**事件，在发生当下上报给可插拔 sink（`status.py`），属于被容忍的超集通道。

## 2. 传输（带内通道）

- 头名称：`X-Agent-Hint`。
- 值：AgentHint 对象序列化为紧凑 UTF-8 JSON（`separators=(",", ":")`、`ensure_ascii=False`），再编码为**无 padding 的 base64url**（URL 安全字母表，无 `=`、无 `+`/`/`），对齐 JS `Buffer.toString("base64url")`。
- 模型请求体不做任何改动。通道仅限头部：hint 绝不出现在请求体中（不含 `agent_hints` / `nvext` 体字段）。
- 接收端必须把该头视为可选、对未知头宽容；头缺失不改变请求语义。

### 示例

```json
{"session":{"session_id":"s-1","parent_session_id":null,"agent_type":"my-agent",
            "call_index":1,"epoch":0,"state":"start"},
 "agent_status":{"state":"running","state_since_ms":1730000000000},
 "priority":{"level":3,"reason":"user_waiting"},
 "priority_control":{"priority":{"level":3,"reason":"user_waiting"}},
 "cache_control":{"type":"ephemeral","ttl_s":240,"msg_offset":1,
                  "range":{"end":1,"unit":"message"}}}
```

紧凑 JSON → UTF-8 字节 → 无 padding base64url → `X-Agent-Hint` 头值。普通请求绝不携带 `context_management`；管理请求形态见 §3.5 与 §4.6。

紧凑 JSON → UTF-8 字节 → 无 padding base64url → `X-Agent-Hint` 头值。

## 3. Schema

JSON 编码丢弃 `null`/未设置字段：**"不适用"的编码方式就是省略**。唯一例外是 `session.parent_session_id`：根会话显式发送 `null`，保证消费方可以始终按键存在判定。

### 3.1 `session`（必传）

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `session_id` | string | 服务会话身份。根运行：LangGraph `thread_id`。子代理运行：`{thread_id}:{agent_name}`。 |
| `agent_type` | string | Agent 的服务侧名称（随组合而定；子代理上报自己的名称）。 |
| `call_index` | int ≥ 1 | 该会话内模型调用的 1 起始序号。首次从持久化消息历史播种（`1 + count(AIMessage)`），之后按会话高水位递增；压缩重写不回退。 |
| `epoch` | int ≥ 0 | 会话的上下文代际计数。消息前缀被重写时递增（历史缩短或首消息变化——summarization 信号）。进程内事实（重启清零）。 |
| `state` | `start` \| `append`（普通）/ `pause` \| `resume` \| `stop` \| `compact`（管理） | 逐请求生命周期：普通请求只有 `start`（根会话首个被接受的调用）或 `append`（其余全部）。生命周期动词只出现在管理请求（§4.6）。 |
| `parent_session_id` | string \| null | 根会话显式 `null`；子（子代理）请求为父会话 id。 |
| `derivation` | object | 仅在子请求出现。见 3.2。 |

### 3.2 `session.derivation`

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `relation` | `fork` \| `spawn` \| `handoff` | 子上下文的由来。子代理委派上报 `spawn`。 |
| `join` | `sync` \| `async` | 父是否阻塞等待子。协议默认 `sync`。前台阻塞子代理（deepagents `task`）显式上报 `sync`。 |
| `fork_point` | int | 子分叉自的消息序号。**v1：永不发送**（需要消息序号契约）。 |

### 3.3 `agent_status`（hint 上必传）

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `state` | `running` \| `pause` \| `resume` \| `finalize` | 四值线缆状态；内部 `tool`/`suspend` 在发送方折成 `pause`。`resume` 是放行后的首条请求（一次性）。 |
| `state_since_ms` | epoch 毫秒 | 当前状态的起始时刻（状态转换时刻；回退为请求准备时刻的墙钟）。 |
| `reason` | string | **仅事件通道**，请求头不发。 |
| `tools` | array | **仅事件通道**：`[{name, call_id?, child_session_id?}]`。 |

### 3.4 `priority` / `priority_control` / `strict_priority`

`{level: int, reason: string}`。级别遵循协议示例：`3` = 等待用户的交互型工作（`user_waiting`）；sync-join 的子代理位于父关键路径上（`critical_path`）。

- `priority`——Agent 自身的服务优先级。简化契约按 reason 定档：只有 `user_waiting` 是 `3`；`critical_path` / `background` 是 `1`。审计 reason 的后缀（如 `background: context_compaction`）归一成基础词。参考实现按请求触发源推导：人类消息/审批恢复 → `3/user_waiting`；工具续接或子 Agent → `1/critical_path`；批量运行 → `1/background`。显式配置/服务值也按同一 reason 规则归一。
- `priority_control`——同一份值的规范位置：`{"priority": {"level": n, "reason": s}}`。两块内容一致；顶层 `priority` 作为兼容镜像保留。优先级关闭时两块一起省略。
- `strict_priority`——**保留给可信控制面；普通 Agent 永不发射。** 仅当组合显式注入（构造参数）时才会序列化，绝不从 Agent 事实推导。

### 3.5 `context_management`（仅管理请求）

```json
{"manage_request": true, "edits": [{"type": "offload", "target": "session"}]}
```

普通请求**必须省略该块**。它只出现在管理请求上：`manage_request` 是 JSON 布尔 `true`，恰好携带一条 edit，`target` 为 `session`。edit 由动词推导：

| 动词 | `edits[].type` |
| --- | --- |
| `pause` | `offload` |
| `resume` | `prefetch` |
| `stop` | `evict` |
| `compact` | `evict` |

`compact` 管理请求携带**旧** epoch；紧随其后的普通请求携带 `epoch + 1`，并抑制一轮 cache 声明。`HttpManageRequestClient` 以占位 body（`{"model": ..., "messages": [], "stream": false}`）发送管理 hint，绝不把响应当作推理结果（协议 §8.3）。

### 3.6 `cache_control`

实现 `CACHE_CONTROL_IMPLEMENTATION_GUIDE.md`，只带**一条**声明，且为**单对象**：

```json
{"type": "ephemeral", "ttl_s": 240, "msg_offset": 12, "range": {"end": 12, "unit": "message"}}
```

- `type`——只用 `ephemeral`（`pin` 已退役，消费方会归一化并告警）。
- `ttl_s`——必填；`clamp(gap × 2, 60, 3600)`，`gap` 取下一工具预计耗时（p90），否则最近调用间隔，否则 300s 兜底。消费方不续期，故每请求都要重新声明。
- `msg_offset`——1-based 的声明生效消息序号，与 `range.end` 同值。
- `range.end`——**逻辑输入条数** `[system?] + messages`（1-based，含），绝不用 append-only 历史长度；`range.unit` 为 `message`。内嵌 `range` 保留给仍按旧形态读取的消费方。

前缀不足一个 128-token 缓存块、消息前缀被重写（压缩/分支，epoch 递增）或模型/工具集变化的当轮都不发声明。

### 3.7 `prediction`（插件内实测）

服务端预估：`latency_ms`、`remaining_calls`、`output_tokens`、`interval_ms`、`remaining_time_ms`、`expected_resume_s`、`next_action: {state, tools?: [{name, expected_duration_s?}], prob?}`。由插件内观测引擎填充（按 agent_type 滚动统计，每指标取单值——最近秩分位，默认 p90）。该对象允许部分字段：样本不足的指标省略，省略即"无预估"；构造参数中的静态值按字段覆盖实测值。

## 4. 发射规则（带内）

1. **身份必传**：无 `thread_id` 的请求不携带 hint。
2. **仅主链**：只有流经中间件链（`wrap_model_call` / `awrap_model_call`）的请求被 hint。旁路模型调用（如 summarization 中间件自身的请求）绝不发射。
3. hint 上 `agent_status.state` 恒为 `running`；`state_since_ms` 是被追踪的转换时刻（回退：请求准备墙钟）。
4. 根请求携带 `parent_session_id: null`、不携带 `derivation`；子请求携带 `parent_session_id` + `derivation`，`join` 按 §3.2 显式给出。
5. 头合并为增量式：上游中间件已设置的 `extra_headers` 全部保留；`X-Agent-Hint` 最后新增/更新。
6. `session.state` 只有 `start` / `append`；普通 hint 绝不包含 `context_management`。

### 4.6 管理请求（生命周期通道）

配置了 `HttpManageRequestClient` 时，中间件发送：

| 动词 | 本 harness 的触发时机 | `call_index` |
| --- | --- | --- |
| `resume` | 接手/恢复的会话首次普通请求之前（历史已存在），或组合调用 `send_manage("pause", ...)` 之后 | 即将发生的调用 |
| `compact` | 中间件观测到已提交的前缀重写之后 | 被重写的那一步（`call_index - 1`） |
| `pause` | **无自动触发**——组合在真实的"离开/空闲"信号上调用 `send_manage("pause", ...)` | 最近一次调用 |
| `stop` | **无自动触发**——组合在销毁/归档时调用 `send_manage("stop", ...)` | 最近一次调用 |

自动动词是 harness 能在请求边界观测到的那些。一轮结束**不是** pause，任务完成**不是** stop（对话线程在两者之后仍然存在），因此这两个动词保留为组合的显式决策。未配置管理客户端时，中间件即 L1：只发普通请求。

传输规则（协议 §8.3/§8.4）：

- body 为占位符 `{"model": ..., "messages": [], "stream": false}`；
- 一个 `X-Agent-Hint` 头携带 `session` + `context_management`；
- 会话内串行；下一条普通请求等待该锁；
- 只有 `resume` 对 408/429/5xx 重试；结果未知的变更类请求会将会话隔离，其后的普通请求以 `ManageRequestError` 失败；
- `HTTP 2xx` 只代表"已接受"，不代表"KV 操作完成"。

## 5. 带外状态事件

事件形状：`{"seq": N, "session": {"session_id": ...}, "agent_status": {state, state_since_ms, reason?, tools?}}`（字段语义见 §3.3；此处 `state` 取全部生命周期值）。

转换规则：

| 转换 | 触发时机 |
| --- | --- |
| `running` | 模型请求开始（也覆盖 reject/respond 恢复后直接回到模型）。重复被抑制——否则每次模型调用与工具完成都会重复上报。 |
| `pause`（`reason: human_approval`） | 模型响应包含被人审门控的工具调用，恰在挂起开始之前。 |
| `pause`（`reason: await_tool`） | 工具调用开始；`tools` 列出 `{name, call_id}`。 |
| `resume`（自 `tool`） | 会话**最后一个**在途工具完成（并行工具批只在最后一个 result 后报 resume）。 |
| `pause`（`reason: join_barrier`） | 委派工具（deepagents `task`）开始；`tools` 给出子 Agent 名（`agent:<type>`）与 `child_session_id`。父会话阻塞等待子会话。 |
| `pause`（`reason: checkpoint`） | 审批挂起超过配置的 `suspend_after_s` 阈值（可选，默认关闭）。 |
| `resume` | 从 `pause`/`suspend`/`tool` 放行（审批或工具结果）后的一次性边。 |
| `finalize` | 图运行真正完成。审批挂起中途暂停不触发 finalize；恢复后的运行恰好 finalize 一次。 |

`pause` 事件可携带预估块：配置了预期等待时附 `"prediction": {"expected_resume_s": N}`。投递契约：**尽力而为**。sink 对瞬时失败重试数次后丢弃并告警——状态通道故障不得拖垮 Agent 运行。会话内顺序由 sink 的串行化 per-session `seq` 保证；跨进程恢复容忍来自 `state_since_ms` 时间戳与幂等的 `running` 转换。

## 6. 发射边界

- `strict_priority`（§3.4）——保留给可信控制面；仅在显式注入时发送。
- `derivation.fork_point`（§3.2）——需要消息序号契约。
- `prediction` 的观测为进程内：重启后从新的滚动窗口开始，除非 `persist_path` 携带聚合。

## 7. 修订记录

- **2026-11-25——对齐 v1（解决三处硬冲突）。**
  1. `context_management` **仅管理请求携带**：普通请求整体省略该块，`manage_request` 为 JSON 布尔 `true`，且恰好一条 edit。
  2. 普通请求的 `session.state` 只有 `start` / `append`；生命周期动词（`pause` / `resume` / `stop` / `compact`）移到管理请求。
  3. 管理请求是独立 HTTP 请求，body 固定为占位符 `{"model": …, "messages": [], "stream": false}`（`manage.py`）。

  已提交的前缀重写现在表达为下一条普通请求的 `epoch + 1` 加一轮 cache 抑制；配置了管理客户端时，中间件还会发送对应的 `compact` 管理请求（携带旧 epoch）。
- **2026-11-25——四块必需字段契约。** 新增 `session.state`；根会话改为发送 `parent_session_id: null`；`priority_control` 镜像 `priority`；`cache_control` 改单对象并携带 `msg_offset`（`= range.end`）；`pin` 不再发送。按数组形态读取的消费方需改读对象形态；内嵌 `range` 保留原有定位信息。
