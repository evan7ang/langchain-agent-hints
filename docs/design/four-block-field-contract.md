# 设计：四块必需字段契约对齐（session.state / context_management / priority_control / cache_control 单对象）

> 状态：2026-11-25。跨 harness 字段规范 v1 落地：Dify / LangGraph / DSH / Pi / Hermes 五个存量插件
> 统一按同一份 `X-Agent-Hint` 四块契约下发。本文记录 LangGraph 侧的语义裁决与取值来源。
>
> 规范原文：`agent-hint-research/AgentHint-下发字段规范-v1.md`（本仓外）。
> 取代关系：本文取代 [PROTOCOL.md](../PROTOCOL.md) §3.1 中「根会话省略 `parent_session_id`」、
> §3.5 `cache_control` 为数组的表述；`agent_status` 四值折叠与二值 `priority` 的裁决不变。
>
> **2026-11-25 范围替代**：本文关于「`context_management` 恒存在」「`session.state` 六值（含
> `pause`/`resume`/`compact`）」「`manage_request` 恒为 `"false"`」的裁决已被
> [v1-ordinary-manage-split.md](v1-ordinary-manage-split.md) 替代（普通请求只发 `start`/`append`，
> `context_management` 只出现在独立管理请求）。本文其余裁决（`priority_control`、单对象
> `cache_control`、根会话显式 `null`）继续有效。

## 1. 背景

下游（Motor / vLLM）按四块解析请求头：`session` / `context_management` / `priority_control` / `cache_control`。
本仓此前只发六键（session / agent_status / priority / strict_priority / cache_control / prediction），其中：

- `session` 无 `state`：请求时刻的生命周期摘要只能从 `agent_status` 反推；
- 无 `context_management`：harness 无法表达本轮请求附带的上下文意图；
- `priority` 只有顶层形态，缺规范位置 `priority_control`；
- `cache_control` 是元组（JSON 数组）且无 `msg_offset`。

## 2. 字段裁决

### 2.1 `session.state`（六值，本期不实现 `stop`）

逐请求取**第一个命中**：

| 序 | 值 | 判据（本仓） |
|---|---|---|
| 1 | `pause` | `status_state == "pause"`（内部 `tool`/`suspend` 折叠为 `pause`，工具等待即 `pause`） |
| 2 | `compact` | `_take_epoch_slot` 返回 `rewritten=True`（消息前缀被重写） |
| 3 | `resume` | `status_state == "resume"`（放行后的首条请求，一次性） |
| 4 | `start` | `call_index == 1` 且无父会话 |
| 5 | `append` | 其余 |
| — | `stop` | 不实现：跑完一轮没有收尾请求，留给带外事件通道 |

`session.state` 读**内部状态机**，不是折叠后的 header 值；本仓 `tool → pause`，与裁决一致。
`session.state` 与 `agent_status.state` 不要求逐值相等（两套词表），但必须同源。

### 2.2 `context_management`（恒存在）

```jsonc
{ "manage_request": "false", "edits": [] }
```

`edits` 由 `session.state` 展开：

| state | edits |
|---|---|
| `pause` | `[{"type":"offload","target":"session"}]` |
| `resume` | `[{"type":"prefetch","target":"session"}]` |
| `compact` | `[{"type":"evict","target":"session"}]` |
| `start` / `append` | `[]` |

`manage_request` 恒 `"false"`：本库不发起纯管理请求，edits 只表达随本请求附带的上下文意图。

### 2.3 `priority_control`（规范位置）

`{"priority": {"level": n, "reason": s}}`，与顶层 `priority` **同值**，由同一份折叠后的
`PriorityHint` 复制。顶层 `priority` 保留为兼容镜像。`priority` 关闭（构造参数 `None`）时两块一起省略。

### 2.4 `cache_control`（单对象 + `msg_offset`）

由元组改为**单对象**：

```jsonc
{"type":"ephemeral","ttl_s":240,"msg_offset":12,"range":{"end":12,"unit":"message"}}
```

- `msg_offset` 为 1-based 生效点，等于 `range.end`（逻辑输入条数 `[system?] + messages`）；
- 内嵌 `range` 保留，兼容仍按旧形态读取的消费方；
- `type` 收敛为 `ephemeral`（`pin` 退役；默认值同步改为 `ephemeral`）；
- 抑制条件不变：前缀不足 128 token、前缀重写当轮、模型/工具集变化当轮。抑制时整块省略。
- 构造参数仍接受元组（历史兼容）；出口只取第一条按单对象发送。

### 2.5 `parent_session_id` 显式 `null`

`hint_to_dict` 在递归丢弃 `None` 后为根会话补回 `parent_session_id: null`，满足下游"键必须存在"的判定；
`derivation` 仍只在有父会话时出现。

## 3. 兼容性与影响

- **header 体积**：新增三块约 120–160 字节，仍远低于 4 KiB 预算。
- **消费方**：`cache_control` 由数组变对象是形状变更；顶层 `priority` 保留，读旧字段的消费方不受影响。
- **状态事件通道**：不变。`finalize` 语义、四值折叠不变。
- **降级规则**：无 `thread_id` 仍整条不发。

## 4. 验证

- `tests/test_schema.py`：wire 形状（含 `parent_session_id: null`、`context_management`、`priority_control`、单对象缓存）、默认值与 None 丢弃。
- `tests/test_field_sending.py` / `tests/test_middleware.py` / `tests/test_wire.py`：真实中间件链路与抓包断言。
- 全量基线：`pytest && ruff check . && ruff format --check . && mypy`。
