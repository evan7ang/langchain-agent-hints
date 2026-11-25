# 设计：普通请求 / 管理请求分离（v1 对齐）

> 状态：2026-11-25。跨 harness 协议文档 `agent-hint-protocol.md` v1.0（2026-11-25）复位三处硬冲突：
> 本设计记录 LangGraph 侧的对齐裁决、新增接口与兼容性影响。
>
> 规范原文：`agent-hint-research/agent-hint-protocol.md`（本仓外）。
> 取代关系：本文替代 [four-block-field-contract.md](four-block-field-contract.md) 中
> 「`context_management` 恒存在」「`session.state` 六值合并表达」「`manage_request` 恒为 `"false"`」
> 三项裁决；其余裁决（`priority_control`、单对象 `cache_control`、根会话显式 `null`）不变。
> 规格侧同步见 [PROTOCOL.md](../PROTOCOL.md) §3.5 / §4.6 / §7。

## 1. 背景与冲突

旧实现把一个会话的生命周期动词折叠进普通请求：

1. `context_management` 恒存在，`manage_request` 是字符串 `"false"`；
2. `session.state` 取六值，`pause` / `compact` / `resume` 内联在普通请求里；
3. 管理意图（offload/prefetch/evict）没有独立请求，接收方无法区分"这轮请求顺带声明"与
   "纯管理变更"。

v1 协议的判定是**按块是否存在**：普通请求没有 `context_management`；有且
`manage_request === true` 才是管理请求；普通请求出现生命周期动词会被判非法并忽略整条 hint。
因此旧形态在严格接收方上会整条失效，必须改。

## 2. 裁决

### 2.1 普通请求只发 `start` / `append`

`session.state` 是普通请求的状态，只有两个值：

| 值 | 条件 |
| --- | --- |
| `start` | 根会话（无 parent）且 `call_index == 1` |
| `append` | 其余全部（工具续接、压缩后续、接手会话） |

生命周期动词（`pause` / `resume` / `stop` / `compact`）只属于管理请求。`agent_status`
作为被容忍的超集字段继续携带请求时刻快照（`running` / `pause` / `resume` / `finalize`），
与 `session.state` 解耦。

### 2.2 `context_management` 可选，管理请求用布尔 `true` + 单 edit

```python
ContextManagement(manage_request=True, edits=(ContextEdit(type="offload"),))
```

- `manage_request` 是 JSON 布尔 `true`（不是字符串）；
- v1 恰好一条 edit，`target` 只能是 `session`（`messages` / `tools` 留给后续版本）；
- 构造器校验：`edits` 长度必须为 1，否则 `ValueError`；
- 普通请求 `AgentHint.context_management is None`，编码后整块省略。

### 2.3 管理请求是独立 HTTP 请求

新增 `manage.py`：

- `HttpManageRequestClient(url, model, api_key?, timeout?, attempts?)`；
- body 固定 `{"model": <model>, "messages": [], "stream": false}`；
- 唯一载荷是 `X-Agent-Hint`（`session` + `context_management`）；
- 响应只当"已接受"，绝不解析为推理结果。

可靠性与隔离：

- 同一 `session_id` 的请求串行（每会话一把锁）；
- 只有 `resume` 对 408 / 429 / 5xx / 网络错误做有界重试，耗尽后返回 `False`（错过一次 prefetch 是建议性损失）；
- 变更类动词（`pause` / `stop` / `compact`）不重试，失败即把会话标记为隔离（结果未知）；
- 被隔离的会话，其后的普通请求在 `ensure_ready()` 处抛 `ManageRequestError`，避免在未知 KV 状态上继续推理；
- `ensure_ready()` 与在途管理请求共用同一把会话锁，构成主链屏障。

### 2.4 触发时机

中间件只在**请求边界**能观测到的事实上自动发管理请求：

| 动词 | 触发 |
| --- | --- |
| `resume` | 接手/恢复的会话首次普通请求之前（历史已存在），或 `send_manage("pause", …)` 之后的下一条请求 |
| `compact` | 观测到已提交的前缀重写之后；携带**旧** epoch，`call_index` 为重写前那一步 |
| `pause` / `stop` | 无自动触发；组合通过 `send_manage(verb, session_id)` 在真实离开/归档事件上发送 |

"一轮对话结束"不是 `pause`，"任务完成"不是 `stop`——会话在其后依然存活，这两个动词属于上层
生命周期决策（协议 §8.1 / §12.5）。未配置管理客户端时，库即 L1：只发普通请求。

### 2.5 `epoch` 的表达

- 普通请求不变：前缀指纹变化时 `epoch += 1`，当轮抑制 `cache_control`；
- 配置了管理客户端时，中间件额外发送 `compact`（旧 epoch、旧 `call_index`）；
- 普通 hint 不含 `context_management`，因此 `epoch` 是重写在普通通道上的唯一痕迹。

## 3. 接口变更

| 接口 | 变化 | 兼容性 |
| --- | --- | --- |
| `AgentHint.context_management` | 必填改 `ContextManagement \| None = None` | 破坏性（wire 形状变化）；构造时仍可显式传入 |
| `ContextManagement.manage_request` | `Literal["true","false"]` → `Literal[True]` | 破坏性 |
| `ContextManagement.edits` | 默认 `()` → 必须恰好 1 条 | 破坏性 |
| `ContextEdit.target` | `session\|messages\|tools` → `session` | 收窄 |
| `SessionHint.agent_type` | 必填 → 可选（管理请求无 agent 类型） | 兼容（普通路径仍填） |
| `build_agent_hint(..., rewritten=)` | 删除该参数（重写不再改变 state） | 破坏性；`state` 参数保留 |
| `build_manage_hint(...)` | 新增 | 新增 |
| `HttpManageRequestClient` / `ManageRequestError` | 新增 | 新增 |
| `AgentHintsMiddleware(manage=...)` / `send_manage()` | 新增 | 新增，默认 L1 |

版本：`0.4.0` → `0.5.0`（minor，含破坏性 wire 变化；本仓未发布过 `0.4.0` 的对外承诺，
按 minor 处理）。

## 4. 兼容性与降级

- 不配 `manage=HttpManageRequestClient` 时行为与 L1 完全一致：普通请求只带 `session` /
  `priority_control` / （合格的）`cache_control`，不发任何管理请求；
- 管理端点故障不改变推理请求本身：`resume` 失败静默降级，变更类失败则按协议隔离会话（显式报错，
  不静默）；
- `agent_status` 仍是超集字段，接收方忽略它不影响判定；
- `priority_control` / `cache_control` / 根会话 `parent_session_id: null` 等可容忍差异本轮不动。

## 5. 验证

见 [tasks/v1-ordinary-manage-split.md](../tasks/v1-ordinary-manage-split.md)：wire 形状、placeholder
body、串行/重试/隔离、中间件自动触发、双语文档一致性，以及全量 `pytest` + `ruff` + `mypy`。
