# 任务：四块必需字段契约对齐（LangGraph）

> 关联设计：[design/four-block-field-contract.md](../design/four-block-field-contract.md)。
> 规范原文：`agent-hint-research/AgentHint-下发字段规范-v1.md`（本仓外）。

## 1. 目标

按跨 harness 字段规范 v1，在 `X-Agent-Hint` 中补齐下游要求的四块：

1. `session.state`（六值，`stop` 本期不实现）+ 根会话 `parent_session_id: null`；
2. `context_management`（恒存在，edits 由 state 展开）；
3. `priority_control`（`priority` 的规范位置，同值）；
4. `cache_control` 由元组改单对象并补 `msg_offset`，默认 `pin` 改 `ephemeral`。

`agent_status` 四值折叠、二值 `priority`、状态事件通道、无 `thread_id` 降级规则均不变。

## 2. 实现路径

| 文件 | 改动 |
| --- | --- |
| `src/langchain_agent_hints/schema.py` | 新增 `SessionState` / `PriorityControl` / `ContextEdit` / `ContextManagement`；`SessionHint.state`；`AgentHint.context_management` + `priority_control`；`CacheControlHint` 收敛 `ephemeral` + `msg_offset`；`build_agent_hint(rewritten=...)`；`hint_to_dict` 补根会话 `parent_session_id: null` |
| `src/langchain_agent_hints/middleware.py` | 把 `_take_epoch_slot` 的 `rewritten` 传入 `build_agent_hint` |
| `tests/test_schema.py` | wire 形状、默认值、None 丢弃断言 |
| `tests/test_field_sending.py` / `test_middleware.py` / `test_wire.py` | 真实链路与抓包断言 |
| `docs/PROTOCOL.md` / `PROTOCOL.zh-CN.md` | §3.1 / §3.4 / §3.5 / §3.6 / §4 与示例 |
| `README.md` / `README.zh-CN.md` | 示例 payload 与字段表 |

## 3. 验收标准

- [x] `session.state` 五值可推导（`pause > compact > resume > start > append`），根会话首请求为 `start`，子会话首请求为 `append`。
- [x] `context_management.edits` 随 state 展开，普通请求为 `[]`。
- [x] `priority_control.priority` 与顶层 `priority` 同值；`priority=None` 时两块同时省略。
- [x] `cache_control` 为单对象，`msg_offset == range.end`；抑制条件不变（抑制时整块省略）。
- [x] 根会话 `session.parent_session_id is None`（键存在）。
- [x] 全量基线 `pytest && ruff check . && ruff format --check . && mypy` 通过。

## 4. 进展与验证记录

- 2026-11-25：按设计完成代码、测试与双语文档修改。

验证命令与结果：

```text
$ source venv-lc312/bin/activate
$ pytest -q
57 passed
$ ruff check .
All checks passed!
$ ruff format --check .
26 files already formatted
$ mypy
Success: no issues found in 7 source files
```

## 5. 剩余限制

- `stop` 不下发：跑完一轮没有收尾请求，留给带外事件通道（与规范一致）。
- `epoch` 仍是进程内前缀指纹启发式，重启清零（既有已文档化限制，本次不涉及）。
- `cache_control` 形状变更对消费方是破坏性变化；本仓未升版本号，由维护者发版时决定。
