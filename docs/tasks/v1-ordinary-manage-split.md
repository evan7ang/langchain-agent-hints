# 任务：普通请求 / 管理请求分离（v1 对齐）

> 关联设计：[design/v1-ordinary-manage-split.md](../design/v1-ordinary-manage-split.md)。
> 规范原文：`agent-hint-research/agent-hint-protocol.md` v1.0（本仓外）。
> 前序任务：[tasks/four-block-field-contract.md](four-block-field-contract.md)（其中三项裁决被本任务替代）。

## 1. 目标

按协议 v1.0 解决三处硬冲突，其余可容忍差异不动：

1. 普通请求不再携带 `context_management`；管理请求 `manage_request` 为布尔 `true` 且恰好一条 edit；
2. 普通请求 `session.state` 只有 `start` / `append`；
3. 管理请求是独立 HTTP 请求，body 固定 `{"model", "messages": [], "stream": false}`。

## 2. 实现路径

| 文件 | 改动 |
| --- | --- |
| `src/langchain_agent_hints/schema.py` | `ContextManagement` 收敛为布尔 `true` + 单 edit；`ContextEdit.target` 只允许 `session`；`AgentHint.context_management` 可选；`SessionHint.agent_type` 可选；`_session_state()` 只产出 `start`/`append`；`build_agent_hint` 去掉 `rewritten` 参数、普通请求不再写块；新增 `MANAGE_EDITS` 与 `build_manage_hint()` |
| `src/langchain_agent_hints/manage.py`（新） | `HttpManageRequestClient`：placeholder body、header 载荷、per-session 串行、`resume` 有界重试、变更类失败即隔离、`ensure_ready()` 主链屏障；`ManageRequestError` |
| `src/langchain_agent_hints/middleware.py` | `manage=` 构造参数；请求边界先过 `ensure_ready`，再算 `epoch`/重写，自动发 `resume`（接手/暂停后）与 `compact`（重写后，旧 epoch）；新增 `send_manage()` 供组合发送 `pause`/`stop` |
| `src/langchain_agent_hints/__init__.py` | 导出 `ContextEdit` / `ContextManagement` / `build_manage_hint` / `HttpManageRequestClient` / `ManageRequestError`；`__version__` → `0.5.0` |
| `tests/test_schema.py` | golden wire 去 `context_management`；manage hint 形状与校验规则 |
| `tests/test_manage.py`（新） | manage wire 形状、placeholder body/bearer、`resume` 重试、变更类不重试并隔离、中间件自动 `resume`/`compact`/`pause→resume` |
| `docs/PROTOCOL.md` / `PROTOCOL.zh-CN.md` | §1 / §2 示例 / §3.1 `state` / §3.5 / §4.6（新）/ §7 修订记录 |
| `README.md` / `README.zh-CN.md` | 示例 payload、字段表、发射规则 |
| `docs/design/` / `docs/tasks/` | 新增本设计与任务；旧设计加范围替代说明 |

## 3. 验收标准

- [x] 普通请求解码后不含 `context_management`；`session.state ∈ {start, append}`。
- [x] 工具/审批等待与 `resume` 只影响 `agent_status`，不改 `session.state`。
- [x] 管理 hint 含 `session` + `context_management{manage_request: true, edits: [1 条]}`，不含 `priority_control` / `cache_control` / `agent_status`。
- [x] `manage_request` 只接受布尔 `true`；`edits` 为空或多于 1 条时构造失败。
- [x] 管理请求 body 为 `{"model", "messages": [], "stream": false}`，载荷仅在头。
- [x] `resume` 对 503 重试后成功；`pause` 不重试、隔离会话，后续 `ensure_ready` 抛 `ManageRequestError`。
- [x] 中间件：新会话不发 `resume`；接手会话发 `resume`；重写后发 `compact`（旧 epoch）且普通请求 `epoch + 1`；`pause` 后下一条请求发 `resume`。
- [x] 未配置 `manage` 时行为等同 L1（不发管理请求）。
- [x] 全量 `pytest`、`ruff check`、`ruff format --check`、`mypy` 通过。

## 4. 进展与验证记录

- 2026-11-25：按设计完成代码、测试与双语文档修改。

验证命令与结果：

```text
$ source venv-lc312/bin/activate
$ pytest -q
65 passed in 4.60s
$ ruff check src tests
All checks passed!
$ ruff format --check src tests
19 files already formatted
$ mypy
Success: no issues found in 8 source files
```

剩余限制：

- 管理客户端为 `urllib` 实现（无第三方依赖），不支持 HTTP/2 与连接池复用；
- `compact` 的 `call_index` 取"重写前那一步"，依赖中间件的高水位计数；跨进程恢复时以持久化历史播种；
- `pause` / `stop` 无自动触发，需组合层在真实生命周期事件上调用 `send_manage()`。
