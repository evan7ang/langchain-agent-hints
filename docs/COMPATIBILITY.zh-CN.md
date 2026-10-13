# 兼容性说明

[English](COMPATIBILITY.md) | **简体中文**

实现所依赖的实测运行时事实，以及已知限制。验证于 `langchain 1.4.1`、`langchain-core 1.6.4`、`deepagents 0.7.15`（Python）、LangGraph 内存检查点（`MemorySaver`）、Python 3.11–3.12；真模型冒烟另经 `langchain 1.4.5` / `deepagents 0.7.24` / `langchain-openai 1.7.1`（OpenAI 兼容网关后的 GLM 推理模型）验证。

## 会话身份来源

- `configurable.thread_id`——会话身份（钩子内经 `ensure_config()` 读取）。
- `metadata.lc_agent_name`——Agent 实例名。
- `configurable["ls_agent_type"] == "subagent"`（同时检查 `metadata`）——deepagents `subagents.py` 对子运行的标记；子会话据此自报为 `{thread_id}:{agent_name}`。

## 头透传链（实测）

`wrap_model_call` → `ModelRequest.model_settings` → `model.bind(**model_settings)` / `bind_tools(..., **model_settings)`（`langchain.agents` 工厂）→ ChatOpenAI 请求 → OpenAI SDK `create(extra_headers=…)`。因此 `X-Agent-Hint` 头在 OpenAI 兼容适配器上到达 wire；fake 模型只是忽略多余的 bind kwargs（keyless 路径不受影响）。

该通道是 SDK 级的，与 API 形态无关：langchain-openai 1.7.x 对部分模型会改走 OpenAI **Responses API**（`POST /v1/responses`）而非 chat completions——已对真实 OpenAI 兼容网关端到端验证：hint 头在两条路径上均到达并被上游接受（HTTP 200）。

## 钩子时序（实测）

- `after_model` 钩子是独立图节点（`graph.add_edge("model", "<mw>.after_model")`）。`middleware=` 序列中靠后的中间件处于内侧，其 `after_model` 节点先触发。deepagents 把 `HumanInTheLoopMiddleware` 放在栈尾，因此 HITL 的节点在用户中间件 `after_model` 钩子**之前**中断。
- `wrap_model_call` 紧随上一个工具返回执行——钩子内的墙钟是 `agent_status.state_since_ms` 可用的最近"running 起点"。
- `after_agent` 只在图运行真正完成时触发；审批挂起（`interrupt()`）中途暂停不会到达它，恢复后的运行恰好 finalize 一次。
- 被门控工具在挂起期间不会到达 `wrap_tool_call`（无虚假 `tool` 事件），批准后才到达——即自然的恢复信号。

## 已知限制（v1）

- **同名并发子代理**在会话 id（`{thread_id}:{agent_name}`）上冲突：同名子代理两个并发运行的事件与 epoch 会交错。
- **`epoch` 是启发式**：通过消息前缀指纹（条数 + 首消息哈希）检测上下文重写。同时保留两者的重写检测不到。
- **`call_index` 可能回落**：summarization 重写会把更早的 AIMessage 压缩掉；并发的 `epoch` 递增会标记这次重写。
- **`pause` 可能误报**：当中断配置使用 `when` 谓词、且在响应已通过静态门控名检查后否决 interrupt 时（库只检查工具名）。
- **sink 是同步的**：异步钩子内联调用 sink。慢传输请使用带缓冲的 sink。
- **非 OpenAI 兼容模型**：hint 经 bind kwargs（`extra_headers`）传递；不支持按请求头的模型/SDK 会直接忽略——Agent 运行永不受影响。
