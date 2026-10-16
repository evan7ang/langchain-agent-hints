# 仓库开发规范

本文件适用于整个仓库。文档的目录分工和维护规则见 [docs/AGENTS.md](docs/AGENTS.md)。

## 仓库目录结构

```text
src/langchain_agent_hints/  # 包源码（导入名 langchain_agent_hints）
├── __init__.py             # 对外导出面；__version__ 是版本单一来源（hatch 动态读取）
├── schema.py               # wire 形状与编码/解码（纯 stdlib，不依赖 harness 即可单测）
├── middleware.py           # AgentHintsMiddleware：会话身份、call_index/epoch 派生、hint 发射
├── status.py               # AgentStatusEventsMiddleware：状态转换事件、sink 协议与 JSONL sink
└── py.typed                # PEP 561 类型标记
tests/                      # keyless pytest（无需任何 API key 即可全量运行）
├── conftest.py             # 进程内捕获 stub 与公共夹具
├── test_schema.py          # 协议层：形状、编码、v1 永不发射边界
├── test_middleware.py      # 发射规则、call_index/epoch 派生、降级
├── test_status.py          # 状态机、sink 契约、HITL pause→resume 全生命周期（同步+异步）
└── test_wire.py            # wire 抓包：真实适配器经进程内捕获，头形状独立解码、header-only 断言
docs/                       # wire 规格与实测兼容性记录，分工见 docs/AGENTS.md
examples/
└── deepagents_quickstart.py  # deepagents 组装示例与组合层 env 开关示范
.github/workflows/
├── ci.yml                  # ruff + mypy strict + pytest（Python 3.11–3.13 matrix）
└── release.yml             # tag v* 触发：测试 → build → twine check → OIDC Trusted Publishing → GitHub Release
```

根目录文件：

| 路径 | 用途 |
| --- | --- |
| `README.md` / `README.zh-CN.md` | 英文/中文双语文档，顶部语言切换行，内容两两对应 |
| `pyproject.toml` | hatch 构建、依赖声明与 ruff / mypy / pytest 配置 |
| `LICENSE` | 木兰宽松许可证，第2版（Mulan PSL v2） |
| `.gitignore` | 排除缓存、venv、构建产物与本地运行产物（`status-events.jsonl`、`trace*.jsonl`） |

`.venv/`、`dist/`、`build/`、各类工具缓存目录与示例运行产物（`examples/status-events.jsonl`）为本地生成物，不直接修改或提交。

## 设计约束（硬性，演进不得无声破坏）

1. **分层不可倒置**：`schema.py` 只用 stdlib（wire 形状不依赖 harness 即可单测）；`middleware.py` / `status.py` 只用 langchain 公开 API（如 `AgentMiddleware`），零私有 import——这是"可发布中间件"的资格线，新增依赖前先确认不破坏该约束。
2. **header-only 通道**：hint 只经 `X-Agent-Hint` 头发射（紧凑 UTF-8 JSON → 无 padding base64url → 头）；请求体不得出现任何 hint 字段。`extra_headers` 合并为增量语义。
3. **v1 永不发射**：`prediction`、`strict_priority`、`derivation.fork_point` 只在 schema 中定义、构造器不填，测试锁死该边界。
4. **派生规则**：`call_index` 由消息历史派生（`1 + count(AIMessage)`）；`epoch` 是进程内前缀指纹启发式，重启清零是文档化限制。若用事件源替代启发式，属接口演进，先升 `docs/PROTOCOL.md` 版本节再动代码。
5. **身份与配置**：`agent_type` 构造必传（身份是组合特有事实）；库本身不读环境变量，env 开关是 README 记录的组合层模式。
6. **sink 契约**：`StatusEventSink` 是同步单方法协议，best-effort 投递（重试后丢弃并告警，不拖垮 agent run）；会话内 `seq` 有序由 sink 保证。
7. **时序事实**：钩子触发时序以 `docs/COMPATIBILITY.md` 的实测记录为基线；升级 langchain / deepagents 后必须重验并更新该记录。
8. **协议语义引用**一律指向本仓 `docs/PROTOCOL.md` 的 `§N` 锚点，不在代码注释中复述 wire 规格。

## 代码开发规范

### 开发流程

1. **开发需求附带设计文档与任务验证文档。** 新功能、协议演进（动 schema / wire 形状 / 发射规则）、跨模块重构，先在 `docs/design/` 落设计文档（范围、接口、兼容性影响），再在 `docs/tasks/` 建任务文档（目标、实现路径、验收标准）；实施过程中在同一任务文档内记录进展、调整与验证结果（验证命令与结果摘要、剩余限制）——任务文档即该需求的验证记录。已有设计覆盖时直接引用，不重复建档。
2. **wire / 协议语义变化先升规格。** 先改 `docs/PROTOCOL.md`（新增版本节，不改写旧版语义），同步 `tests/test_schema.py` 等边界测试，再动实现。
3. **小 bug 和局部改动可以 ad hoc。** 范围清楚、影响有限的修复、文字调整和小重构可直接修改并验证，不强制建档；涉及 wire 形状、协议语义或时序事实时升级为第 1 条流程。
4. **修改前确认当前状态。** 阅读相关实现、PROTOCOL/COMPATIBILITY 对应章节和已有测试；不顺带整理无关文件，不扩大改动范围。
5. **按影响范围验证。** 协议、发射规则、状态机改动跑全量基线并补回归测试；纯文字改动检查内容、链接与示例即可。不仅为复述实现而增加测试。
6. **交付说明实际结果。** 说明改动、验证命令与结果、剩余限制；未执行的验证明确标注待验收，不记为通过。影响使用者操作的变化同步更新 README 双语版与 `examples/`。

### 验证命令

```bash
pytest && ruff check . && ruff format --check . && mypy   # 交付前全绿基线
```

临时验证脚本、抓包输出与试运行日志放仓库外的临时目录，不提交；需要长期化的验证证据脱敏后进 `tests/`。

### Python 代码风格

- 遵循 ruff 配置（line-length 100、target py311）与现有分层；mypy strict 全通过，不通过关闭规则规避类型问题。
- 公开 API 带完整类型注解；内部逻辑优先类型推断，外部输入用窄化校验，不使用无依据的 `Any` 与非空断言。
- 文件名用 `snake_case`，类用 `PascalCase`，函数与变量用 `snake_case`，模块级常量可用 `UPPER_SNAKE_CASE`；名称说明职责。
- 注释解释约束、原因和非显然边界，不复述代码字面含义；docstring 描述公开契约，语义细节引用 `docs/PROTOCOL.md`。
- 文件为 UTF-8、LF，并以换行结束；遵循邻近代码风格，不把无关格式化混入功能提交。

## 发布流程

- 版本单一来源：`src/langchain_agent_hints/__init__.py` 的 `__version__`（hatch 动态读取）。
- 发版 = 改该字段 → commit → `git tag vX.Y.Z` → push tag；`release.yml` 自动执行测试、build、twine check、OIDC Trusted Publishing 并创建 GitHub Release。tag 与字段不一致会导致发布失败。
- 元数据变化（如 license、依赖、Python 版本）随下一次版本发布生效；发布后核对 PyPI 页面元数据。

## Git Commit 规范

- 提交信息使用 conventional commits，与现有历史一致：`feat:` / `fix:` / `docs:` / `ci:` / `test:` / `chore:` 前缀 + 一行简短英文说明；一项改动一个提交，不展开实施过程或验证细节。
- 不添加 `Author`、`Co-authored-by` 等署名信息。
- 提交前检查 diff 与暂存区，只提交本次任务相关改动，不夹带无关文件。
- **严禁提交**：任何 key / token / 真实网关地址、抓包原始日志、本地运行产物、临时验证脚本；依赖变更同步更新 `pyproject.toml` 并确认 CI 矩阵（3.11–3.13）可通过。

示例：

```text
feat: buffered status sink wrapper
fix: keep call_index stable across resume
docs: extend compatibility notes for deepagents upgrade
```
