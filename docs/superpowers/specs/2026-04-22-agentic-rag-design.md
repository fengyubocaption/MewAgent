# Agentic RAG 设计文档

## 背景

当前 SuperMew 的 RAG 流程是"单次调用"模式：Agent 每轮对话最多调用一次 `search_knowledge_base`，无论检索结果是否充分都不能再次检索。这由三重机制保证——代码层单次调用守卫、系统提示词禁令、RAG 图无循环的线性结构。

这种模式在简单问题上够用，但面对多维度、需要分步探索的问题时，Agent 无法自主补充检索，只能用不充分的信息回答或直接说"不知道"。

## 设计目标

让 Agent 具备多轮检索 + 自我反思能力：
- Agent 可以多次调用知识库工具，自主判断结果是否充分
- 不充分时换角度继续检索，充分时直接回答
- 保留现有 RAG 管线作为底层能力不变
- 有硬限制防止无限循环

## 设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 自主性程度 | 多轮检索 + 自我反思 | 用户选择，Agent 自主决定检索节奏 |
| RAG 管线定位 | 保留为底层能力 | Agent 只决定调不调、调几次，管线内部逻辑不变 |
| 工具参数暴露 | 只加 `top_k` | 最小化，Agent 只需控制召回数量 |
| 充分性判断 | 完全交给 Agent 推理 | 简单，依赖模型能力 |
| 迭代控制 | 固定硬上限 3 次/轮 | 简单可靠 |

## 核心改动

### 1. 工具改造：`search_knowledge_base`

**文件**：`backend/agent/tools.py`

**签名变化**：
```python
# 之前
@tool("search_knowledge_base")
def search_knowledge_base(query: str) -> str

# 之后
@tool("search_knowledge_base")
def search_knowledge_base(query: str, top_k: int = 5) -> str
```

**行为变化**：

1. **移除单次调用守卫**，改为请求级调用计数器（见第 3 节），上限 3 次/轮
2. **去重累积**：每次调用后，新文档与已检索文档按 `(filename, page_number, text)` 去重，合并到累积结果集
3. **返回格式增加引导语**：
   ```
   [检索结果 - 第2次/最多3次] 已累积 8 篇文档，本次新增 3 篇：

   [1] xxx.pdf (Page 3): ...
   [2] xxx.pdf (Page 5): ...

   ---本次检索摘要---
   查询: "A产品安全架构"
   策略: step_back → 重检索
   rerank 最高分: 0.87

   如需补充检索其他方面，可再次调用；如信息已充分，请直接回答。
   ```
4. **达到上限时**返回累积结果 + "已达检索上限，请基于以上信息回答"
5. **rag_trace 改为请求级存储**，累积所有调用的 trace 信息

### 2. 系统提示词改造

**文件**：`backend/agent/agent.py`

移除"最多一次知识库调用"和"不允许再次调用工具"的禁令，替换为：

```
你是一个知识库助手。回答问题时遵循以下检索策略：

1. 分析用户问题，判断是否需要检索知识库
2. 调用 search_knowledge_base 进行检索，可通过 top_k 控制召回数量
3. 阅读检索结果，判断信息是否充分：
   - 如果充分：直接回答
   - 如果不充分但有其他可探索的角度：再次调用，换一个查询角度
   - 如果完全不相关或已到上限：诚实说明无法回答
4. 每轮对话最多 3 次知识库调用，请合理使用，避免重复查询相同内容
5. 多次检索的结果会自动去重累积，无需担心重复
6. 禁止凭空编造知识库中不存在的信息
```

### 3. 请求级状态隔离（并发安全）

**文件**：`backend/agent/tools.py` + `backend/agent/agent.py`

**当前问题**：三个全局可变变量在并发请求下不安全：
- `_KNOWLEDGE_TOOL_CALLS_THIS_TURN`：调用计数
- `_LAST_RAG_CONTEXT`：rag_trace 存储
- `_RAG_STEP_QUEUE`：SSE 步骤推送

**改为**：用 Python `contextvars` 实现请求级隔离。

```python
@dataclass
class RetrievalState:
    call_count: int = 0
    max_calls: int = 3
    accumulated_docs: list          # 累积文档
    rag_traces: list                # 累积 trace
    seen_keys: set                  # (filename, page, text) 去重键
    step_queue: Optional[asyncio.Queue] = None

_retrieval_state: ContextVar[RetrievalState] = ContextVar('retrieval_state')
```

**生命周期**：
1. `chat_with_agent_stream` 开始时创建并设置 `RetrievalState`
2. `search_knowledge_base` 每次调用从 context var 读取状态，更新后写回
3. `emit_rag_step` 从 context var 获取 queue
4. 请求结束后 context var 自动清理

**删除**：
- `_KNOWLEDGE_TOOL_CALLS_THIS_TURN` 全局变量
- `_LAST_RAG_CONTEXT` 全局变量
- `_RAG_STEP_QUEUE` 全局变量
- `set_rag_step_queue()` 全局函数
- `reset_tool_call_guards()` 全局函数

全部由 `RetrievalState` + `contextvars` 替代。

### 4. Agent 递归限制与流式适配

**文件**：`backend/agent/agent.py`

**递归限制**：`recursion_limit` 从 8 提升到 12。
- 3 次检索 = 6 步（每次调用 + 观察）
- 初始推理 1-2 步
- 中间反思 1-2 步
- 最终生成 1-2 步
- 留有余量但不会失控

**流式适配**：
1. 多次检索的 `rag_step` 事件持续推送，前端无需改动
2. `trace` 事件在 Agent 完成后统一发送，包含所有检索调用的累积 trace
3. `search_knowledge_base` 保持同步 `@tool`，LangGraph 的 `astream` 内部处理线程调度

## 不动的部分

| 文件 | 理由 |
|------|------|
| `rag_pipeline.py` | RAG 管线内部逻辑不变，仍是检索→评分→改写→重检索 |
| `rag_utils.py` | 检索/重排/合并工具函数不变 |
| `routes/api.py` | 路由层不需要改动 |
| 前端 SSE 逻辑 | 现有状态机已支持持续的 rag_step 事件 |

## 变更量估算

| 文件 | 变更量 | 说明 |
|------|--------|------|
| `backend/agent/tools.py` | 中 | 工具改造 + contextvars 替代全局变量 |
| `backend/agent/agent.py` | 小 | 系统提示词 + recursion_limit + RetrievalState 初始化 |
| `backend/rag/rag_pipeline.py` | 无 | 不动 |
| `backend/rag/rag_utils.py` | 无 | 不动 |
| `backend/routes/api.py` | 无 | 不动 |
| 前端 | 无 | 不动 |
