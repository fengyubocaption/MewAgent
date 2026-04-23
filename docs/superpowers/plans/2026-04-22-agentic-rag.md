# Agentic RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable the Agent to perform multi-turn retrieval with self-reflection — calling `search_knowledge_base` up to 3 times per conversation turn, with automatic document deduplication and accumulation.

**Architecture:** Replace global mutable state with request-scoped `contextvars`. Remove the single-call guard in `search_knowledge_base`, add `top_k` parameter, implement dedup+accumulation with guidance text. Update the system prompt to encourage multi-step retrieval. Raise `recursion_limit` from 8 to 12.

**Tech Stack:** Python `contextvars`, LangChain `@tool`, LangGraph `astream`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `backend/agent/tools.py` | Modify | `RetrievalState` dataclass, `contextvars`, rewritten `search_knowledge_base`, rewritten `emit_rag_step` |
| `backend/agent/agent.py` | Modify | System prompt, `chat_with_agent_stream` initialization, trace accumulation, `recursion_limit` |
| `backend/rag/rag_pipeline.py` | No change | RAG pipeline internals stay the same |

---

### Task 1: Introduce `RetrievalState` and `contextvars` in `tools.py`

**Files:**
- Modify: `backend/agent/tools.py`

This task replaces all global mutable state with a request-scoped dataclass backed by `contextvars`.

- [ ] **Step 1: Add `RetrievalState` dataclass and `ContextVar`**

Replace lines 1-18 of `backend/agent/tools.py` with:

```python
from typing import Optional
from dataclasses import dataclass, field
import os
import asyncio
import requests
from contextvars import ContextVar
from dotenv import load_dotenv
try:
    from langchain_core.tools import tool
except ImportError:
    from langchain_core.tools import tool

load_dotenv()

AMAP_WEATHER_API = os.getenv("AMAP_WEATHER_API")
AMAP_API_KEY = os.getenv("AMAP_API_KEY")


@dataclass
class RetrievalState:
    """请求级检索状态，替代全局变量，每个请求独立隔离。"""
    call_count: int = 0
    max_calls: int = 3
    accumulated_docs: list = field(default_factory=list)
    rag_traces: list = field(default_factory=list)
    seen_keys: set = field(default_factory=set)
    step_queue: Optional[object] = None  # asyncio.Queue or _RagStepProxy
    step_loop: Optional[asyncio.AbstractEventLoop] = None


_retrieval_state: ContextVar[RetrievalState] = ContextVar('retrieval_state')
```

- [ ] **Step 2: Replace `emit_rag_step` and remove old global functions**

Replace lines 20-64 (the old `_set_last_rag_context`, `get_last_rag_context`, `reset_tool_call_guards`, `set_rag_step_queue`, `emit_rag_step`) with:

```python
def init_retrieval_state() -> RetrievalState:
    """创建并设置请求级检索状态。在 chat_with_agent_stream 开头调用。"""
    state = RetrievalState()
    _retrieval_state.set(state)
    return state


def set_rag_step_queue(queue):
    """设置 RAG 步骤队列到当前请求的检索状态中。"""
    try:
        state = _retrieval_state.get()
    except LookupError:
        return
    state.step_queue = queue
    if queue:
        try:
            state.step_loop = asyncio.get_running_loop()
        except RuntimeError:
            state.step_loop = asyncio.get_event_loop()
    else:
        state.step_loop = None


def emit_rag_step(icon: str, label: str, detail: str = ""):
    """向队列发送一个 RAG 检索步骤。支持跨线程安全调用。"""
    try:
        state = _retrieval_state.get()
    except LookupError:
        return
    if state.step_queue is not None and state.step_loop is not None:
        step = {"icon": icon, "label": label, "detail": detail}
        try:
            if not state.step_loop.is_closed():
                state.step_loop.call_soon_threadsafe(state.step_queue.put_nowait, step)
        except Exception:
            pass
```

Key changes:
- No more global variables (`_LAST_RAG_CONTEXT`, `_KNOWLEDGE_TOOL_CALLS_THIS_TURN`, `_RAG_STEP_QUEUE`, `_RAG_STEP_LOOP`)
- `init_retrieval_state()` creates and sets the ContextVar for each request
- `emit_rag_step` reads from ContextVar instead of globals
- `get_last_rag_context`, `reset_tool_call_guards`, `_set_last_rag_context` are removed entirely

- [ ] **Step 3: Commit**

```bash
git add backend/agent/tools.py
git commit -m "refactor(tools): 引入 RetrievalState + contextvars 替代全局变量"
```

---

### Task 2: Rewrite `search_knowledge_base` tool

**Files:**
- Modify: `backend/agent/tools.py`

Replace the existing `search_knowledge_base` function (lines 129-168) with the new multi-call version.

- [ ] **Step 1: Write the new `search_knowledge_base` function**

Replace the entire `search_knowledge_base` function with:

```python
@tool("search_knowledge_base")
def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """Search for information in the knowledge base using hybrid retrieval (dense + sparse vectors).
    Can be called multiple times per turn (max 3) to iteratively refine retrieval."""
    try:
        state = _retrieval_state.get()
    except LookupError:
        state = init_retrieval_state()

    # 检查调用次数上限
    if state.call_count >= state.max_calls:
        # 已达上限，返回累积结果
        if state.accumulated_docs:
            formatted = _format_accumulated_docs(state)
            return f"{formatted}\n\n⚠️ 已达检索上限（{state.max_calls}次），请基于以上信息回答。"
        return f"已达检索上限（{state.max_calls}次），且未找到任何相关文档。请诚实说明无法回答。"

    state.call_count += 1
    current_call = state.call_count

    from backend.rag.rag_pipeline import run_rag_graph

    rag_result = run_rag_graph(query, top_k=top_k)

    new_docs = rag_result.get("docs", []) if isinstance(rag_result, dict) else []
    rag_trace = rag_result.get("rag_trace", {}) if isinstance(rag_result, dict) else {}
    if rag_trace:
        state.rag_traces.append(rag_trace)

    # 去重：与已累积文档比对
    added_count = 0
    for doc in new_docs:
        key = (doc.get("filename"), doc.get("page_number"), doc.get("text"))
        if key not in state.seen_keys:
            state.seen_keys.add(key)
            state.accumulated_docs.append(doc)
            added_count += 1

    # 构建返回内容
    if not state.accumulated_docs:
        if current_call < state.max_calls:
            return (
                f"未找到相关文档（第{current_call}次/最多{state.max_calls}次）。\n"
                "建议换一个查询角度再次检索，或诚实说明无法回答。"
            )
        return "No relevant documents found in the knowledge base."

    formatted = _format_accumulated_docs(state)

    # 本次检索摘要
    strategy = rag_trace.get("rewrite_strategy", "direct") if rag_trace else "direct"
    rerank_score = None
    if state.accumulated_docs:
        for doc in reversed(state.accumulated_docs):
            if doc.get("rerank_score") is not None:
                rerank_score = doc.get("rerank_score")
                break

    summary_lines = [
        f"查询: {query}",
        f"策略: {strategy}",
    ]
    if rerank_score is not None:
        summary_lines.append(f"rerank 最高分: {rerank_score:.2f}")

    # 引导语
    if current_call >= state.max_calls:
        guidance = f"⚠️ 已达检索上限（{state.max_calls}次），请基于以上信息回答。"
    else:
        remaining = state.max_calls - current_call
        guidance = f"如需补充检索其他方面，可再次调用（剩余{remaining}次）；如信息已充分，请直接回答。"

    return (
        f"[检索结果 - 第{current_call}次/最多{state.max_calls}次] "
        f"已累积 {len(state.accumulated_docs)} 篇文档，本次新增 {added_count} 篇：\n\n"
        f"{formatted}\n\n"
        f"---本次检索摘要---\n"
        + "\n".join(summary_lines)
        + f"\n\n{guidance}"
    )


def _format_accumulated_docs(state: RetrievalState) -> str:
    """格式化累积文档列表。"""
    formatted = []
    for i, result in enumerate(state.accumulated_docs, 1):
        source = result.get("filename", "Unknown")
        page = result.get("page_number", "N/A")
        text = result.get("text", "")
        formatted.append(f"[{i}] {source} (Page {page}):\n{text}")
    return "\n\n---\n\n".join(formatted)
```

Key changes:
- New `top_k` parameter (default 5)
- Request-scoped call counting via `_retrieval_state`
- Document deduplication via `seen_keys` set
- Accumulated results across calls
- Guidance text at the end of each return
- `_format_accumulated_docs` helper extracted for reuse

- [ ] **Step 2: Update `run_rag_graph` call to pass `top_k`**

The current `rag_pipeline.py` `run_rag_graph` function signature is `run_rag_graph(question: str)` — it needs to accept and forward `top_k`. This is a minimal change in `rag_pipeline.py`.

In `backend/rag/rag_pipeline.py`, change `run_rag_graph` (lines 389-402):

```python
def run_rag_graph(question: str, top_k: int = 5) -> dict:
    return rag_graph.invoke({
        "question": question,
        "query": question,
        "top_k": top_k,
        "context": "",
        "docs": [],
        "route": None,
        "expansion_type": None,
        "expanded_query": None,
        "step_back_question": None,
        "step_back_answer": None,
        "hypothetical_doc": None,
        "rag_trace": None,
    })
```

And add `top_k` to `RAGState` (line 77-88):

```python
class RAGState(TypedDict):
    question: str
    query: str
    top_k: int
    context: str
    docs: List[dict]
    route: Optional[str]
    expansion_type: Optional[str]
    expanded_query: Optional[str]
    step_back_question: Optional[str]
    step_back_answer: Optional[str]
    hypothetical_doc: Optional[str]
    rag_trace: Optional[dict]
```

And update `retrieve_initial` (line 106) and `retrieve_expanded` (lines 265, 294) to use `state["top_k"]` instead of hardcoded `5`:

In `retrieve_initial`, line 106:
```python
    top_k = state.get("top_k", 5)
    retrieved = retrieve_documents(query, top_k=top_k)
```

In `retrieve_expanded`, line 265:
```python
        top_k = state.get("top_k", 5)
        retrieved_hyde = retrieve_documents(hypothetical_doc, top_k=top_k)
```

In `retrieve_expanded`, line 294:
```python
        top_k = state.get("top_k", 5)
        retrieved_stepback = retrieve_documents(expanded_query, top_k=top_k)
```

- [ ] **Step 3: Commit**

```bash
git add backend/agent/tools.py backend/rag/rag_pipeline.py
git commit -m "feat(tools): search_knowledge_base 支持多轮检索 + top_k 参数"
```

---

### Task 3: Update `chat_with_agent_stream` in `agent.py`

**Files:**
- Modify: `backend/agent/agent.py`

- [ ] **Step 1: Update imports**

Replace line 9:
```python
from backend.agent.tools import get_current_weather, search_knowledge_base, get_last_rag_context, reset_tool_call_guards, set_rag_step_queue
```

With:
```python
from backend.agent.tools import get_current_weather, search_knowledge_base, init_retrieval_state, set_rag_step_queue, _retrieval_state
```

- [ ] **Step 2: Update `chat_with_agent_stream` initialization**

Replace lines 435-449 (the initialization block that calls `get_last_rag_context(clear=True)`, `reset_tool_call_guards()`, creates `output_queue`, creates `_RagStepProxy`, calls `set_rag_step_queue`):

```python
    messages = storage.load(user_id, session_id)

    # 初始化请求级检索状态（替代全局变量重置）
    retrieval_state = init_retrieval_state()

    # 统一输出队列：所有事件（content / rag_step）都汇入这里
    output_queue = asyncio.Queue()

    class _RagStepProxy:
        """代理对象：将 emit_rag_step 的原始 step dict 包装后放入统一输出队列。"""
        def put_nowait(self, step):
            output_queue.put_nowait({"type": "rag_step", "step": step})

    set_rag_step_queue(_RagStepProxy())
```

- [ ] **Step 3: Update `recursion_limit` from 8 to 12**

In `_agent_worker` (line 464), change:
```python
                config={"recursion_limit": 12},
```

- [ ] **Step 4: Update RAG trace extraction after agent completes**

Replace lines 516-522 (the trace extraction block):

```python
    # 获取累积的 RAG traces
    rag_trace = None
    try:
        final_state = _retrieval_state.get()
        if final_state.rag_traces:
            # 多次检索时合并 trace 信息
            if len(final_state.rag_traces) == 1:
                rag_trace = final_state.rag_traces[0]
            else:
                rag_trace = {
                    "multi_retrieval": True,
                    "call_count": final_state.call_count,
                    "traces": final_state.rag_traces,
                }
    except LookupError:
        pass

    # 发送 trace 信息
    if rag_trace:
        yield f"data: {json.dumps({'type': 'trace', 'rag_trace': rag_trace})}\n\n"
```

- [ ] **Step 5: Commit**

```bash
git add backend/agent/agent.py
git commit -m "feat(agent): 请求级检索状态初始化 + recursion_limit 提升 + trace 累积"
```

---

### Task 4: Update system prompt

**Files:**
- Modify: `backend/agent/agent.py`

- [ ] **Step 1: Replace the system prompt in `_build_system_prompt`**

Replace the `base_prompt` string in `_build_system_prompt` (lines 354-392) with:

```python
    base_prompt = (
        "You are a cute cat bot that loves to help users. "
        "When responding, you may use tools to assist.\n"
        "\n"
        "## 检索策略\n"
        "1. 分析用户问题，判断是否需要检索知识库\n"
        "2. 调用 search_knowledge_base 进行检索，可通过 top_k 控制召回数量\n"
        "3. 阅读检索结果，判断信息是否充分：\n"
        "   - 如果充分：直接回答\n"
        "   - 如果不充分但有其他可探索的角度：再次调用，换一个查询角度\n"
        "   - 如果完全不相关或已到上限：诚实说明无法回答\n"
        "4. 每轮对话最多 3 次知识库调用，请合理使用，避免重复查询相同内容\n"
        "5. 多次检索的结果会自动去重累积，无需担心重复\n"
        "6. 禁止凭空编造知识库中不存在的信息\n"
        "\n"
        "## Memory Tools - CRITICAL\n"
        "WHEN TO CALL save_user_memory TOOL:\n"
        "- User explicitly says 'remember' / '请记住' / '记住'\n"
        "- User states a preference (e.g., 'I prefer Chinese answers')\n"
        "- User tells you something important about themselves\n"
        "- ALWAYS call save_user_memory BEFORE responding to the user\n"
        "\n"
        "WHEN TO CALL search_memories TOOL:\n"
        "- User asks about previous conversation\n"
        "- User asks 'do you remember...' / '你还记得...吗'\n"
        "- At the start of conversation to recall user preferences\n"
        "\n"
        "Use memory tools to store information that is valuable across sessions:\n"
        "- save_user_memory: User preferences (coding style, language preference)\n"
        "- save_feedback_memory: User corrections or validated approaches\n"
        "- save_project_memory: Project decisions and their reasons\n"
        "- save_reference_memory: External resource pointers (dashboards, docs)\n"
        "\n"
        "## Memory Boundary\n"
        "DO NOT store in memory:\n"
        "- File paths, function names, code structure (can be re-read)\n"
        "- Current task progress (belongs to conversation context)\n"
        "- Temporary state, branch names, PR numbers (quickly outdated)\n"
        "- Specific code fixes (code is the source of truth)\n"
        "\n"
        "Only store information that remains valuable across sessions and cannot be derived from code."
    )
```

Key changes:
- Removed "At most one knowledge tool call per turn"
- Removed "MUST NOT call any tool again"
- Removed "Once you call search_knowledge_base and receive its result, you MUST immediately produce the Final Answer"
- Added explicit multi-step retrieval guidance with clear decision criteria
- Memory tools section preserved as-is

- [ ] **Step 2: Commit**

```bash
git add backend/agent/agent.py
git commit -m "feat(agent): 系统提示词支持多轮检索策略"
```

---

### Task 5: Manual verification

**Files:**
- No code changes — manual testing only

- [ ] **Step 1: Start the server and verify no import errors**

Run: `uv run python -c "from backend.agent.tools import search_knowledge_base, init_retrieval_state, set_rag_step_queue, _retrieval_state; print('imports OK')"`

Expected: `imports OK`

- [ ] **Step 2: Start the server and verify no startup errors**

Run: `uv run python -c "from backend.agent.agent import chat_with_agent_stream; print('agent OK')"`

Expected: `agent OK`

- [ ] **Step 3: Start the full server and test via browser**

Run: `uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload`

Test scenarios in browser at `http://127.0.0.1:8000/`:

1. **Simple question** (1 retrieval should suffice): Ask a factual question from the knowledge base. Verify the Agent calls `search_knowledge_base` once and answers.

2. **Complex question** (should trigger multiple retrievals): Ask a question that spans multiple topics, e.g. "对比A产品和B产品的技术架构差异". Verify the Agent makes 2-3 calls with different queries.

3. **Insufficient results**: Ask a question not in the knowledge base. Verify the Agent either stops after 1 call (if clearly irrelevant) or tries once more with a different angle, then admits it doesn't know.

4. **Aborting mid-stream**: Start a question, then click abort. Verify no errors in server logs and the asyncio task is properly cancelled.

- [ ] **Step 4: Verify SSE event format unchanged**

Check that:
- `rag_step` events still arrive during retrieval
- `content` events still stream token-by-token
- `trace` event arrives at the end (now with `multi_retrieval: true` and `traces: [...]` if multiple calls were made)
- `[DONE]` event still marks stream end

- [ ] **Step 5: Final commit if any fixes were needed**

```bash
git add -A
git commit -m "fix: Agentic RAG 集成修复"
```
