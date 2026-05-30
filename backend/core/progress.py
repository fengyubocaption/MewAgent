"""请求级检索进度回调 — 跨模块共享，打破 agent/ ↔ rag/ 循环依赖。"""

from typing import Optional
from dataclasses import dataclass, field
import asyncio
from contextvars import ContextVar


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
