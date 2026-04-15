"""LangMem 记忆管理模块 — 基于 PostgreSQL + 内存 fallback"""
import json
import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langmem import create_manage_memory_tool, create_search_memory_tool, create_memory_manager

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM 模型配置（用于记忆提取）
# ---------------------------------------------------------------------------
_API_KEY = os.getenv("ARK_API_KEY")
_MODEL = os.getenv("MODEL")
_BASE_URL = os.getenv("BASE_URL")


def _llm_spec() -> str:
    """返回用于记忆提取的 LLM 标识。"""
    return f"{_MODEL or 'claude-sonnet-4-6'}"


def _create_chat_model():
    """创建用于记忆提取的聊天模型。"""
    from langchain.chat_models import init_chat_model

    return init_chat_model(
        model=_MODEL or "claude-sonnet-4-6",
        model_provider="openai",
        api_key=_API_KEY,
        base_url=_BASE_URL,
        temperature=0,
    )


# ---------------------------------------------------------------------------
# PostgreSQL Store 实现
# ---------------------------------------------------------------------------
# LangGraph 没有内置 PostgreSQL store，这里用 SQLAlchemy 实现一个轻量版本。
# 记忆以 JSON 形式存储在 PostgreSQL 的 langmem_memories 表中。

from backend.db.models import _LangMemMemory
from sqlalchemy import text


def _init_memory_table(engine):
    """在数据库中创建 langmem_memories 表。"""
    try:
        _LangMemMemory.__table__.create(bind=engine, checkfirst=True)
    except Exception as e:
        logger.warning("langmem_memories 表创建失败（可能已存在）: %s", e)


class _SQLAlchemyStore:
    """基于 SQLAlchemy 的 LangGraph BaseStore 兼容实现。

    命名空间用 `.` 连接的字符串路径表示，如 ("memories", "user_profile", "alice")
    存储为 "memories.user_profile.alice"。
    """

    def __init__(self, engine):
        self._engine = engine
        _init_memory_table(engine)

    @staticmethod
    def _ns_to_str(namespace: tuple) -> str:
        if isinstance(namespace, str):
            return namespace
        return ".".join(namespace)

    def put(self, namespace: tuple, key: str, value: dict) -> None:
        ns = self._ns_to_str(namespace)
        from backend.db.database import SessionLocal

        db = SessionLocal()
        try:
            record = (
                db.query(_LangMemMemory)
                .filter(_LangMemMemory.namespace == ns, _LangMemMemory.key == key)
                .first()
            )
            if record:
                record.value = json.dumps(value, ensure_ascii=False)
                record.updated_at = None  # 由 server_default 处理
                db.execute(
                    text(
                        "UPDATE langmem_memories SET value = :val, updated_at = NOW() "
                        "WHERE namespace = :ns AND key = :key"
                    ),
                    {"val": json.dumps(value, ensure_ascii=False), "ns": ns, "key": key},
                )
            else:
                db.add(
                    _LangMemMemory(
                        namespace=ns, key=key, value=json.dumps(value, ensure_ascii=False)
                    )
                )
            db.commit()
        finally:
            db.close()

    def get(self, namespace: tuple, key: str) -> Optional[dict]:
        ns = self._ns_to_str(namespace)
        from backend.db.database import SessionLocal

        db = SessionLocal()
        try:
            record = (
                db.query(_LangMemMemory)
                .filter(_LangMemMemory.namespace == ns, _LangMemMemory.key == key)
                .first()
            )
            if record:
                return json.loads(record.value)
            return None
        finally:
            db.close()

    def delete(self, namespace: tuple, key: str) -> None:
        ns = self._ns_to_str(namespace)
        from backend.db.database import SessionLocal

        db = SessionLocal()
        try:
            db.query(_LangMemMemory).filter(
                _LangMemMemory.namespace == ns, _LangMemMemory.key == key
            ).delete(synchronize_session=False)
            db.commit()
        finally:
            db.close()

    def search(
        self,
        namespace: tuple,
        query: Optional[str] = None,
        limit: int = 10,
        **kwargs: Any,
    ) -> List[dict]:
        """在命名空间中搜索记忆。当前实现为简单前缀匹配 + 全文扫描。

        没有 pgvector 时，返回该命名空间下所有记忆，由调用方自行过滤。
        """
        ns = self._ns_to_str(namespace)
        from backend.db.database import SessionLocal

        db = SessionLocal()
        try:
            records = (
                db.query(_LangMemMemory)
                .filter(_LangMemMemory.namespace.like(f"{ns}%"))
                .limit(limit)
                .all()
            )
            results = []
            for r in records:
                value = json.loads(r.value)
                results.append(
                    {
                        "namespace": tuple(r.namespace.split(".")),
                        "key": r.key,
                        "value": value,
                        "created_at": str(r.created_at) if r.created_at else None,
                        "updated_at": str(r.updated_at) if r.updated_at else None,
                    }
                )
            # 如果有 query，做简单的文本匹配过滤
            if query:
                query_lower = query.lower()
                filtered = []
                for item in results:
                    text_content = json.dumps(item.get("value", ""), ensure_ascii=False).lower()
                    if query_lower in text_content:
                        filtered.append(item)
                results = filtered[:limit]
            return results
        finally:
            db.close()

    def list_namespaces(self, prefix: tuple = (), max_depth: int = 1) -> List[tuple]:
        ns_prefix = self._ns_to_str(prefix)
        from backend.db.database import SessionLocal

        db = SessionLocal()
        try:
            records = (
                db.query(_LangMemMemory.namespace)
                .filter(_LangMemMemory.namespace.like(f"{ns_prefix}%"))
                .distinct()
                .all()
            )
            return [tuple(r[0].split(".")) for r in records]
        finally:
            db.close()


# ---------------------------------------------------------------------------
# 全局 Store 实例（惰性初始化）
# ---------------------------------------------------------------------------
_store: Optional[_SQLAlchemyStore] = None


def get_store() -> _SQLAlchemyStore:
    """获取或创建全局 Store 实例。"""
    global _store
    if _store is None:
        from backend.db.database import engine

        _store = _SQLAlchemyStore(engine)
    return _store


# ---------------------------------------------------------------------------
# 核心 API
# ---------------------------------------------------------------------------

def create_memory_tools(user_id: str):
    """为指定用户创建 LangMem 记忆工具（manage + search）。

    返回 (manage_tool, search_tool) 元组，可直接注册到 Agent。
    """
    store = get_store()
    ns = ("memories", user_id)

    manage_tool = create_manage_memory_tool(
        namespace=ns,
        store=store,
        name="manage_memory",
    )
    search_tool = create_search_memory_tool(
        namespace=ns,
        store=store,
        name="search_memory",
    )
    return manage_tool, search_tool


def extract_conversation_memories(messages: list, user_id: str) -> list:
    """从对话消息列表中提取长期记忆并存储。

    Args:
        messages: LangChain 消息列表（HumanMessage / AIMessage）
        user_id: 用户标识

    Returns:
        提取的记忆条目列表
    """
    model = _create_chat_model()
    manager = create_memory_manager(model)

    # 转换为 LangMem 期望的格式
    langmem_messages = []
    for msg in messages:
        role = "user" if getattr(msg, "type", None) == "human" else "assistant"
        langmem_messages.append({"role": role, "content": str(msg.content)})

    # 获取该用户已有记忆（用于增量更新）
    store = get_store()
    existing = store.search(("memories", user_id), limit=50)

    # 执行提取
    result = manager.invoke({
        "messages": langmem_messages,
        "existing_memories": existing,
    })

    # 存储提取结果
    for memory in result:
        if hasattr(memory, "content") and hasattr(memory, "memory_id"):
            # 新提取的记忆
            key = memory.memory_id or f"mem_{hash(memory.content) & 0xFFFFFFFF:08x}"
            store.put(
                ("memories", user_id, "extracted"),
                key,
                {"content": memory.content, "kind": getattr(memory, "kind", "semantic")},
            )

    return result


def get_user_memories(user_id: str, query: str = "") -> str:
    """获取用户的长期记忆，格式化为可注入系统提示的文本。

    Args:
        user_id: 用户标识
        query: 可选查询词，用于过滤相关记忆

    Returns:
        格式化的记忆文本，适合拼接到系统提示
    """
    store = get_store()

    # 搜索相关记忆
    memories = store.search(("memories", user_id), query=query, limit=20)

    if not memories:
        return ""

    parts = ["以下为该用户的长期记忆（跨会话保留）："]
    for item in memories:
        value = item.get("value", {})
        if isinstance(value, dict):
            content = value.get("content", str(value))
        else:
            content = str(value)
        parts.append(f"- {content}")

    return "\n".join(parts)
