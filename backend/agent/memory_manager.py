"""LangMem 记忆管理模块 — 基于 PostgreSQL + 内存 fallback"""
import json
import logging
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM 模型配置（用于记忆提取）
# ---------------------------------------------------------------------------
_API_KEY = os.getenv("ARK_API_KEY")
_MODEL = os.getenv("MODEL")
_BASE_URL = os.getenv("BASE_URL")


# ---------------------------------------------------------------------------
# PostgreSQL Store 实现
# ---------------------------------------------------------------------------
# LangGraph 没有内置 PostgreSQL store，这里用 SQLAlchemy 实现一个轻量版本。
# 记忆以 JSON 形式存储在 PostgreSQL 的 langmem_memories 表中。

from backend.db.models import LangMemMemory
from sqlalchemy import text
from langgraph.store.base import BaseStore, Item, SearchItem, GetOp, PutOp, SearchOp, ListNamespacesOp


def _init_memory_table(engine):
    """在数据库中创建 langmem_memories 表。"""
    try:
        LangMemMemory.__table__.create(bind=engine, checkfirst=True)
    except Exception as e:
        logger.warning("langmem_memories 表创建失败（可能已存在）: %s", e)


class _SQLAlchemyStore(BaseStore):
    """基于 SQLAlchemy 的 LangGraph BaseStore 实现。

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

    def batch(self, ops) -> list:
        """批量执行操作。"""
        results = []
        for op in ops:
            if isinstance(op, GetOp):
                results.append(self._get(op.namespace, op.key))
            elif isinstance(op, PutOp):
                if op.value is None:
                    self._delete(op.namespace, op.key)
                else:
                    self._put(op.namespace, op.key, op.value)
                results.append(None)
            elif isinstance(op, SearchOp):
                results.append(self._search(op.namespace_prefix, op.query, op.limit))
            elif isinstance(op, ListNamespacesOp):
                prefix = None
                for cond in op.match_conditions:
                    if cond.match_type == "prefix":
                        prefix = cond.path
                        break
                results.append(self._list_namespaces(prefix or (), op.limit))
            else:
                raise NotImplementedError(f"Unsupported operation: {type(op)}")
        return results

    async def abatch(self, ops) -> list:
        """异步批量执行（同步实现）。"""
        return self.batch(ops)

    def _put(self, namespace: tuple, key: str, value: dict) -> None:
        ns = self._ns_to_str(namespace)
        # 从 namespace 解析 memory_type: ("memories", user_id, memory_type)
        memory_type = "user"  # 默认值
        if isinstance(namespace, tuple) and len(namespace) >= 3:
            memory_type = namespace[2]

        from backend.db.database import SessionLocal

        db = SessionLocal()
        try:
            record = (
                db.query(LangMemMemory)
                .filter(LangMemMemory.namespace == ns, LangMemMemory.key == key)
                .first()
            )
            if record:
                # 直接用原生 SQL 更新，确保 updated_at 由数据库刷新
                db.execute(
                    text(
                        "UPDATE langmem_memories SET value = :val, memory_type = :mt, updated_at = NOW() "
                        "WHERE namespace = :ns AND key = :key"
                    ),
                    {"val": json.dumps(value, ensure_ascii=False), "mt": memory_type, "ns": ns, "key": key},
                )
            else:
                db.add(
                    LangMemMemory(
                        namespace=ns, key=key, value=json.dumps(value, ensure_ascii=False),
                        memory_type=memory_type
                    )
                )
            db.commit()
        finally:
            db.close()

    def _get(self, namespace: tuple, key: str) -> Item | None:
        ns = self._ns_to_str(namespace)
        from backend.db.database import SessionLocal

        db = SessionLocal()
        try:
            record = (
                db.query(LangMemMemory)
                .filter(LangMemMemory.namespace == ns, LangMemMemory.key == key)
                .first()
            )
            if record:
                return Item(
                    value=json.loads(record.value),
                    key=key,
                    namespace=tuple(record.namespace.split(".")),
                    created_at=record.created_at,
                    updated_at=record.updated_at,
                )
            return None
        finally:
            db.close()

    def _delete(self, namespace: tuple, key: str) -> None:
        ns = self._ns_to_str(namespace)
        from backend.db.database import SessionLocal

        db = SessionLocal()
        try:
            db.query(LangMemMemory).filter(
                LangMemMemory.namespace == ns, LangMemMemory.key == key
            ).delete(synchronize_session=False)
            db.commit()
        finally:
            db.close()

    def _search(self, namespace_prefix: tuple, query: str | None, limit: int) -> list[SearchItem]:
        """基于命名空间前缀匹配 + 全文字符串扫描。"""
        ns = self._ns_to_str(namespace_prefix)

        # 判断是否指定了 memory_type: ("memories", user_id, memory_type) vs ("memories", user_id)
        filter_by_type = len(namespace_prefix) >= 3
        memory_type_filter = namespace_prefix[2] if filter_by_type else None

        from backend.db.database import SessionLocal

        db = SessionLocal()
        try:
            q = db.query(LangMemMemory).filter(LangMemMemory.namespace.like(f"{ns}%"))

            # 如果指定了 memory_type，额外过滤
            if memory_type_filter:
                q = q.filter(LangMemMemory.memory_type == memory_type_filter)

            records = q.limit(limit).all()
            results = []
            for r in records:
                item = SearchItem(
                    value=json.loads(r.value),
                    key=r.key,
                    namespace=tuple(r.namespace.split(".")),
                    created_at=r.created_at,
                    updated_at=r.updated_at,
                )
                results.append(item)
            if query:
                query_lower = query.lower()
                filtered = []
                for item in results:
                    text_content = json.dumps(item.value, ensure_ascii=False).lower()
                    if query_lower in text_content:
                        filtered.append(item)
                results = filtered[:limit]
            return results
        finally:
            db.close()

    def _list_namespaces(self, prefix: tuple, limit: int) -> list[tuple]:
        ns_prefix = self._ns_to_str(prefix) if prefix else ""
        from backend.db.database import SessionLocal

        db = SessionLocal()
        try:
            q = db.query(LangMemMemory.namespace).distinct()
            if ns_prefix:
                q = q.filter(LangMemMemory.namespace.like(f"{ns_prefix}%"))
            q = q.limit(limit)
            records = q.all()
            return [tuple(r[0].split(".")) for r in records]
        finally:
            db.close()


# ---------------------------------------------------------------------------
# 全局 Store 实例（惰性初始化）
# ---------------------------------------------------------------------------
_store: Optional[BaseStore] = None


def get_store() -> BaseStore:
    """获取或创建全局 Store 实例。"""
    global _store
    if _store is None:
        from backend.db.database import engine

        _store = _SQLAlchemyStore(engine)
    return _store


# ---------------------------------------------------------------------------
# 核心 API
# ---------------------------------------------------------------------------

def get_user_memories(user_id: str, memory_type: str | None = None, query: str = "") -> str:
    """获取用户的长期记忆，格式化为可注入系统提示的文本。

    Args:
        user_id: 用户标识
        memory_type: 可选，限定类型 (user/feedback/project/reference)
        query: 可选查询词，用于过滤相关记忆

    Returns:
        格式化的记忆文本，适合拼接到系统提示
    """
    store = get_store()

    # 构建命名空间前缀
    if memory_type:
        ns = ("memories", user_id, memory_type)
    else:
        ns = ("memories", user_id)

    # 搜索相关记忆
    memories = store.search(ns, query=query, limit=20)

    if not memories:
        return ""

    parts = ["以下为该用户的长期记忆（跨会话保留）："]
    for item in memories:
        # item 是 SearchItem，.value 是 dict
        value = item.value
        if isinstance(value, dict):
            content = value.get("content", str(value))
        else:
            content = str(value)
        parts.append(f"- {content}")

    return "\n".join(parts)
