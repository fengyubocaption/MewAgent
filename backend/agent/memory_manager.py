"""记忆管理模块 — 直接操作 PostgreSQL，支持四类记忆分类。"""
import json
import logging
import uuid

from backend.db.database import SessionLocal
from backend.db.models import LangMemMemory

logger = logging.getLogger(__name__)

# 允许的记忆类型
MEMORY_TYPES = ("user", "feedback", "project", "reference")


def save_memory(user_id: str, memory_type: str, content: str) -> str:
    """存储记忆到数据库。

    Args:
        user_id: 用户标识
        memory_type: 记忆类型 (user/feedback/project/reference)
        content: 记忆内容

    Returns:
        操作结果消息
    """
    if not memory_type or memory_type not in MEMORY_TYPES:
        return f"错误：不支持的记忆类型 '{memory_type}'，允许值：{MEMORY_TYPES}"

    namespace = f"memories.{user_id}.{memory_type}"
    key = str(uuid.uuid4())

    db = SessionLocal()
    try:
        db.add(LangMemMemory(
            namespace=namespace,
            key=key,
            value=json.dumps({"content": content}, ensure_ascii=False),
            memory_type=memory_type,
        ))
        db.commit()
        return f"已存储 {memory_type} 类型记忆"
    except Exception as e:
        db.rollback()
        logger.error("存储记忆失败: %s", e)
        return f"存储失败: {e}"
    finally:
        db.close()


def get_user_memories(user_id: str, memory_type: str | None = None, query: str = "") -> str:
    """获取用户的长期记忆，格式化为可注入系统提示的文本。

    Args:
        user_id: 用户标识
        memory_type: 可选，限定类型 (user/feedback/project/reference)
        query: 可选查询词，用于过滤相关记忆

    Returns:
        格式化的记忆文本，适合拼接到系统提示
    """
    db = SessionLocal()
    try:
        # 构建命名空间前缀
        if memory_type:
            ns_prefix = f"memories.{user_id}.{memory_type}"
            q = db.query(LangMemMemory).filter(
                LangMemMemory.namespace == ns_prefix
            )
        else:
            ns_prefix = f"memories.{user_id}."
            q = db.query(LangMemMemory).filter(
                LangMemMemory.namespace.like(f"{ns_prefix}%")
            )

        records = q.limit(20).all()

        if not records:
            return ""

        # 如果有 query，在内存中过滤
        if query:
            query_lower = query.lower()
            filtered = []
            for r in records:
                text_content = r.value.lower()
                if query_lower in text_content:
                    filtered.append(r)
            records = filtered

        if not records:
            return ""

        parts = ["以下为该用户的长期记忆（跨会话保留）："]
        for r in records:
            try:
                value = json.loads(r.value)
                content = value.get("content", str(value))
            except json.JSONDecodeError:
                content = r.value
            parts.append(f"- {content}")

        return "\n".join(parts)
    finally:
        db.close()
