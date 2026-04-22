"""类型化记忆工具 — 封装 LangMem，支持四类记忆分类。"""
import uuid
from langchain_core.tools import tool
from backend.agent.memory_manager import get_store, get_user_memories

# 允许的记忆类型
MEMORY_TYPES = ("user", "feedback", "project", "reference")


def _save_memory(user_id: str, memory_type: str, content: str) -> str:
    """内部方法：存储记忆到 LangMem Store。"""
    if memory_type not in MEMORY_TYPES:
        return f"错误：不支持的记忆类型 '{memory_type}'，允许值：{MEMORY_TYPES}"

    store = get_store()
    ns = ("memories", user_id, memory_type)
    key = str(uuid.uuid4())

    store.put(ns, key, {"content": content})
    return f"已存储 {memory_type} 类型记忆"


def create_typed_memory_tools(user_id: str):
    """创建带类型标注的记忆工具。

    Args:
        user_id: 用户标识，用于隔离不同用户的记忆

    Returns:
        工具列表：[save_user_memory, save_feedback_memory, save_project_memory,
                  save_reference_memory, search_memories]
    """

    @tool
    def save_user_memory(content: str) -> str:
        """存储用户偏好信息（如代码风格、回答详细程度偏好）。

        仅存储跨会话仍有价值的信息。
        不要存储：文件路径、代码片段、临时状态、当前任务进度。
        """
        return _save_memory(user_id, "user", content)

    @tool
    def save_feedback_memory(content: str) -> str:
        """存储用户纠正或验证过的做法。

        格式建议：规则 + Why + How to apply。
        例如："不要在测试中使用 mock 数据库。Why: 曾导致生产环境故障。How to apply: 集成测试必须连真实数据库。"
        """
        return _save_memory(user_id, "feedback", content)

    @tool
    def save_project_memory(content: str) -> str:
        """存储项目级约定或决策背景。

        例如："auth 中间件重写是因为合规要求，不是技术债清理。"
        """
        return _save_memory(user_id, "project", content)

    @tool
    def save_reference_memory(name: str, url: str, description: str) -> str:
        """存储外部资源指针。

        例如：看板位置、监控面板 URL、文档链接。
        """
        content = f"{name}: {url} — {description}"
        return _save_memory(user_id, "reference", content)

    @tool
    def search_memories(query: str, memory_type: str | None = None) -> str:
        """搜索已存储的记忆。

        Args:
            query: 搜索关键词
            memory_type: 可选，限定类型 (user/feedback/project/reference)
        """
        if memory_type and memory_type not in MEMORY_TYPES:
            return f"错误：不支持的记忆类型 '{memory_type}'，允许值：{MEMORY_TYPES}"

        result = get_user_memories(user_id, memory_type=memory_type, query=query)
        if not result:
            return "未找到相关记忆"
        return result

    return [save_user_memory, save_feedback_memory, save_project_memory,
            save_reference_memory, search_memories]
