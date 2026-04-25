"""类型化记忆工具 — Agent 可调用的记忆存储/检索工具。"""
from langchain_core.tools import tool
from backend.agent.memory_manager import save_memory, get_user_memories, MEMORY_TYPES


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
        """Store user preferences (e.g., coding style, response verbosity).

        Only store information valuable across sessions.
        DO NOT store: file paths, code snippets, temporary state, task progress.
        """
        return save_memory(user_id, "user", content)

    @tool
    def save_feedback_memory(content: str) -> str:
        """Store user corrections or validated approaches.

        Suggested format: Rule + Why + How to apply.
        Example: "Don't use mock database in tests. Why: Caused production incident. How to apply: Integration tests must use real database."
        """
        return save_memory(user_id, "feedback", content)

    @tool
    def save_project_memory(content: str) -> str:
        """Store project-level conventions or decision context.

        Example: "Auth middleware rewrite was driven by compliance requirements, not tech debt cleanup."
        """
        return save_memory(user_id, "project", content)

    @tool
    def save_reference_memory(name: str, url: str, description: str) -> str:
        """Store external resource pointers.

        Examples: Kanban board location, monitoring dashboard URL, documentation links.
        """
        content = f"{name}: {url} — {description}"
        return save_memory(user_id, "reference", content)

    @tool
    def search_memories(query: str, memory_type: str | None = None) -> str:
        """Search stored memories.

        Args:
            query: Search keywords
            memory_type: Optional, filter by type (user/feedback/project/reference)
        """
        if memory_type and memory_type not in MEMORY_TYPES:
            return f"Error: Unsupported memory type '{memory_type}'. Allowed: {MEMORY_TYPES}"

        result = get_user_memories(user_id, memory_type=memory_type, query=query)
        if not result:
            return "No relevant memories found."
        return result

    return [save_user_memory, save_feedback_memory, save_project_memory,
            save_reference_memory, search_memories]
