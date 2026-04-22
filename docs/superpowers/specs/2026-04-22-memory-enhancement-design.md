# 记忆系统增强设计文档

## 概述

在现有 LangMem + PostgreSQL 记忆系统基础上，增加四类记忆分类机制：
- `user`: 用户偏好（代码风格、回答详细程度）
- `feedback`: 用户纠正或验证过的做法
- `project`: 项目级约定或决策背景
- `reference`: 外部资源指针（看板、监控面板）

核心原则：
1. Agent 仅通过工具主动存储，移除自动提取逻辑
2. 边界原则：只有跨会话有价值且无法从代码推导的信息才存入 memory
3. 最小改动，保留 LangMem 核心能力

## 改动清单

### 1. 数据库层

**文件**: `backend/db/models.py`

**改动**: `LangMemMemory` 表增加 `memory_type` 字段

```python
memory_type = Column(String(20), default="user", nullable=False)
```

- 允许值：`"user"`, `"feedback"`, `"project"`, `"reference"`
- 默认值 `"user"`，兼容现有数据
- 不加索引（按 namespace 过滤已够用）

### 2. Store 层

**文件**: `backend/agent/memory_manager.py`

**改动**: `_SQLAlchemyStore` 支持按类型存储和检索

**Namespace 约定**:
- 格式：`("memories", user_id, memory_type)`
- 例如：`("memories", "alice", "feedback")` 表示 feedback 类型

**方法调整**:

1. `_put`: 从 namespace 解析 memory_type，写入对应字段
2. `_search`: 支持 namespace 前缀过滤，`("memories", user_id)` 返回所有类型，`("memories", user_id, "feedback")` 只返回该类型
3. `get_user_memories`: 增加 `memory_type` 可选参数

```python
def get_user_memories(user_id: str, memory_type: str | None = None, query: str = "") -> str:
    if memory_type:
        ns = ("memories", user_id, memory_type)
    else:
        ns = ("memories", user_id)
    # ...
```

### 3. 工具层

**新建文件**: `backend/agent/memory_tools.py`

**工具列表**:

| 工具名 | 用途 | 参数 |
|--------|------|------|
| `save_user_memory` | 存储用户偏好 | `content: str` |
| `save_feedback_memory` | 存储用户纠正 | `content: str` |
| `save_project_memory` | 存储项目约定 | `content: str` |
| `save_reference_memory` | 存储外部资源指针 | `name: str, url: str, description: str` |
| `search_memories` | 搜索记忆 | `query: str, memory_type: str \| None` |

**工具描述**（嵌入边界原则）:

```python
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
    return _save_memory(user_id, "reference", f"{name}: {url} — {description}")

@tool
def search_memories(query: str, memory_type: str | None = None) -> str:
    """搜索已存储的记忆。

    Args:
        query: 搜索关键词
        memory_type: 可选，限定类型 (user/feedback/project/reference)
    """
    # 调用 get_user_memories
```

**内部实现 `_save_memory`**:
- 调用 LangMem 的 `store.put()`
- namespace 格式：`("memories", user_id, memory_type)`
- key 使用 UUID 或时间戳，避免内容变化时的哈希冲突

### 4. Agent 集成

**文件**: `backend/agent/agent.py`

**改动**:

1. **替换工具** — `create_agent_with_memory` 使用新工具

```python
from backend.agent.memory_tools import create_typed_memory_tools

def create_agent_with_memory(user_id: str):
    # ...
    typed_tools = create_typed_memory_tools(user_id)
    agent = create_agent(
        model=model,
        tools=[get_current_weather, search_knowledge_base] + typed_tools,
        # ...
    )
```

2. **移除自动提取** — 删除 `_schedule_memory_extraction` 调用
   - `chat_with_agent`: 删除末尾调用
   - `chat_with_agent_stream`: 删除末尾调用
   - 可保留函数定义，以备后用

3. **更新 system_prompt** — 加入记忆使用指南

```python
system_prompt=(
    "You are a cute cat bot that loves to help users. "
    # ... 现有内容 ...

    "## Memory Tools\n"
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

## 迁移策略

1. 数据库迁移：新增 `memory_type` 字段，默认值 `"user"`，现有数据自动归类为 user 类型
2. 无需数据迁移脚本，SQLAlchemy 会自动添加字段（若使用 Alembic 需生成迁移脚本）
3. 向后兼容：现有调用 `get_user_memories(user_id)` 不指定类型，返回所有记忆

## 测试要点

1. 四类记忆工具的存储和检索
2. `search_memories` 按类型过滤
3. `get_user_memories` 注入系统提示时包含所有类型
4. 边界原则在工具描述中清晰表达

## 文件清单

| 文件 | 操作 |
|------|------|
| `backend/db/models.py` | 修改：增加 `memory_type` 字段 |
| `backend/agent/memory_manager.py` | 修改：Store 支持类型过滤 |
| `backend/agent/memory_tools.py` | 新建：类型化记忆工具 |
| `backend/agent/agent.py` | 修改：集成新工具、移除自动提取、更新 system_prompt |
