# 记忆系统增强实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 LangMem + PostgreSQL 基础上增加四类记忆分类（user/feedback/project/reference），Agent 仅通过工具主动存储。

**Architecture:** 在数据库层增加 memory_type 字段，Store 层支持按类型过滤，新建类型化工具包装 LangMem，Agent 集成新工具并移除自动提取逻辑。

**Tech Stack:** SQLAlchemy, LangMem, LangChain Tools, PostgreSQL

---

## 文件结构

| 文件 | 职责 |
|------|------|
| `backend/db/models.py` | ORM 模型，增加 memory_type 字段 |
| `backend/agent/memory_manager.py` | Store 层，支持类型过滤 |
| `backend/agent/memory_tools.py` | 新建，类型化记忆工具 |
| `backend/agent/agent.py` | Agent 集成，移除自动提取 |

---

### Task 1: 数据库模型增加 memory_type 字段

**Files:**
- Modify: `backend/db/models.py:65-80`

- [ ] **Step 1: 修改 LangMemMemory 模型**

在 `LangMemMemory` 类中增加 `memory_type` 字段：

```python
class LangMemMemory(Base):
    """LangMem 长期记忆存储表。"""

    __tablename__ = "langmem_memories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    namespace: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    key: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    memory_type: Mapped[str] = mapped_column(String(20), default="user", nullable=False)
    created_at = mapped_column(DateTime, server_default=text("NOW()"))
    updated_at = mapped_column(DateTime, server_default=text("NOW()"))

    __table_args__ = (
        Index("ix_langmem_ns_key", "namespace", "key"),
    )
```

- [ ] **Step 2: 验证语法正确**

运行: `uv run python -c "from backend.db.models import LangMemMemory; print('OK')"`

期望: 输出 `OK`

- [ ] **Step 3: 提交**

```bash
git add backend/db/models.py
git commit -m "$(cat <<'EOF'
feat(db): LangMemMemory 表增加 memory_type 字段

支持四类记忆分类：user/feedback/project/reference
默认值 user，向后兼容现有数据

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Store 层支持 memory_type

**Files:**
- Modify: `backend/agent/memory_manager.py`

- [ ] **Step 1: 修改 _put 方法，解析并存储 memory_type**

找到 `_put` 方法（约第 106 行），替换为：

```python
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
```

- [ ] **Step 2: 修改 _search 方法，支持按类型过滤**

找到 `_search` 方法（约第 172 行），在查询时增加 memory_type 过滤：

```python
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
```

- [ ] **Step 3: 修改 get_user_memories 函数，增加 memory_type 参数**

找到 `get_user_memories` 函数（约第 325 行），替换为：

```python
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
```

- [ ] **Step 4: 验证语法正确**

运行: `uv run python -c "from backend.agent.memory_manager import get_user_memories; print('OK')"`

期望: 输出 `OK`

- [ ] **Step 5: 提交**

```bash
git add backend/agent/memory_manager.py
git commit -m "$(cat <<'EOF'
feat(memory): Store 层支持 memory_type 分类存储和检索

- _put 从 namespace 解析 memory_type
- _search 支持按类型过滤
- get_user_memories 增加 memory_type 参数

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: 创建类型化记忆工具

**Files:**
- Create: `backend/agent/memory_tools.py`

- [ ] **Step 1: 创建 memory_tools.py 文件**

```python
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
```

- [ ] **Step 2: 验证语法正确**

运行: `uv run python -c "from backend.agent.memory_tools import create_typed_memory_tools; tools = create_typed_memory_tools('test'); print(f'{len(tools)} tools created')"`

期望: 输出 `5 tools created`

- [ ] **Step 3: 提交**

```bash
git add backend/agent/memory_tools.py
git commit -m "$(cat <<'EOF'
feat(agent): 新建类型化记忆工具

- save_user_memory: 用户偏好
- save_feedback_memory: 用户纠正
- save_project_memory: 项目约定
- save_reference_memory: 外部资源指针
- search_memories: 搜索记忆

工具描述嵌入边界原则

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Agent 集成新工具

**Files:**
- Modify: `backend/agent/agent.py`

- [ ] **Step 1: 导入新工具模块**

在文件顶部导入区域（约第 14 行后），增加：

```python
from backend.agent.memory_tools import create_typed_memory_tools
```

- [ ] **Step 2: 修改 create_agent_with_memory 函数**

找到 `create_agent_with_memory` 函数（约第 251 行），替换为：

```python
def create_agent_with_memory(user_id: str):
    """创建带类型化记忆工具的 Agent 实例。

    在基础工具（天气、知识库）之上，加入类型化记忆工具，
    让 Agent 能在对话中主动存储和检索用户长期记忆。
    """
    model = init_chat_model(
        model=MODEL,
        model_provider="openai",
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0.3,
        stream_usage=True,
    )

    typed_tools = create_typed_memory_tools(user_id)

    agent = create_agent(
        model=model,
        tools=[get_current_weather, search_knowledge_base] + typed_tools,
        system_prompt=(
            "You are a cute cat bot that loves to help users. "
            "When responding, you may use tools to assist. "
            "Use search_knowledge_base when users ask document/knowledge questions. "
            "Do not call the same tool repeatedly in one turn. At most one knowledge tool call per turn. "
            "Once you call search_knowledge_base and receive its result, you MUST immediately produce the Final Answer based on that result. "
            "After receiving search_knowledge_base result, you MUST NOT call any tool again (including get_current_weather or search_knowledge_base). "
            "If the retrieved context is insufficient, answer honestly that you don't know instead of making up facts. "
            "If tool results include a Step-back Question/Answer, use that general principle to reason and answer, "
            "but do not reveal chain-of-thought. "
            "If you don't know the answer, admit it honestly.\n"
            "\n"
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
        ),
    )
    return agent, model
```

- [ ] **Step 3: 移除 chat_with_agent 中的自动提取调用**

找到 `chat_with_agent` 函数（约第 345 行），删除末尾的自动提取调用：

找到这行并删除：
```python
# 后台提取长期记忆（不阻塞返回）
_schedule_memory_extraction(user_id, messages)
```

- [ ] **Step 4: 移除 chat_with_agent_stream 中的自动提取调用**

找到 `chat_with_agent_stream` 函数（约第 393 行），删除末尾的自动提取调用：

找到这行并删除：
```python
# 后台提取长期记忆（不阻塞返回）
_schedule_memory_extraction(user_id, messages)
```

- [ ] **Step 5: 验证语法正确**

运行: `uv run python -c "from backend.agent.agent import create_agent_with_memory; print('OK')"`

期望: 输出 `OK`

- [ ] **Step 6: 提交**

```bash
git add backend/agent/agent.py
git commit -m "$(cat <<'EOF'
feat(agent): 集成类型化记忆工具，移除自动提取

- create_agent_with_memory 使用新工具
- system_prompt 加入记忆使用指南和边界原则
- 移除 chat_with_agent 和 chat_with_agent_stream 中的自动提取

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: 集成测试

**Files:**
- 无需修改

- [ ] **Step 1: 启动服务**

运行: `uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload`

期望: 服务正常启动

- [ ] **Step 2: 测试记忆工具存储**

在 Swagger UI (`http://127.0.0.1:8000/docs`) 中：

1. 登录获取 token
2. 发送聊天请求，让 Agent 存储一条记忆：
   ```
   请记住：我喜欢用中文回答问题
   ```

期望: Agent 调用 `save_user_memory` 工具并返回确认

- [ ] **Step 3: 测试记忆检索**

发送聊天请求：
```
你还记得我喜欢什么吗？
```

期望: Agent 调用 `search_memories` 并正确回忆起之前存储的信息

- [ ] **Step 4: 提交最终版本**

```bash
git add -A
git commit -m "$(cat <<'EOF'
feat: 记忆系统增强完成

- 四类记忆分类：user/feedback/project/reference
- Agent 主动存储，移除自动提取
- 边界原则约束

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## 检查清单

- [ ] 所有文件修改完成
- [ ] 语法验证通过
- [ ] 服务正常启动
- [ ] 记忆工具可正常调用
- [ ] 记忆检索正常工作
- [ ] 所有提交完成
