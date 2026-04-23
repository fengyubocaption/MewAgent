from dotenv import load_dotenv
import logging
import os
import json
import asyncio
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, SystemMessage
from backend.agent.tools import get_current_weather, search_knowledge_base, init_retrieval_state, set_rag_step_queue, _retrieval_state
from datetime import datetime
from backend.db.cache import cache
from backend.db.database import SessionLocal
from backend.db.models import User, ChatSession, ChatMessage
from backend.agent.memory_tools import create_typed_memory_tools
from backend.agent.memory_manager import get_user_memories

load_dotenv()

logger = logging.getLogger(__name__)

API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")

# 上下文窗口配置
MAX_CONTEXT_MESSAGES = 20       # 运行时保留的最近消息数
COMPRESS_THRESHOLD = 30         # 超过此条数触发压缩
COMPRESS_KEEP_RECENT = 10       # 压缩时保留的最近消息数
_SUMMARY_PREFIX = "[对话摘要] " # 摘要消息的标识前缀

class ConversationStorage:
    """对话存储（PostgreSQL + Redis）。"""

    @staticmethod
    def _messages_cache_key(user_id: str, session_id: str) -> str:
        return f"chat_messages:{user_id}:{session_id}"

    @staticmethod
    def _sessions_cache_key(user_id: str) -> str:
        return f"chat_sessions:{user_id}"

    @staticmethod
    def _to_langchain_messages(records: list[dict]) -> list:
        messages = []
        for msg_data in records:
            msg_type = msg_data.get("type")
            content = msg_data.get("content", "")
            if msg_type == "human":
                messages.append(HumanMessage(content=content))
            elif msg_type == "ai":
                messages.append(AIMessage(content=content))
            elif msg_type == "system":
                messages.append(SystemMessage(content=content))
        return messages

    @staticmethod
    def _format_messages_for_summary(messages: list) -> str:
        """将消息列表格式化为摘要输入文本"""
        lines = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "用户"
            elif isinstance(msg, AIMessage):
                role = "助手"
            elif isinstance(msg, SystemMessage):
                # 跳过已有的摘要消息，避免嵌套
                if msg.content.startswith(_SUMMARY_PREFIX):
                    continue
                role = "系统"
            else:
                continue
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

    @staticmethod
    async def _generate_summary(messages: list) -> str:
        """用 LLM 生成对话连续性摘要"""
        if not messages:
            return ""

        model = init_chat_model(
            model=MODEL,
            model_provider="openai",
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=0.3,
        )

        history_text = ConversationStorage._format_messages_for_summary(messages)
        prompt = f"""请用中文生成这段对话的连续性摘要，保留关键信息和上下文关联：

        {history_text}

        摘要要求：
        - 保留重要的用户偏好、需求、约定
        - 保留已讨论过的关键结论
        - 语言简洁，不超过 300 字
        - 摘要格式：直接输出内容，不需要前缀
        """

        response = await model.ainvoke(prompt)
        summary = response.content if hasattr(response, "content") else str(response)
        return _SUMMARY_PREFIX + summary

    @staticmethod
    def _has_existing_summary(messages: list) -> tuple[bool, int]:
        """检查是否已有摘要消息，返回 (是否存在, 摘要索引)"""
        for idx, msg in enumerate(messages):
            if isinstance(msg, SystemMessage) and msg.content.startswith(_SUMMARY_PREFIX):
                return True, idx
        return False, -1

    def save(self, user_id: str, session_id: str, messages: list, metadata: dict = None, extra_message_data: list = None):
        """保存对话"""
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return

            session = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
                .first()
            )
            if not session:
                session = ChatSession(user_id=user.id, session_id=session_id, metadata_json=metadata or {})
                db.add(session)
                db.flush()
            else:
                session.metadata_json = metadata or {}

            db.query(ChatMessage).filter(ChatMessage.session_ref_id == session.id).delete(synchronize_session=False)

            serialized = []
            now = datetime.utcnow()
            for idx, msg in enumerate(messages):
                rag_trace = None
                if extra_message_data and idx < len(extra_message_data):
                    extra = extra_message_data[idx] or {}
                    rag_trace = extra.get("rag_trace")

                db.add(
                    ChatMessage(
                        session_ref_id=session.id,
                        message_type=msg.type,
                        content=str(msg.content),
                        timestamp=now,
                        rag_trace=rag_trace,
                    )
                )
                serialized.append(
                    {
                        "type": msg.type,
                        "content": str(msg.content),
                        "timestamp": now.isoformat(),
                        "rag_trace": rag_trace,
                    }
                )

            session.updated_at = now
            db.commit()

            cache.set_json(self._messages_cache_key(user_id, session_id), serialized)
            cache.delete(self._sessions_cache_key(user_id))
        finally:
            db.close()

    async def save_with_compress(self, user_id: str, session_id: str, messages: list, extra_message_data: list = None):
        """保存对话，超长时自动压缩上下文

        当消息数 > COMPRESS_THRESHOLD 时：
        - 将早期消息压缩为一条摘要
        - 保留最近 COMPRESS_KEEP_RECENT 条消息
        - 摘要 + 最近消息 = 压缩后的完整上下文
        """
        if len(messages) > COMPRESS_THRESHOLD:
            logger.info("触发上下文压缩: %d 条消息 → 摘要 + %d 条", len(messages), COMPRESS_KEEP_RECENT)

            # 分离：早期消息（待压缩） + 最近消息（保留）
            split_idx = len(messages) - COMPRESS_KEEP_RECENT
            messages_to_compress = messages[:split_idx]
            recent_messages = messages[split_idx:]

            # 检查早期消息中是否已有摘要，有的话合并
            has_summary, summary_idx = self._has_existing_summary(messages_to_compress)
            if has_summary:
                # 把旧摘要也纳入重新压缩
                messages_to_compress = messages_to_compress
            else:
                # 纯新消息，直接压缩
                pass

            try:
                # 生成摘要
                summary_text = await self._generate_summary(messages_to_compress)
                summary_msg = SystemMessage(content=summary_text)

                # 压缩后：摘要 + 最近消息
                messages = [summary_msg] + recent_messages

                # 同步调整 extra_message_data（只保留最近消息对应的部分）
                if extra_message_data:
                    extra_message_data = [None] + extra_message_data[split_idx:]

                logger.info("上下文压缩完成，摘要长度: %d 字", len(summary_text))
            except Exception as e:
                logger.warning("上下文压缩失败，使用原始消息保存: %s", e)
                # 压缩失败回退：只保留最近的消息
                messages = messages[-MAX_CONTEXT_MESSAGES:]
                if extra_message_data:
                    extra_message_data = extra_message_data[-MAX_CONTEXT_MESSAGES:]

        # 执行保存
        self.save(user_id, session_id, messages, extra_message_data=extra_message_data)

    def load(self, user_id: str, session_id: str) -> list:
        """加载对话（带上下文截断，摘要消息始终保留）

        策略：如果有摘要消息，摘要 + 最近 MAX_CONTEXT_MESSAGES-1 条
              否则只保留最近 MAX_CONTEXT_MESSAGES 条
        """
        def _truncate_with_summary(records: list) -> list:
            if not records:
                return []

            # 检查第一条是否是摘要
            first = records[0]
            has_summary = (
                first.get("type") == "system" and
                first.get("content", "").startswith(_SUMMARY_PREFIX)
            )

            # 增加长度判断，避免切片重叠导致摘要翻倍
            if len(records) <= MAX_CONTEXT_MESSAGES:
                return records

            if has_summary:
                # 摘要 + 最近 N-1 条消息
                return [first] + records[-(MAX_CONTEXT_MESSAGES - 1):]
            else:
                # 无摘要，只取最近 N 条
                return records[-MAX_CONTEXT_MESSAGES:]

        cached = cache.get_json(self._messages_cache_key(user_id, session_id))
        if cached is not None:
            return self._to_langchain_messages(_truncate_with_summary(cached))

        records = self.get_session_messages(user_id, session_id)
        cache.set_json(self._messages_cache_key(user_id, session_id), records)
        return self._to_langchain_messages(_truncate_with_summary(records))

    def list_sessions(self, user_id: str) -> list:
        """列出用户的所有会话"""
        return [item["session_id"] for item in self.list_session_infos(user_id)]

    def list_session_infos(self, user_id: str) -> list[dict]:
        cached = cache.get_json(self._sessions_cache_key(user_id))
        if cached is not None:
            return cached

        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return []

            sessions = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id)
                .order_by(ChatSession.updated_at.desc())
                .all()
            )
            result = []
            for s in sessions:
                count = db.query(ChatMessage).filter(ChatMessage.session_ref_id == s.id).count()
                result.append(
                    {
                        "session_id": s.session_id,
                        "updated_at": s.updated_at.isoformat(),
                        "message_count": count,
                    }
                )
            cache.set_json(self._sessions_cache_key(user_id), result)
            return result
        finally:
            db.close()

    def get_session_messages(self, user_id: str, session_id: str) -> list[dict]:
        cached = cache.get_json(self._messages_cache_key(user_id, session_id))
        if cached is not None:
            return cached

        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return []
            session = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
                .first()
            )
            if not session:
                return []

            rows = (
                db.query(ChatMessage)
                .filter(ChatMessage.session_ref_id == session.id)
                .order_by(ChatMessage.id.asc())
                .all()
            )
            result = [
                {
                    "type": row.message_type,
                    "content": row.content,
                    "timestamp": row.timestamp.isoformat(),
                    "rag_trace": row.rag_trace,
                }
                for row in rows
            ]
            cache.set_json(self._messages_cache_key(user_id, session_id), result)
            return result
        finally:
            db.close()

    def delete_session(self, user_id: str, session_id: str) -> bool:
        """删除指定用户的会话，返回是否删除成功"""
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return False
            session = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
                .first()
            )
            if not session:
                return False

            db.delete(session)
            db.commit()
            cache.delete(self._messages_cache_key(user_id, session_id))
            cache.delete(self._sessions_cache_key(user_id))
            return True
        finally:
            db.close()



def _build_system_prompt(user_memories: str) -> str:
    """构建包含用户长期记忆的系统提示词。

    记忆直接注入到系统提示词中，而非污染消息历史，
    这样不会在上下文压缩时出现问题。
    """
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

    if user_memories:
        return base_prompt + "\n\n## User Long-Term Memory\n" + user_memories

    return base_prompt


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
    user_memories = get_user_memories(user_id)

    agent = create_agent(
        model=model,
        tools=[get_current_weather, search_knowledge_base] + typed_tools,
        system_prompt=_build_system_prompt(user_memories),
    )
    return agent, model


storage = ConversationStorage()


async def chat_with_agent_stream(user_text: str, user_id: str = "default_user", session_id: str = "default_session"):
    """使用 Agent 处理用户消息并流式返回响应。

    架构：使用统一输出队列 + 后台任务，确保 RAG 检索步骤在工具执行期间实时推送，
    而非等待工具完成后才显示。
    """
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

    messages.append(HumanMessage(content=user_text))

    full_response = ""
    # 使用带记忆工具的 Agent 实例
    memory_agent, _ = create_agent_with_memory(user_id)

    async def _agent_worker():
        """后台任务：运行 agent 并将内容 chunk 推入输出队列。"""
        nonlocal full_response
        try:
            async for msg, _ in memory_agent.astream(
                {"messages": messages},
                stream_mode="messages",
                config={"recursion_limit": 12},
            ):
                if not isinstance(msg, AIMessageChunk):
                    continue
                if getattr(msg, "tool_call_chunks", None):
                    continue

                content = ""
                if isinstance(msg.content, str):
                    content = msg.content
                elif isinstance(msg.content, list):
                    for block in msg.content:
                        if isinstance(block, str):
                            content += block
                        elif isinstance(block, dict) and block.get("type") == "text":
                            content += block.get("text", "")

                if content:
                    full_response += content
                    await output_queue.put({"type": "content", "content": content})
        except Exception as e:
            await output_queue.put({"type": "error", "content": str(e)})
        finally:
            # 哨兵：通知主循环 agent 已完成
            await output_queue.put(None)

    # 启动后台任务
    agent_task = asyncio.create_task(_agent_worker())

    try:
        # 主循环：持续从统一队列取事件并 yield SSE
        # RAG 步骤在工具执行期间通过 call_soon_threadsafe 实时入队，不需要等 agent 产出 chunk
        while True:
            event = await output_queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"
    except GeneratorExit:
        # 客户端断开连接（AbortController）时，FastAPI 会向此生成器抛出 GeneratorExit
        # 我们必须在此处取消后台任务
        agent_task.cancel()
        try:
            await agent_task
        except asyncio.CancelledError:
            pass  # 任务已成功取消
        raise  # 重新抛出 GeneratorExit 以便 FastAPI 正确处理关闭
    finally:
        # 正常结束或异常退出时清理
        set_rag_step_queue(None)
        if not agent_task.done():
             agent_task.cancel()

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

    # 发送结束信号
    yield "data: [DONE]\n\n"

    # 保存对话（带上下文压缩）
    messages.append(AIMessage(content=full_response))
    extra_message_data = [None] * (len(messages) - 1) + [{"rag_trace": rag_trace}]
    await storage.save_with_compress(user_id, session_id, messages, extra_message_data=extra_message_data)
