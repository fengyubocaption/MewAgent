"""会话管理服务 — 会话列表、消息、删除。"""

from fastapi import HTTPException

from backend.schemas import (
    MessageInfo,
    SessionDeleteResponse,
    SessionInfo,
    SessionListResponse,
    SessionMessagesResponse,
)


class SessionService:
    """会话管理业务逻辑。"""

    def __init__(self, storage):
        self.storage = storage

    def get_messages(self, username: str, session_id: str) -> SessionMessagesResponse:
        """获取指定会话的所有消息。"""
        messages = [
            MessageInfo(
                type=msg["type"],
                content=msg["content"],
                timestamp=msg["timestamp"],
                rag_trace=msg.get("rag_trace"),
            )
            for msg in self.storage.get_session_messages(username, session_id)
        ]
        return SessionMessagesResponse(messages=messages)

    def list_sessions(self, username: str) -> SessionListResponse:
        """获取当前用户的所有会话列表。"""
        sessions = [SessionInfo(**item) for item in self.storage.list_session_infos(username)]
        sessions.sort(key=lambda x: x.updated_at, reverse=True)
        return SessionListResponse(sessions=sessions)

    def delete_session(self, username: str, session_id: str) -> SessionDeleteResponse:
        """删除当前用户的指定会话。"""
        deleted = self.storage.delete_session(username, session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="会话不存在")
        return SessionDeleteResponse(session_id=session_id, message="成功删除会话")
