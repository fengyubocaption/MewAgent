"""API Schema 定义 — 按业务域分模块，统一导出。"""

from backend.schemas.auth import (
    AuthResponse,
    CurrentUserResponse,
    LoginRequest,
    RegisterRequest,
)
from backend.schemas.chat import (
    ChatRequest,
    ChatResponse,
    MessageInfo,
    RagTrace,
    RetrievedChunk,
)
from backend.schemas.sessions import (
    SessionDeleteResponse,
    SessionInfo,
    SessionListResponse,
    SessionMessagesResponse,
)
from backend.schemas.documents import (
    DocumentDeleteResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentUploadResponse,
)
from backend.schemas.resume import (
    ResumeDeleteResponse,
    ResumeDetailResponse,
    ResumeInfo,
    ResumeListResponse,
    ResumeUploadResponse,
)
from backend.schemas.jd import (
    JDCreateRequest,
    JDCreateResponse,
    JDDeleteResponse,
    JDDetailResponse,
    JDInfo,
    JDListResponse,
)

__all__ = [
    # auth
    "AuthResponse",
    "CurrentUserResponse",
    "LoginRequest",
    "RegisterRequest",
    # chat
    "ChatRequest",
    "ChatResponse",
    "MessageInfo",
    "RagTrace",
    "RetrievedChunk",
    # sessions
    "SessionDeleteResponse",
    "SessionInfo",
    "SessionListResponse",
    "SessionMessagesResponse",
    # documents
    "DocumentDeleteResponse",
    "DocumentInfo",
    "DocumentListResponse",
    "DocumentUploadResponse",
    # resume
    "ResumeDeleteResponse",
    "ResumeDetailResponse",
    "ResumeInfo",
    "ResumeListResponse",
    "ResumeUploadResponse",
    # jd
    "JDCreateRequest",
    "JDCreateResponse",
    "JDDeleteResponse",
    "JDDetailResponse",
    "JDInfo",
    "JDListResponse",
]
