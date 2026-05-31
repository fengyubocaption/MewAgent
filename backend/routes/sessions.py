from fastapi import APIRouter, Depends, HTTPException

from backend.agent.agent import storage
from backend.core.rate_limit import rate_limit
from backend.core.security import get_current_user
from backend.schemas import (
    SessionDeleteResponse,
    SessionListResponse,
    SessionMessagesResponse,
)
from backend.services.session_service import SessionService
from backend.db.models import User

router = APIRouter()

session_service = SessionService(storage)


@router.get("/sessions/{session_id}", response_model=SessionMessagesResponse, dependencies=[Depends(rate_limit("read", 30, 60))])
async def get_session_messages(session_id: str, current_user: User = Depends(get_current_user)):
    """获取指定会话的所有消息"""
    try:
        return session_service.get_messages(current_user.username, session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions", response_model=SessionListResponse, dependencies=[Depends(rate_limit("read", 30, 60))])
async def list_sessions(current_user: User = Depends(get_current_user)):
    """获取当前用户的所有会话列表"""
    try:
        return session_service.list_sessions(current_user.username)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}", response_model=SessionDeleteResponse)
async def delete_session(session_id: str, current_user: User = Depends(get_current_user)):
    """删除当前用户的指定会话"""
    try:
        return session_service.delete_session(current_user.username, session_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
