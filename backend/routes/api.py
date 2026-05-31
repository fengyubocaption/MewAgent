from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.core.rate_limit import rate_limit
from backend.db.database import get_db
from backend.core.security import get_current_user
from backend.db.models import User
from backend.schemas import (
    AuthResponse,
    CurrentUserResponse,
    LoginRequest,
    RegisterRequest,
)
from backend.services.auth_service import AuthService

# ---- 子路由模块 ----
from backend.routes.chat import router as chat_router
from backend.routes.sessions import router as sessions_router
from backend.routes.documents import router as documents_router
from backend.routes.resume import router as resume_router
from backend.routes.jd import router as jd_router

router = APIRouter()

# ==================== 认证 ====================


@router.post("/auth/register", response_model=AuthResponse, dependencies=[Depends(rate_limit("auth", 5, 60))])
async def register(request: RegisterRequest, db: Session = Depends(get_db)):
    return AuthService.register(db, request)


@router.post("/auth/login", response_model=AuthResponse, dependencies=[Depends(rate_limit("auth", 5, 60))])
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    return AuthService.login(db, request)


@router.get("/auth/me", response_model=CurrentUserResponse)
async def me(current_user: User = Depends(get_current_user)):
    return CurrentUserResponse(username=current_user.username, role=current_user.role)


# ==================== 注册子路由 ====================

router.include_router(chat_router)
router.include_router(sessions_router)
router.include_router(documents_router)
router.include_router(resume_router)
router.include_router(jd_router)
