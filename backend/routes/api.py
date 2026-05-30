from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.middleware.rate_limit import rate_limit
from backend.db.database import get_db
from backend.routes.security import authenticate_user, create_access_token, get_current_user, get_password_hash, resolve_role
from backend.db.models import User
from backend.schemas import (
    AuthResponse,
    CurrentUserResponse,
    LoginRequest,
    RegisterRequest,
)

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
    username = (request.username or "").strip()
    password = (request.password or "").strip()
    if not username or not password:
        raise HTTPException(status_code=400, detail="用户名和密码不能为空")

    exists = db.query(User).filter(User.username == username).first()
    if exists:
        raise HTTPException(status_code=409, detail="用户名已存在")

    role = resolve_role(request.role, request.admin_code)
    user = User(username=username, password_hash=get_password_hash(password), role=role)
    db.add(user)
    db.commit()

    token = create_access_token(username=username, role=role)
    return AuthResponse(access_token=token, username=username, role=role)


@router.post("/auth/login", response_model=AuthResponse, dependencies=[Depends(rate_limit("auth", 5, 60))])
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = authenticate_user(db, request.username, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = create_access_token(username=user.username, role=user.role)
    return AuthResponse(access_token=token, username=user.username, role=user.role)


@router.get("/auth/me", response_model=CurrentUserResponse)
async def me(current_user: User = Depends(get_current_user)):
    return CurrentUserResponse(username=current_user.username, role=current_user.role)


# ==================== 注册子路由 ====================

router.include_router(chat_router)
router.include_router(sessions_router)
router.include_router(documents_router)
router.include_router(resume_router)
router.include_router(jd_router)
