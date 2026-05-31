"""认证服务 — 注册、登录业务逻辑。"""

from fastapi import HTTPException
from sqlalchemy.orm import Session

from backend.core.security import (
    create_access_token,
    get_password_hash,
    resolve_role,
    verify_password,
)
from backend.db.models import User
from backend.schemas import AuthResponse, LoginRequest, RegisterRequest


class AuthService:
    """用户认证业务逻辑。"""

    @staticmethod
    def register(db: Session, request: RegisterRequest) -> AuthResponse:
        """用户注册。"""
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

    @staticmethod
    def login(db: Session, request: LoginRequest) -> AuthResponse:
        """用户登录。"""
        user = db.query(User).filter(User.username == request.username).first()
        if not user or not verify_password(request.password, user.password_hash):
            raise HTTPException(status_code=401, detail="用户名或密码错误")

        token = create_access_token(username=user.username, role=user.role)
        return AuthResponse(access_token=token, username=user.username, role=user.role)
