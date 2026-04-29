"""API 限流模块 — 基于 Redis 的固定窗口限流，使用 Lua 脚本保证原子性。"""
import logging
import os
import time
from typing import Callable

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from backend.db.cache import cache

logger = logging.getLogger(__name__)

# 限流配置
# RATE_LIMIT_ENABLED: 设为 "false" 可全局禁用限流（默认启用）
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() != "false"
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-this-secret")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

# Lua 脚本：原子性地执行 INCR + EXPIRE
# KEYS[1]: 限流 key
# ARGV[1]: 过期时间（秒）
# 返回值: 当前计数
_RATE_LIMIT_SCRIPT = """
local current = redis.call('INCR', KEYS[1])
if current == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
end
return current
"""

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


def get_client_ip(request: Request) -> str:
    """获取客户端真实 IP，支持反向代理场景。

    优先级：X-Forwarded-For > X-Real-IP > request.client.host

    注意：此函数信任 X-Forwarded-For 和 X-Real-IP 头。
    在生产环境中，请确保服务部署在受信任的反向代理（如 Nginx）之后，
    否则恶意客户端可能伪造这些头来绕过限流。
    """
    # X-Forwarded-For 可能包含多个 IP，取第一个
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()

    # X-Real-IP 常见于 Nginx
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()

    # 直连场景
    if request.client:
        return request.client.host

    return "unknown"


async def get_identity(request: Request, token: str | None = None) -> str:
    """获取限流标识（username 或 IP）。

    策略：
    1. 如果有有效的 JWT token，解析出 username
    2. 否则使用客户端 IP
    """
    if token:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
            if username:
                return f"user:{username}"
        except JWTError:
            pass  # Token 无效，回退到 IP

    return f"ip:{get_client_ip(request)}"


def rate_limit(group: str, max_requests: int, window_seconds: int = 60) -> Callable:
    """FastAPI 依赖函数，用于路由限流。

    Args:
        group: 限流组名（auth/chat/upload/read）
        max_requests: 窗口内最大请求数
        window_seconds: 窗口秒数，默认 60

    Returns:
        FastAPI 依赖函数

    Usage:
        @router.post("/chat", dependencies=[Depends(rate_limit("chat", 20, 60))])
    """

    async def _rate_limit_dependency(
        request: Request,
        token: str | None = Depends(oauth2_scheme),
    ):
        if not RATE_LIMIT_ENABLED:
            return

        # 获取限流标识
        identity = await get_identity(request, token)

        # 获取 Redis 客户端（使用单例）
        redis_client = cache._get_client()

        # 计算当前窗口
        current_window = int(time.time()) // window_seconds
        key = f"{cache.key_prefix}:ratelimit:{group}:{identity}:{current_window}"

        # 固定窗口计数（使用 Lua 脚本保证原子性）
        try:
            count = redis_client.eval(_RATE_LIMIT_SCRIPT, 1, key, window_seconds)
        except Exception as e:
            logger.error("Redis unavailable, allowing request: %s", e)
            return  # fail-open

        # 检查是否超限
        if count > max_requests:
            # 计算剩余时间
            ttl = redis_client.ttl(key)
            retry_after = max(1, ttl) if ttl > 0 else window_seconds

            logger.warning(
                "Rate limit hit: group=%s, identity=%s, count=%d, path=%s",
                group, identity, count, request.url.path
            )

            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="请求过于频繁，请稍后再试",
                headers={"Retry-After": str(retry_after)},
            )

    return _rate_limit_dependency
