# API 限流实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 SuperMew 添加基于 Redis 的 API 限流功能，防止接口滥用。

**Architecture:** FastAPI 依赖注入模式 + Redis 固定窗口计数。限流函数从 JWT token 解析用户名（已登录）或使用客户端 IP（未登录），对不同端点组应用不同配额。

**Tech Stack:** FastAPI, Redis, python-jose (JWT)

---

## File Structure

| 文件 | 操作 | 职责 |
|---|---|---|
| `pyproject.toml` | 修改 | 添加 pytest, pytest-asyncio, httpx 测试依赖 |
| `backend/middleware/__init__.py` | 创建 | 模块初始化 |
| `backend/middleware/rate_limit.py` | 创建 | 限流核心逻辑 |
| `backend/routes/api.py` | 修改 | 为各端点添加限流依赖 |
| `tests/__init__.py` | 创建 | 测试模块初始化 |
| `tests/middleware/__init__.py` | 创建 | 测试模块初始化 |
| `tests/middleware/test_rate_limit.py` | 创建 | 限流单元测试 |

---

### Task 1: 添加测试依赖

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: 添加 pytest 相关依赖到 pyproject.toml**

在 `[project]` 的 `dependencies` 数组末尾添加：

```toml
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "httpx>=0.27.0",
```

- [ ] **Step 2: 安装依赖**

Run: `cd D:/code/fyb/SuperMew && uv sync`

Expected: 依赖安装成功

- [ ] **Step 3: 提交**

```bash
git add pyproject.toml
git commit -m "chore: 添加 pytest 测试依赖"
```

---

### Task 2: 创建 middleware 目录结构

**Files:**
- Create: `backend/middleware/__init__.py`

- [ ] **Step 1: 创建 middleware 目录和 __init__.py**

创建目录 `backend/middleware/` 并添加空的 `__init__.py` 文件：

```python
# backend/middleware/__init__.py
"""中间件模块。"""
```

- [ ] **Step 2: 提交**

```bash
git add backend/middleware/__init__.py
git commit -m "chore: 创建 middleware 模块目录"
```

---

### Task 3: 实现限流核心逻辑

**Files:**
- Create: `backend/middleware/rate_limit.py`

- [ ] **Step 1: 实现 get_client_ip 函数**

创建 `backend/middleware/rate_limit.py`，首先实现 IP 获取函数：

```python
"""API 限流模块 — 基于 Redis 的固定窗口限流。"""
import logging
import os
import time
from typing import Callable

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from backend.db.cache import RedisCache

logger = logging.getLogger(__name__)

# 限流配置
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() != "false"
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-this-secret")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


def get_client_ip(request: Request) -> str:
    """获取客户端真实 IP，支持反向代理场景。
    
    优先级：X-Forwarded-For > X-Real-IP > request.client.host
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
```

- [ ] **Step 2: 实现 get_identity 函数**

在 `rate_limit.py` 中继续添加身份识别函数：

```python
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
```

- [ ] **Step 3: 实现 rate_limit 依赖函数**

在 `rate_limit.py` 中继续添加核心限流函数：

```python
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
        
        # 获取 Redis 客户端
        cache = RedisCache()
        redis_client = cache._get_client()
        
        # 计算当前窗口
        current_window = int(time.time()) // window_seconds
        key = f"{cache.key_prefix}:ratelimit:{group}:{identity}:{current_window}"
        
        # 固定窗口计数
        count = redis_client.incr(key)
        if count == 1:
            redis_client.expire(key, window_seconds)
        
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
```

- [ ] **Step 4: 提交**

```bash
git add backend/middleware/rate_limit.py
git commit -m "feat(middleware): 实现 Redis 固定窗口限流"
```

---

### Task 4: 创建测试目录结构

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/middleware/__init__.py`

- [ ] **Step 1: 创建测试目录**

```python
# tests/__init__.py
"""SuperMew 测试模块。"""
```

```python
# tests/middleware/__init__.py
"""中间件测试模块。"""
```

- [ ] **Step 2: 提交**

```bash
git add tests/__init__.py tests/middleware/__init__.py
git commit -m "chore: 创建测试目录结构"
```

---

### Task 5: 编写限流单元测试

**Files:**
- Create: `tests/middleware/test_rate_limit.py`

- [ ] **Step 1: 编写 get_client_ip 测试**

创建 `tests/middleware/test_rate_limit.py`：

```python
"""限流模块单元测试。"""
import pytest
from unittest.mock import MagicMock
from backend.middleware.rate_limit import get_client_ip


class TestGetClientIp:
    """测试 get_client_ip 函数。"""
    
    def test_x_forwarded_for_first_ip(self):
        """X-Forwarded-For 存在时返回第一个 IP。"""
        request = MagicMock()
        request.headers = {"X-Forwarded-For": "192.168.1.1, 10.0.0.1"}
        request.client = MagicMock(host="127.0.0.1")
        
        result = get_client_ip(request)
        assert result == "192.168.1.1"
    
    def test_x_real_ip(self):
        """X-Real-IP 存在时返回该 IP。"""
        request = MagicMock()
        request.headers = {"X-Real-IP": "203.0.113.1"}
        request.client = MagicMock(host="127.0.0.1")
        
        result = get_client_ip(request)
        assert result == "203.0.113.1"
    
    def test_fallback_to_client_host(self):
        """没有代理头时回退到 client.host。"""
        request = MagicMock()
        request.headers = {}
        request.client = MagicMock(host="127.0.0.1")
        
        result = get_client_ip(request)
        assert result == "127.0.0.1"
    
    def test_no_client_returns_unknown(self):
        """没有 client 对象时返回 unknown。"""
        request = MagicMock()
        request.headers = {}
        request.client = None
        
        result = get_client_ip(request)
        assert result == "unknown"
```

- [ ] **Step 2: 运行测试验证**

Run: `cd D:/code/fyb/SuperMew && uv run pytest tests/middleware/test_rate_limit.py -v`

Expected: 4 tests passed

- [ ] **Step 3: 提交**

```bash
git add tests/middleware/test_rate_limit.py
git commit -m "test(middleware): 添加 get_client_ip 单元测试"
```

---

### Task 6: 在 API 路由上应用限流

**Files:**
- Modify: `backend/routes/api.py`

- [ ] **Step 1: 导入 rate_limit 依赖**

在 `backend/routes/api.py` 文件顶部的导入区域添加：

```python
from backend.middleware.rate_limit import rate_limit
```

在第 9 行之后添加此导入。

- [ ] **Step 2: 为 auth 端点添加限流**

修改 `register` 和 `login` 路由，添加 `dependencies` 参数：

```python
@router.post("/auth/register", response_model=AuthResponse, dependencies=[Depends(rate_limit("auth", 5, 60))])
async def register(request: RegisterRequest, db: Session = Depends(get_db)):
    ...

@router.post("/auth/login", response_model=AuthResponse, dependencies=[Depends(rate_limit("auth", 5, 60))])
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    ...
```

- [ ] **Step 3: 为 chat 端点添加限流**

修改 `chat_stream_endpoint` 路由：

```python
@router.post("/chat/stream", dependencies=[Depends(rate_limit("chat", 20, 60))])
async def chat_stream_endpoint(request: ChatRequest, current_user: User = Depends(get_current_user)):
    ...
```

- [ ] **Step 4: 为 upload 端点添加限流**

修改 `upload_document` 路由：

```python
@router.post("/documents/upload", response_model=DocumentUploadResponse, dependencies=[Depends(rate_limit("upload", 3, 60))])
async def upload_document(file: UploadFile = File(...), _: User = Depends(require_admin)):
    ...
```

- [ ] **Step 5: 为 read 端点添加限流**

修改 GET 端点：

```python
@router.get("/sessions/{session_id}", response_model=SessionMessagesResponse, dependencies=[Depends(rate_limit("read", 30, 60))])
async def get_session_messages(session_id: str, current_user: User = Depends(get_current_user)):
    ...

@router.get("/sessions", response_model=SessionListResponse, dependencies=[Depends(rate_limit("read", 30, 60))])
async def list_sessions(current_user: User = Depends(get_current_user)):
    ...

@router.get("/documents", response_model=DocumentListResponse, dependencies=[Depends(rate_limit("read", 30, 60))])
async def list_documents(_: User = Depends(require_admin)):
    ...
```

- [ ] **Step 6: 提交**

```bash
git add backend/routes/api.py
git commit -m "feat(api): 为各端点添加限流保护"
```

---

### Task 7: 验证限流功能

**Files:**
- 无文件修改，运行验证

- [ ] **Step 1: 启动服务**

Run: `cd D:/code/fyb/SuperMew && uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000`

在后台运行，或使用新终端。

- [ ] **Step 2: 手动测试限流**

使用 curl 或 httpie 快速发送多次请求：

```bash
# 测试 auth 限流（5次/分钟）
for i in {1..7}; do
  curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:8000/auth/login \
    -H "Content-Type: application/json" \
    -d '{"username":"test","password":"test"}'
done
```

Expected: 前 5 次返回 401（用户名密码错误），第 6-7 次返回 429（限流）

- [ ] **Step 3: 检查日志**

查看服务日志，确认限流触发时有 warning 日志输出。

---

## Self-Review Checklist

- [x] **Spec coverage:** 设计文档中所有要求都已覆盖
  - Redis 固定窗口实现 ✓
  - 用户/IP 身份识别 ✓
  - 端点分组限流 ✓
  - 429 响应格式 ✓
  - 日志记录 ✓
  - 环境变量配置 ✓

- [x] **Placeholder scan:** 无 TBD/TODO/实现细节缺失

- [x] **Type consistency:** 函数签名在 Task 3 定义后，Task 6 使用一致
