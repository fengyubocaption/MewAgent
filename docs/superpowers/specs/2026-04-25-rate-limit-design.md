# API 限流设计文档

## 概述

为 SuperMew 添加 API 限流功能，防止恶意用户刷接口（暴力登录、刷聊天、批量上传），保护后端资源。

## 目标

- 防滥用为主：阻止恶意请求消耗系统资源
- 用户维度限流：已登录用户按 username，未登录按 IP
- 分组限流：不同端点组设置不同配额

## 技术方案

### 方案选择

FastAPI 依赖注入 + Redis 固定窗口计数。

- 优点：遵循 FastAPI 惯例，粒度精确，复用现有 Redis，无额外依赖
- 实现简单，与现有 `Depends(get_current_user)` 模式一致

### 限流键格式

```
supermew:ratelimit:{group}:{identity}:{window}
```

- `group`：限流组名（auth/chat/upload/read）
- `identity`：用户名（已登录）或 IP（未登录）
- `window`：当前时间戳整除窗口秒数

示例：`supermew:ratelimit:auth:admin:1714032000`

### Redis 实现

固定窗口算法：

```python
key = f"ratelimit:{group}:{identity}:{timestamp // window}"
count = redis.incr(key)
if count == 1:
    redis.expire(key, window)
if count > max_requests:
    raise HTTP429
```

### 身份识别策略

1. 尝试从 `request.state.user` 获取已认证用户（由 `get_current_user` 依赖注入）
2. 若无用户对象，从请求获取客户端 IP：
   - 优先读取 `X-Forwarded-For` 或 `X-Real-IP` header（反向代理场景）
   - 否则用 `request.client.host`

### 端点分组与配额

| 组 | 端点 | 限制 | 说明 |
|---|---|---|---|
| `auth` | `/auth/register`, `/auth/login` | 5次/分钟 | 防暴力破解 |
| `chat` | `/chat/stream` | 20次/分钟 | 防刷对话 |
| `upload` | `/documents/upload` | 3次/分钟 | 上传计算密集 |
| `read` | `/sessions/*`, `/documents` 等 GET | 30次/分钟 | 读取操作较轻 |

### 超限响应

HTTP 429 Too Many Requests

```json
{
  "detail": "请求过于频繁，请稍后再试",
  "retry_after": 45
}
```

Header: `Retry-After: 45`

## 代码结构

### 新增文件

`backend/middleware/rate_limit.py`

```python
from fastapi import Request, HTTPException
from typing import Optional

def get_client_ip(request: Request) -> str:
    """获取客户端真实 IP，支持反向代理。"""
    ...

def get_identity(request: Request) -> str:
    """获取限流标识（username 或 IP）。"""
    ...

def rate_limit(group: str, max_requests: int, window_seconds: int = 60):
    """FastAPI 依赖函数，用于路由限流。"""
    ...
```

### 路由集成

```python
# backend/routes/api.py
from backend.middleware.rate_limit import rate_limit

@router.post("/auth/login", dependencies=[Depends(rate_limit("auth", 5, 60))])
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    ...

@router.post("/chat/stream", dependencies=[Depends(rate_limit("chat", 20, 60))])
async def chat_stream_endpoint(request: ChatRequest, current_user: User = Depends(get_current_user)):
    ...

@router.post("/documents/upload", dependencies=[Depends(rate_limit("upload", 3, 60))])
async def upload_document(file: UploadFile = File(...), _: User = Depends(require_admin)):
    ...
```

## 日志与监控

### 日志记录

限流触发时记录 warning 级别日志：

```python
logger.warning("Rate limit hit: group=%s, identity=%s, count=%d, path=%s",
               group, identity, count, request.url.path)
```

## 配置项（可选扩展）

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `RATE_LIMIT_ENABLED` | `true` | 全局开关 |
| `RATE_LIMIT_AUTH` | `5/60` | 登录/注册限制 |
| `RATE_LIMIT_CHAT` | `20/60` | 聊天限制 |
| `RATE_LIMIT_UPLOAD` | `3/60` | 上传限制 |
| `RATE_LIMIT_READ` | `30/60` | 读取限制 |

格式：`{max_requests}/{window_seconds}`

## 实现步骤

1. 创建 `backend/middleware/` 目录
2. 实现 `rate_limit.py` 核心逻辑
3. 在 `api.py` 路由上添加限流依赖
4. 测试验证各端点限流效果

## 注意事项

- Redis 连接复用现有 `cache._get_client()`
- 限流检查在认证之后，避免认证请求被无差别限流
- 未登录请求用 IP 限流，注意反向代理场景获取真实 IP
- SSE 流式响应的限流在连接建立时检查，不控制流持续时间
