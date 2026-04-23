# MewAgent 项目说明

RAG 聊天机器人，基于 FastAPI + Vue 3 (CDN)，集成 LangChain Agent、混合检索、知识图谱。

## 本地部署

### 1) 环境准备
- Python `3.12+`
- 包管理：`uv`（或 `pip`）
- Docker / Docker Compose（用于启动 Milvus、Neo4j 依赖）

### 2) 安装依赖

```bash
# 推荐 uv
uv sync

# 或 pip
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 3) 创建 `.env` 文件

参考 `.env.example`：

```env
# ===== LLM =====
# 必填，LLM API 密钥
ARK_API_KEY=
# 必填，主模型，用于 Agent 推理和回答生成
MODEL=
# 可选，相关性评分模型，默认与 MODEL 相同
GRADE_MODEL=
# 必填，LLM API 地址
BASE_URL=

# ===== Embedding =====
# 向量模型名称，默认 BAAI/bge-m3
EMBEDDING_MODEL=BAAI/bge-m3
# 运行设备：cpu / cuda /mps
EMBEDDING_DEVICE=cpu
# 向量维度，需与 Milvus 集合维度一致，默认 1024
DENSE_EMBEDDING_DIM=1024

# ===== Rerank (可选，不配则跳过精排) =====
RERANK_MODEL=
RERANK_BINDING_HOST=
RERANK_API_KEY=

# ===== Milvus =====
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530
# Milvus 集合名称
MILVUS_COLLECTION=embeddings_collection

# ===== Database / Cache =====
# PostgreSQL 连接串
DATABASE_URL=postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/langchain_app
# Redis 连接串
REDIS_URL=redis://127.0.0.1:6379/0

# ===== Auth =====
# 必填，JWT 签名密钥，建议用 openssl rand -hex 32 生成
JWT_SECRET_KEY=
# 管理员注册邀请码
ADMIN_INVITE_CODE=supermew-admin-2026
JWT_ALGORITHM=HS256
# Token 有效期（分钟）
JWT_EXPIRE_MINUTES=1440
# 密码哈希迭代次数
PASSWORD_PBKDF2_ROUNDS=310000

# ===== Neo4j 知识图谱 =====
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=
# 是否启用图谱功能，false 则跳过图谱构建和检索
GRAPH_ENABLED=true

# ===== BM25 稀疏检索 (可选) =====
# BM25 状态文件路径，存储词表和文档频次统计，默认 data/bm25_state.json
# BM25_STATE_PATH=

# ===== Auto-merging (可选) =====
# 是否启用自动合并（L3→L2→L1）
# AUTO_MERGE_ENABLED=true
# 合并阈值：同一父块下有 N 个子块被召回时触发合并
# AUTO_MERGE_THRESHOLD=2
# 检索的叶子层级，默认 3
# LEAF_RETRIEVE_LEVEL=3

# ===== Tools (可选) =====
# 高德天气 API
AMAP_WEATHER_API=https://restapi.amap.com/v3/weather/weatherInfo
AMAP_API_KEY=

# ===== LangSmith 调试追踪 (可选) =====
# LANGSMITH_TRACING=true
# LANGSMITH_ENDPOINT=https://api.smith.langchain.com
# LANGCHAIN_API_KEY=
# LANGCHAIN_PROJECT=supermew

```

### 4) Docker 启动基础设施

```bash
docker compose up -d
```

端口说明：
- PostgreSQL：`5432`
- Redis：`6379`
- Milvus：`19530`
- Neo4j：`7687` (Bolt)、`7474` (HTTP)
- Attu (Milvus GUI)：`8080`

### 5) 启动应用

```bash
uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

访问：
- 前端：`http://127.0.0.1:8000/`
- API 文档：`http://127.0.0.1:8000/docs`

## 项目架构

```
backend/
├── app.py              — FastAPI 入口
├── routes/             — API 路由 (api.py, auth.py, schemas.py)
├── db/                 — 数据库模型、Redis 缓存
├── agent/              — LangGraph Agent、工具、记忆管理
├── rag/                — RAG Pipeline、分块、向量检索
├── milvus/             — Milvus 客户端、向量写入、Embedding
└── graph/              — Neo4j 客户端、图谱构建、图谱检索

frontend/               — Vue 3 SPA (index.html + script.js + style.css)
data/                   — bm25_state.json, 上传文档
```

**核心能力：**
- **混合检索**：稠密向量 (BAAI/bge-m3) + BM25 稀疏向量 → Milvus RRF 融合 → Jina Rerank 精排
- **Graph RAG**：Neo4j 知识图谱存储实体关系，LLM 抽取，向量+图谱并行融合检索，多跳推理
- **三级分块 + Auto-merging**：L1/L2 父块存 PostgreSQL，L3 叶子块入 Milvus，检索时自动合并
- **流式输出**：SSE + asyncio.Queue，跨线程 RAG 步骤实时推送
- **LangMem 长期记忆**：用户画像、会话摘要、程序经验三类记忆，跨会话保留
- **RBAC 鉴权**：JWT Bearer Token，admin/user 角色权限隔离

## 目录详解

### backend/routes/
- `api.py` — 聊天、会话管理、文档管理接口
- `auth.py` — 注册登录、JWT 鉴权、密码哈希
- `schemas.py` — Pydantic 请求/响应模型

### backend/db/
- `database.py` — SQLAlchemy 引擎、会话工厂
- `models.py` — ORM 模型（用户、会话、消息、父文档）
- `cache.py` — Redis JSON 缓存封装

### backend/agent/
- `agent.py` — LangGraph Agent、会话存储
- `tools.py` — 天气查询、知识库检索工具
- `memory_manager.py` — LangMem 记忆管理
- `memory_tools.py` — 记忆存储/检索工具

### backend/rag/
- `rag_pipeline.py` — RAG 工作流（检索 → 评分 → 重写 → 再检索）
- `rag_utils.py` — 检索、Rerank、Auto-merging、HyDE 等辅助函数
- `document_loader.py` — PDF/Word/Excel 加载与三级分片
- `parent_chunk_store.py` — 父级分块仓储（PostgreSQL + Redis）

### backend/milvus/
- `milvus_client.py` — Milvus 集合定义、混合检索、分页查询
- `milvus_writer.py` — 向量写入（稠密+稀疏）
- `embedding.py` — HuggingFace 稠密向量 + BM25 稀疏向量

### backend/graph/
- `neo4j_client.py` — Neo4j 连接管理
- `graph_builder.py` — 实体抽取、关系构建
- `graph_retriever.py` — 图谱检索、多跳查询

## 核心流程

### 聊天流程
1. 用户输入 → `POST /chat/stream` (SSE)
2. Agent 判断是否调用工具（知识库检索/天气查询）
3. 若命中知识库 → RAG Pipeline 执行检索 → 实时推送步骤到前端
4. Agent 流式生成回答（打字机效果）
5. 消息持久化到 PostgreSQL，Redis 缓存热点会话

### RAG Pipeline
1. **检索**：Milvus Hybrid (Dense + Sparse + RRF) → Jina Rerank 精排 → Auto-merging 合并父块
2. **评分**：LLM 判断文档相关性，不相关则触发查询重写
3. **重写**：Step-Back / HyDE 策略生成扩展查询
4. **再检索**：对重写查询再次检索，返回上下文

### 文档入库
1. 上传 → 三级滑动窗口分块
2. L1/L2 父块写入 PostgreSQL，L3 叶子块写入 Milvus
3. 同步构建 Neo4j 知识图谱（实体抽取、关系建立）
4. BM25 统计增量更新到 `bm25_state.json`

## API 速览

| 路由 | 说明 |
|------|------|
| `POST /auth/register` | 注册（支持管理员邀请码） |
| `POST /auth/login` | 登录，返回 Bearer Token |
| `GET /auth/me` | 获取当前用户信息 |
| `POST /chat/stream` | 流式聊天 (SSE) |
| `GET /sessions` | 列出当前用户会话 |
| `DELETE /sessions/{id}` | 删除会话 |
| `GET /documents` | 列出文档 (admin) |
| `POST /documents/upload` | 上传文档 (admin) |
| `DELETE /documents/{filename}` | 删除文档 (admin) |

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `ARK_API_KEY` | LLM API 密钥 | - |
| `MODEL` | 主模型，用于 Agent 推理 | - |
| `GRADE_MODEL` | 相关性评分模型 | 与 MODEL 相同 |
| `BASE_URL` | LLM API 地址 | - |
| `EMBEDDING_MODEL` | 向量模型名称 | `BAAI/bge-m3` |
| `EMBEDDING_DEVICE` | 运行设备 | `cpu` |
| `DENSE_EMBEDDING_DIM` | 向量维度 | `1024` |
| `RERANK_MODEL`, `RERANK_BINDING_HOST`, `RERANK_API_KEY` | Rerank 精排配置 | - |
| `MILVUS_HOST`, `MILVUS_PORT`, `MILVUS_COLLECTION` | Milvus 向量库 | `127.0.0.1:19530` |
| `DATABASE_URL` | PostgreSQL 连接串 | - |
| `REDIS_URL` | Redis 连接串 | - |
| `JWT_SECRET_KEY` | JWT 签名密钥 | - |
| `ADMIN_INVITE_CODE` | 管理员注册邀请码 | - |
| `JWT_EXPIRE_MINUTES` | Token 有效期（分钟） | `1440` |
| `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` | Neo4j 图谱连接 | - |
| `GRAPH_ENABLED` | 是否启用图谱功能 | `true` |
| `BM25_STATE_PATH` | BM25 统计文件路径 | `data/bm25_state.json` |
| `AUTO_MERGE_ENABLED` | 是否启用自动合并 | `true` |
| `AUTO_MERGE_THRESHOLD` | 合并阈值（子块数） | `2` |
| `LEAF_RETRIEVE_LEVEL` | 检索叶子层级 | `3` |
| `AMAP_API_KEY` | 高德天气 API 密钥 | - |

## SSE 事件格式

`data: {JSON}\n\n`，类型包括：
- `content` — 文本 token
- `rag_step` — 检索步骤
- `trace` — RAG 追踪信息
- `error` — 错误
- `[DONE]` — 流结束

## 技术栈

- **后端**：FastAPI, LangChain/LangGraph, SQLAlchemy, PostgreSQL, Redis
- **向量**：Milvus (HNSW + SPARSE_INVERTED_INDEX), RRF 融合, Jina Rerank
- **图谱**：Neo4j, 实体抽取, 多跳查询
- **Embedding**：BAAI/bge-m3 + 自定义 BM25
- **前端**：Vue 3 (CDN), marked, highlight.js
- **记忆**：LangMem
