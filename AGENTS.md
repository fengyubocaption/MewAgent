# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

SuperMew is an interview preparation assistant built on RAG (Retrieval-Augmented Generation) with LangChain Agent, FastAPI + Vue 3 (CDN). Core features: resume parsing, JD analysis, resume-JD matching, mock interview (question generation + answer evaluation), hybrid retrieval (dense + BM25 sparse via Milvus), Graph RAG (Neo4j), streaming SSE, RBAC auth (JWT), session memory (PostgreSQL + Redis), three-level chunking with auto-merging, and typed long-term memory.

## Commands

```bash
# Install dependencies
uv sync

# Run the server
uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload

# Start infrastructure (PostgreSQL, Redis, Milvus)
docker compose up -d
```

## Architecture

```
backend/
├── app.py              — FastAPI entry point
├── core/               — Cross-cutting: JWT auth, RBAC, rate limiting, progress tracking
├── routes/             — API endpoints (thin HTTP layer, delegates to services)
├── services/           — Business logic (auth, document, resume, JD, session)
├── schemas/            — Pydantic request/response models
├── db/                 — Database (SQLAlchemy), models, Redis cache
├── agent/              — LangGraph Agent + tools (weather, KB search, memory, interview)
├── rag/                — RAG pipeline (retrieve → grade → rewrite → re-retrieve), MinerU parser
├── milvus/             — Vector storage, embedding, document loading
└── graph/              — Neo4j 客户端、图谱构建、图谱检索

frontend/               — Vue 3 SPA (index.html + script.js + style.css)
data/                   — bm25_state.json, uploaded documents
evals/                  — LangSmith and other evaluation scripts
experiments/            — notebooks, learning examples, and scratch experiments
tests/                  — automated tests
```

Key components:
- **Hybrid Search**: Dense (BAAI/bge-m3) + BM25 sparse via Milvus RRF fusion
- **Graph RAG**: Neo4j 知识图谱 + 向量检索融合，实体关系推理，多跳查询
- **Streaming**: SSE with asyncio.Queue, cross-thread RAG step emission
- **Three-level Chunking**: L1/L2 parent chunks in PostgreSQL + Redis, L3 leaves in Milvus
- **Auto-merging**: Retrieval merges L3→L2→L1 based on similarity threshold
- **MinerU Integration** (optional): High-quality document parsing with OCR, table structure preservation, formula recognition via MinerU LocalAPIServer subprocess; falls back to LangChain loaders on failure

## Key Patterns

- **Streaming**: Unified `asyncio.Queue` with `_agent_worker` background task; cross-thread RAG step emission via `call_soon_threadsafe`
- **Hybrid Search**: Dense (BAAI/bge-m3) + BM25 sparse → RRF fusion (k=60) → Jina Rerank; falls back to dense-only on failure
- **Graph RAG**: Neo4j 知识图谱存储实体关系；LLM 抽取实体；向量+图谱并行融合检索；多跳推理查询
- **BM25 State**: Persisted to `data/bm25_state.json`, `increment_add`/`increment_remove` on write/delete
- **Auth**: JWT Bearer tokens, PBKDF2-SHA256 passwords, RBAC (admin/user), user data isolation
- **Service Layer**: Business logic in `services/`, routes are thin HTTP delegates; DB sessions via `Depends(get_db)`
- **Long-term Memory**: Four typed memories (user/feedback/project/reference), stored in PostgreSQL, Agent can store/retrieve via tools
- **Interview Tools**: Resume parsing, JD analysis, resume-JD matching, mock interview (question generation + answer evaluation), all via LLM-powered structured extraction
- **MinerU Parser**: Optional high-quality document parsing via MinerU subprocess; enabled via `MINERU_ENABLED=true`; supports PDF/DOCX/PPTX/images with OCR, table structure, formula recognition; falls back to LangChain loaders on failure

## Environment Variables

See README.md for full list with templates. Key ones:

| Variable | Purpose |
|---|---|
| `ARK_API_KEY`, `MODEL`, `BASE_URL` | LLM config |
| `EMBEDDING_MODEL`, `EMBEDDING_DEVICE`, `DENSE_EMBEDDING_DIM` | Local embedding (default BAAI/bge-m3, 1024d) |
| `RERANK_MODEL`, `RERANK_BINDING_HOST`, `RERANK_API_KEY` | Jina Rerank (optional) |
| `MILVUS_HOST`, `MILVUS_PORT`, `MILVUS_COLLECTION` | Vector DB |
| `DATABASE_URL`, `REDIS_URL` | PostgreSQL + Redis |
| `JWT_SECRET_KEY`, `ADMIN_INVITE_CODE`, `JWT_ALGORITHM`, `JWT_EXPIRE_MINUTES` | Auth |
| `PASSWORD_PBKDF2_ROUNDS` | Password hashing |
| `BM25_STATE_PATH` | BM25 persistence path (default `data/bm25_state.json`) |
| `AUTO_MERGE_ENABLED`, `AUTO_MERGE_THRESHOLD`, `LEAF_RETRIEVE_LEVEL` | Auto-merging config |
| `AMAP_WEATHER_API`, `AMAP_API_KEY` | Weather tool |
| `MINERU_ENABLED`, `MINERU_BACKEND`, `MINERU_PARSE_METHOD`, `MINERU_LANG`, `MINERU_MODEL_SOURCE` | MinerU document parsing (optional, default disabled) |

## API Overview

- Auth: `POST /auth/register`, `POST /auth/login`, `GET /auth/me`
- Chat: `POST /chat`, `POST /chat/stream` (SSE)
- Sessions: `GET /sessions`, `GET /sessions/{id}`, `DELETE /sessions/{id}`
- Resume: `POST /resume/upload`, `GET /resume`, `GET /resume/{id}`, `DELETE /resume/{id}`
- JD: `POST /jd`, `GET /jd`, `GET /jd/{id}`, `DELETE /jd/{id}`
- Documents (admin): `GET /documents`, `POST /documents/upload`, `DELETE /documents/{filename}`

## SSE Events

`data: {JSON}\n\n` with types: `content`, `rag_step`, `trace`, `error`, `[DONE]`

## Tech Stack

- **Backend**: FastAPI, LangChain/LangGraph, SQLAlchemy, PostgreSQL, Redis
- **Vector**: Milvus (HNSW dense + SPARSE_INVERTED_INDEX), RRF fusion, Jina Rerank
- **Embedding**: BAAI/bge-m3 + custom BM25
- **Frontend**: Vue 3 (CDN), marked, highlight.js
- **Memory**: PostgreSQL-based typed memory (user/feedback/project/reference)
