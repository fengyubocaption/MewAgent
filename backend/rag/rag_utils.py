import logging
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional
import os
import json
import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

from backend.milvus.milvus_client import MilvusManager
from backend.milvus.embedding import embedding_service as _embedding_service
from backend.rag.parent_chunk_store import ParentChunkStore
from langchain.chat_models import init_chat_model

load_dotenv()

ARK_API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")
RERANK_MODEL = os.getenv("RERANK_MODEL")
RERANK_BINDING_HOST = os.getenv("RERANK_BINDING_HOST")
RERANK_API_KEY = os.getenv("RERANK_API_KEY")
AUTO_MERGE_ENABLED = os.getenv("AUTO_MERGE_ENABLED", "true").lower() != "false"
AUTO_MERGE_THRESHOLD = int(os.getenv("AUTO_MERGE_THRESHOLD", "2"))
LEAF_RETRIEVE_LEVEL = int(os.getenv("LEAF_RETRIEVE_LEVEL", "3"))

# 全局初始化检索依赖（与 api 共用 embedding_service，保证 BM25 状态一致）
_milvus_manager = MilvusManager()
_parent_chunk_store = ParentChunkStore()

_stepback_model = None


def _get_rerank_endpoint() -> str:
    if not RERANK_BINDING_HOST:
        return ""
    host = RERANK_BINDING_HOST.strip().rstrip("/")
    # 阿里云 DashScope 原生 API
    if "dashscope.aliyuncs.com" in host:
        return "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    # 支持多种 rerank API 路径格式
    if host.endswith("/rerank") or host.endswith("/v1/rerank"):
        return host
    # vLLM / OpenAI 兼容格式
    return f"{host}/v1/rerank"


def _merge_to_parent_level(docs: List[dict], threshold: int = 2) -> Tuple[List[dict], int]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    for doc in docs:
        parent_id = (doc.get("parent_chunk_id") or "").strip()
        if parent_id:
            groups[parent_id].append(doc)

    merge_parent_ids = [parent_id for parent_id, children in groups.items() if len(children) >= threshold]
    if not merge_parent_ids:
        return docs, 0

    parent_docs = _parent_chunk_store.get_documents_by_ids(merge_parent_ids)
    parent_map = {item.get("chunk_id", ""): item for item in parent_docs if item.get("chunk_id")}

    merged_docs: List[dict] = []
    merged_count = 0
    for doc in docs:
        parent_id = (doc.get("parent_chunk_id") or "").strip()
        if not parent_id or parent_id not in parent_map:
            merged_docs.append(doc)
            continue
        parent_doc = dict(parent_map[parent_id])
        score = doc.get("score")
        if score is not None:
            parent_doc["score"] = max(float(parent_doc.get("score", score)), float(score))
        parent_doc["merged_from_children"] = True
        parent_doc["merged_child_count"] = len(groups[parent_id])
        merged_docs.append(parent_doc)
        merged_count += 1

    deduped: List[dict] = []
    seen = set()
    for item in merged_docs:
        key = item.get("chunk_id") or (item.get("filename"), item.get("page_number"), item.get("text"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return deduped, merged_count


def _auto_merge_documents(docs: List[dict], top_k: int) -> Tuple[List[dict], Dict[str, Any]]:
    if not AUTO_MERGE_ENABLED or not docs:
        return docs[:top_k], {
            "auto_merge_enabled": AUTO_MERGE_ENABLED,
            "auto_merge_applied": False,
            "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
            "auto_merge_replaced_chunks": 0,
            "auto_merge_steps": 0,
        }

    # 两段自动合并：L3->L2，再 L2->L1。
    merged_docs, merged_count_l3_l2 = _merge_to_parent_level(docs, threshold=AUTO_MERGE_THRESHOLD)
    merged_docs, merged_count_l2_l1 = _merge_to_parent_level(merged_docs, threshold=AUTO_MERGE_THRESHOLD)

    merged_docs.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    merged_docs = merged_docs[:top_k]

    replaced_count = merged_count_l3_l2 + merged_count_l2_l1
    return merged_docs, {
        "auto_merge_enabled": AUTO_MERGE_ENABLED,
        "auto_merge_applied": replaced_count > 0,
        "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
        "auto_merge_replaced_chunks": replaced_count,
        "auto_merge_steps": int(merged_count_l3_l2 > 0) + int(merged_count_l2_l1 > 0),
    }


def _try_rerank_request(url: str, payload: dict, headers: dict, docs_with_rank: List[dict], top_k: int, meta: Dict[str, Any], is_dashscope: bool = False) -> Optional[Tuple[List[dict], Dict[str, Any]]]:
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        if response.status_code >= 400:
            meta["rerank_error"] = f"HTTP {response.status_code} (URL: {url}): {response.text}"
            return None

        data = response.json()
        # 阿里云 DashScope 返回格式是 {"output": {"results": [...]}}
        items = data.get("output", {}).get("results", []) if is_dashscope else data.get("results", [])
        reranked = []
        for item in items:
            idx = item.get("index")
            if isinstance(idx, int) and 0 <= idx < len(docs_with_rank):
                doc = dict(docs_with_rank[idx])
                score = item.get("relevance_score")
                if score is not None:
                    doc["rerank_score"] = score
                reranked.append(doc)

        if reranked:
            meta["rerank_endpoint"] = url
            return reranked[:top_k], meta

        meta["rerank_error"] = "empty_rerank_results"
        return None
    except (requests.RequestException, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        meta["rerank_error"] = str(e)
        return None


def _rerank_documents(query: str, docs: List[dict], top_k: int) -> Tuple[List[dict], Dict[str, Any]]:
    docs_with_rank = [{**doc, "rrf_rank": i} for i, doc in enumerate(docs, 1)]
    meta: Dict[str, Any] = {
        "rerank_enabled": bool(RERANK_MODEL and RERANK_API_KEY and RERANK_BINDING_HOST),
        "rerank_applied": False,
        "rerank_model": RERANK_MODEL,
        "rerank_endpoint": _get_rerank_endpoint(),
        "rerank_error": None,
        "candidate_count": len(docs_with_rank),
    }
    if not docs_with_rank or not meta["rerank_enabled"]:
        return docs_with_rank[:top_k], meta

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RERANK_API_KEY}",
    }

    meta["rerank_applied"] = True
    host = RERANK_BINDING_HOST.strip().rstrip("/")

    # 检测是否是阿里云 DashScope
    is_dashscope = "dashscope.aliyuncs.com" in host
    if is_dashscope:
        # 阿里云 DashScope 原生 API 格式
        url = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
        payload = {
            "model": RERANK_MODEL,
            "input": {
                "query": query,
                "documents": [doc.get("text", "") for doc in docs_with_rank],
            },
        }
        result = _try_rerank_request(url, payload, headers, docs_with_rank, top_k, meta, is_dashscope=True)
        if result is not None:
            return result
    else:
        # OpenAI/Jina 兼容格式
        payload = {
            "model": RERANK_MODEL,
            "query": query,
            "documents": [doc.get("text", "") for doc in docs_with_rank],
            "top_n": min(top_k, len(docs_with_rank)),
            "return_documents": False,
        }

        # 尝试多个路径格式
        endpoints_to_try = [
            f"{host}/v1/rerank",
            f"{host}/rerank",
        ]
        # 如果 host 已经包含路径，就只试它一个
        if host.endswith("/rerank") or host.endswith("/v1/rerank"):
            endpoints_to_try = [host]

        for url in endpoints_to_try:
            result = _try_rerank_request(url, payload, headers, docs_with_rank, top_k, meta)
            if result is not None:
                return result

    return docs_with_rank[:top_k], meta


def _get_stepback_model():
    global _stepback_model
    if not ARK_API_KEY or not MODEL:
        return None
    if _stepback_model is None:
        _stepback_model = init_chat_model(
            model=MODEL,
            model_provider="openai",
            api_key=ARK_API_KEY,
            base_url=BASE_URL,
            temperature=0.2,
        )
    return _stepback_model


def _generate_step_back_question(query: str) -> str:
    model = _get_stepback_model()
    if not model:
        return ""
    prompt = (
        "请将用户的具体问题抽象成更高层次、更概括的’退步问题’，"
        "用于探寻背后的通用原理或核心概念。只输出退步问题一句话，不要解释。\n"
        f"用户问题：{query}"
    )
    try:
        return (model.invoke(prompt).content or "").strip()
    except Exception as e:
        logger.warning(f"生成退步问题失败: {e}")
        return ""


def _answer_step_back_question(step_back_question: str) -> str:
    model = _get_stepback_model()
    if not model or not step_back_question:
        return ""
    prompt = (
        "请简要回答以下退步问题，提供通用原理/背景知识，"
        "控制在120字以内。只输出答案，不要列出推理过程。\n"
        f"退步问题：{step_back_question}"
    )
    try:
        return (model.invoke(prompt).content or "").strip()
    except Exception as e:
        logger.warning(f"回答退步问题失败: {e}")
        return ""


def generate_hypothetical_document(query: str) -> str:
    model = _get_stepback_model()
    if not model:
        return ""
    prompt = (
        "请基于用户问题生成一段’假设性文档’，内容应像真实资料片段，"
        "用于帮助检索相关信息。文档可以包含合理推测，但需与问题语义相关。"
        "只输出文档正文，不要标题或解释。\n"
        f"用户问题：{query}"
    )
    try:
        return (model.invoke(prompt).content or "").strip()
    except Exception as e:
        logger.warning(f"生成假设性文档失败: {e}")
        return ""


def step_back_expand(query: str) -> dict:
    step_back_question = _generate_step_back_question(query)
    step_back_answer = _answer_step_back_question(step_back_question)
    if step_back_question or step_back_answer:
        expanded_query = (
            f"{query}\n\n"
            f"退步问题：{step_back_question}\n"
            f"退步问题答案：{step_back_answer}"
        )
    else:
        expanded_query = query
    return {
        "step_back_question": step_back_question,
        "step_back_answer": step_back_answer,
        "expanded_query": expanded_query,
    }


def retrieve_documents(query: str, top_k: int = 5) -> Dict[str, Any]:
    candidate_k = top_k * 3
    filter_expr = f"chunk_level == {LEAF_RETRIEVE_LEVEL}"
    try:
        dense_embeddings = _embedding_service.get_embeddings([query])
        dense_embedding = dense_embeddings[0]
        sparse_embedding = _embedding_service.get_sparse_embedding(query)

        retrieved = _milvus_manager.hybrid_retrieve(
            dense_embedding=dense_embedding,
            sparse_embedding=sparse_embedding,
            top_k=candidate_k,
            filter_expr=filter_expr,
        )
        reranked, rerank_meta = _rerank_documents(query=query, docs=retrieved, top_k=top_k)
        merged_docs, merge_meta = _auto_merge_documents(docs=reranked, top_k=top_k)
        rerank_meta["retrieval_mode"] = "hybrid"
        rerank_meta["candidate_k"] = candidate_k
        rerank_meta["leaf_retrieve_level"] = LEAF_RETRIEVE_LEVEL
        rerank_meta.update(merge_meta)
        return {"docs": merged_docs, "meta": rerank_meta}
    except Exception as e:
        logger.warning(f"混合检索失败，降级为纯向量检索: {e}")
        try:
            dense_embeddings = _embedding_service.get_embeddings([query])
            dense_embedding = dense_embeddings[0]
            retrieved = _milvus_manager.dense_retrieve(
                dense_embedding=dense_embedding,
                top_k=candidate_k,
                filter_expr=filter_expr,
            )
            reranked, rerank_meta = _rerank_documents(query=query, docs=retrieved, top_k=top_k)
            merged_docs, merge_meta = _auto_merge_documents(docs=reranked, top_k=top_k)
            rerank_meta["retrieval_mode"] = "dense_fallback"
            rerank_meta["candidate_k"] = candidate_k
            rerank_meta["leaf_retrieve_level"] = LEAF_RETRIEVE_LEVEL
            rerank_meta.update(merge_meta)
            return {"docs": merged_docs, "meta": rerank_meta}
        except Exception as e2:
            logger.error(f"纯向量检索也失败: {e2}")
            return {
                "docs": [],
                "meta": {
                    "rerank_enabled": bool(RERANK_MODEL and RERANK_API_KEY and RERANK_BINDING_HOST),
                    "rerank_applied": False,
                    "rerank_model": RERANK_MODEL,
                    "rerank_endpoint": _get_rerank_endpoint(),
                    "rerank_error": "retrieve_failed",
                    "retrieval_mode": "failed",
                    "candidate_k": candidate_k,
                    "leaf_retrieve_level": LEAF_RETRIEVE_LEVEL,
                    "auto_merge_enabled": AUTO_MERGE_ENABLED,
                    "auto_merge_applied": False,
                    "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
                    "auto_merge_replaced_chunks": 0,
                    "auto_merge_steps": 0,
                    "candidate_count": 0,
                },
            }


# ===== 向量 + 图谱融合检索 =====

GRAPH_ENABLED = os.getenv("GRAPH_ENABLED", "true").lower() != "false"


def retrieve_documents_with_graph(query: str, top_k: int = 5) -> Dict[str, Any]:
    """向量 + 图谱 并行融合检索

    Returns:
        {
            "docs": [...],
            "meta": {
                ...原有 meta 信息...,
                "graph_enabled": bool,
                "graph_entities": list[str],
                "graph_chunk_count": int,
            }
        }
    """
    # 1. 向量检索（原有逻辑，已包含 auto_merge）
    # 多取一些向量结果用于与图谱结果融合，最终会在融合后截取 top_k
    vector_top_k = min(top_k * 2, 20)  # 限制最大 20，避免过度检索
    vector_result = retrieve_documents(query, top_k=vector_top_k)
    vector_docs = vector_result.get("docs", [])
    meta = vector_result.get("meta", {})

    meta["graph_enabled"] = False
    meta["graph_entities"] = []
    meta["graph_chunk_count"] = 0
    meta["graph_auto_merge_applied"] = False

    if not GRAPH_ENABLED:
        return {"docs": vector_docs[:top_k], "meta": meta}

    # 2. 图谱检索（同步调用，图谱检索器内部是轻量级 Cypher 查询）
    try:
        from backend.graph.graph_retriever import GraphRetriever
        from backend.graph.neo4j_client import get_neo4j_client

        retriever = GraphRetriever(get_neo4j_client())
        graph_result = retriever.retrieve_by_query(query, top_k)

        if not graph_result.get("graph_enabled", True):
            meta["graph_enabled"] = False
            return {"docs": vector_docs[:top_k], "meta": meta}

    except Exception as e:
        logger.warning(f"图谱检索失败，降级为纯向量检索: {e}")
        meta["graph_enabled"] = False
        meta["graph_error"] = str(e)
        return {"docs": vector_docs[:top_k], "meta": meta}

    graph_docs = graph_result.get("chunks", [])
    meta["graph_enabled"] = True
    meta["graph_entities"] = graph_result.get("entities", [])
    meta["graph_chunk_count"] = len(graph_docs)

    # 3. 对图谱结果应用 auto_merge（与向量检索保持一致）
    if graph_docs:
        graph_docs, graph_merge_meta = _auto_merge_documents(graph_docs, top_k=top_k)
        meta["graph_auto_merge_applied"] = graph_merge_meta.get("auto_merge_applied", False)
        meta["graph_auto_merge_replaced_chunks"] = graph_merge_meta.get("auto_merge_replaced_chunks", 0)

    # 4. 去重融合
    seen = set()
    merged = []

    # 先加入向量检索结果（优先级高）
    for doc in vector_docs:
        key = doc.get("chunk_id") or (doc.get("filename"), doc.get("text", "")[:100])
        if key not in seen:
            seen.add(key)
            doc["source"] = "vector"
            merged.append(doc)

    # 再加入图谱检索结果
    for doc in graph_docs:
        key = doc.get("chunk_id") or (doc.get("doc_id"), doc.get("text", "")[:100])
        if key not in seen:
            seen.add(key)
            doc["source"] = "graph"
            merged.append(doc)

    # 5. 排序（优先向量 rerank_score，其次图谱 graph_score）
    def sort_key(doc):
        score = doc.get("rerank_score") or doc.get("score") or doc.get("graph_score") or 0
        return -score if isinstance(score, (int, float)) else 0

    merged.sort(key=sort_key)

    return {"docs": merged[:top_k], "meta": meta}
