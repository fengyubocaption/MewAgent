from typing import Optional
import os
import requests
from dotenv import load_dotenv
try:
    from langchain_core.tools import tool
except ImportError:
    from langchain_core.tools import tool

from backend.core.progress import RetrievalState, _retrieval_state, init_retrieval_state  # noqa: F401 — re-export for backward compat

load_dotenv()

AMAP_WEATHER_API = os.getenv("AMAP_WEATHER_API")
AMAP_API_KEY = os.getenv("AMAP_API_KEY")


@tool("get_current_weather")
def get_current_weather(location: str, extensions: Optional[str] = "base") -> str:
    """Get weather information for a location.

    Args:
        location: City name or adcode
        extensions: 'base' for current weather, 'all' for forecast
    """
    if not location:
        return "location参数不能为空"
    if extensions not in ("base", "all"):
        return "extensions参数错误，请输入base或all"

    if not AMAP_WEATHER_API or not AMAP_API_KEY:
        return "天气服务未配置（缺少 AMAP_WEATHER_API 或 AMAP_API_KEY）"

    params = {
        "key": AMAP_API_KEY,
        "city": location,
        "extensions": extensions,
        "output": "json",
    }

    try:
        resp = requests.get(AMAP_WEATHER_API, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "1":
            return f"查询失败：{data.get('info', '未知错误')}"

        if extensions == "base":
            lives = data.get("lives", [])
            if not lives:
                return f"未查询到 {location} 的天气数据"
            w = lives[0]
            return (
                f"【{w.get('city', location)} 实时天气】\n"
                f"天气状况：{w.get('weather', '未知')}\n"
                f"温度：{w.get('temperature', '未知')}℃\n"
                f"湿度：{w.get('humidity', '未知')}%\n"
                f"风向：{w.get('winddirection', '未知')}\n"
                f"风力：{w.get('windpower', '未知')}级\n"
                f"更新时间：{w.get('reporttime', '未知')}"
            )

        forecasts = data.get("forecasts", [])
        if not forecasts:
            return f"未查询到 {location} 的天气预报数据"
        f0 = forecasts[0]
        out = [f"【{f0.get('city', location)} 天气预报】", f"更新时间：{f0.get('reporttime', '未知')}", ""]
        today = (f0.get("casts") or [])[0] if f0.get("casts") else {}
        out += [
            "今日天气：",
            f"  白天：{today.get('dayweather','未知')}",
            f"  夜间：{today.get('nightweather','未知')}",
            f"  气温：{today.get('nighttemp','未知')}~{today.get('daytemp','未知')}℃",
        ]
        return "\n".join(out)

    except requests.exceptions.Timeout:
        return "错误：请求天气服务超时"
    except requests.exceptions.RequestException as e:
        return f"错误：天气服务请求失败 - {e}"
    except Exception as e:
        return f"错误：解析天气数据失败 - {e}"


@tool("search_knowledge_base")
def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """Search for information in the knowledge base using hybrid retrieval (dense + sparse vectors).
    The system automatically evaluates document relevance and rewrites query if needed.
    Can be called multiple times per turn (max 3) to explore different perspectives."""
    # 限制 top_k 范围，防止资源滥用
    top_k = max(1, min(top_k, 20))

    try:
        state = _retrieval_state.get()
    except LookupError:
        state = init_retrieval_state()

    # 检查调用次数上限
    if state.call_count >= state.max_calls:
        # 已达上限，返回累积结果
        if state.accumulated_docs:
            formatted = _format_accumulated_docs(state)
            return f"{formatted}\n\n⚠️ 已达检索上限（{state.max_calls}次），请基于以上信息回答。"
        return f"已达检索上限（{state.max_calls}次），且未找到任何相关文档。请诚实说明无法回答。"

    state.call_count += 1
    current_call = state.call_count

    from backend.rag.rag_pipeline import run_rag_graph

    rag_result = run_rag_graph(query, top_k=top_k)

    new_docs = rag_result.get("docs", []) if isinstance(rag_result, dict) else []
    rag_trace = rag_result.get("rag_trace", {}) if isinstance(rag_result, dict) else {}
    if rag_trace:
        state.rag_traces.append(rag_trace)

    # 去重：与已累积文档比对
    added_count = 0
    for doc in new_docs:
        key = (doc.get("filename"), doc.get("page_number"), doc.get("text"))
        if key not in state.seen_keys:
            state.seen_keys.add(key)
            state.accumulated_docs.append(doc)
            added_count += 1

    # 构建返回内容
    if not state.accumulated_docs:
        if current_call < state.max_calls:
            return (
                f"未找到相关文档（第{current_call}次/最多{state.max_calls}次）。\n"
                "建议换一个查询角度再次检索，或诚实说明无法回答。"
            )
        return "No relevant documents found in the knowledge base."

    formatted = _format_accumulated_docs(state)

    # 本次检索摘要
    strategy = rag_trace.get("rewrite_strategy", "direct") if rag_trace else "direct"
    rerank_score = None
    if state.accumulated_docs:
        for doc in reversed(state.accumulated_docs):
            if doc.get("rerank_score") is not None:
                rerank_score = doc.get("rerank_score")
                break

    summary_lines = [
        f"查询: {query}",
        f"策略: {strategy}",
    ]
    if rerank_score is not None:
        summary_lines.append(f"rerank 最高分: {rerank_score:.2f}")

    # 引导语
    if current_call >= state.max_calls:
        guidance = f"⚠️ 已达检索上限（{state.max_calls}次），请基于以上信息回答。"
    else:
        remaining = state.max_calls - current_call
        guidance = f"如需补充检索其他方面，可再次调用（剩余{remaining}次）；如信息已充分，请直接回答。"

    return (
        f"[检索结果 - 第{current_call}次/最多{state.max_calls}次] "
        f"已累积 {len(state.accumulated_docs)} 篇文档，本次新增 {added_count} 篇：\n\n"
        f"{formatted}\n\n"
        f"---本次检索摘要---\n"
        + "\n".join(summary_lines)
        + f"\n\n{guidance}"
    )


def _format_accumulated_docs(state: RetrievalState) -> str:
    """格式化累积文档列表。"""
    formatted = []
    for i, result in enumerate(state.accumulated_docs, 1):
        source = result.get("filename", "Unknown")
        page = result.get("page_number", "N/A")
        text = result.get("text", "")
        formatted.append(f"[{i}] {source} (Page {page}):\n{text}")
    return "\n\n---\n\n".join(formatted)
