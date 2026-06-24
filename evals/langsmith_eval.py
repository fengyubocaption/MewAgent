import asyncio
import json
import os
import sys
from typing import Any, Optional
from uuid import uuid4

from dotenv import load_dotenv
from langsmith import evaluate

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

load_dotenv()

from backend.agent.agent import chat_with_agent_stream  # noqa: E402


def _extract_answer(outputs: Any) -> str:
    if isinstance(outputs, dict):
        # 优先取真实最终回复字段
        answer = outputs.get("response") or outputs.get("answer") or outputs.get("output")
        return str(answer or "").strip()
    if hasattr(outputs, "outputs") and isinstance(outputs.outputs, dict):
        answer = (
            outputs.outputs.get("response")
            or outputs.outputs.get("answer")
            or outputs.outputs.get("output")
        )
        return str(answer or "").strip()
    return ""


def _extract_reference(reference_outputs: Optional[dict]) -> str:
    if not isinstance(reference_outputs, dict):
        return ""
    for key in ("response", "answer", "output", "expected_answer"):
        value = reference_outputs.get(key)
        if value:
            return str(value).strip()
    return ""

DATASET_NAME = os.getenv("LANGSMITH_EVAL_DATASET", "RAG")
EVAL_USER_ID = os.getenv("LANGSMITH_EVAL_USER", "langsmith_eval_user")


async def _run_agent_stream(question: str, session_id: str) -> dict:
    response_text = ""
    rag_trace = {}

    async for event in chat_with_agent_stream(
        user_text=question,
        user_id=EVAL_USER_ID,
        session_id=session_id,
    ):
        if not event.startswith("data: "):
            continue
        payload = event[len("data: "):].strip()
        if payload == "[DONE]":
            break

        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue

        if data.get("type") == "content":
            response_text += data.get("content", "")
        elif data.get("type") == "trace":
            rag_trace = data.get("rag_trace") or {}
        elif data.get("type") == "error":
            raise RuntimeError(data.get("content") or "Agent stream failed")

    return {
        "response": response_text,
        "rag_trace": rag_trace,
    }


def custom_evaluator(run_outputs: dict, reference_outputs: dict) -> bool:
    """评估最终答案，不评估检索块。"""
    answer = _extract_answer(run_outputs)
    if not answer:
        return False
    if "Retrieved Chunks:" in answer:
        return False

    reference = _extract_reference(reference_outputs)
    if not reference:
        return True

    # 有参考答案时，至少保证存在一定语义重合（使用字符集合重合率做轻量检查）
    answer_chars = {ch for ch in answer if not ch.isspace()}
    ref_chars = {ch for ch in reference if not ch.isspace()}
    if not answer_chars or not ref_chars:
        return False

    overlap = len(answer_chars & ref_chars) / max(1, len(ref_chars))
    return overlap >= 0.2


def target_function(inputs: dict) -> dict:
    """直接调用完整 Agent 流程作为评估对象。"""
    question = inputs["question"]
    session_id = f"langsmith_eval_{uuid4().hex}"
    return asyncio.run(_run_agent_stream(question, session_id))


if __name__ == "__main__":
    evaluate(
        target_function,
        data=DATASET_NAME,
        evaluators=[custom_evaluator],
        experiment_prefix="RAG Pipeline Real Evaluation",
    )
