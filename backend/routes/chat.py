import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from backend.agent.agent import chat_with_agent_stream
from backend.middleware.rate_limit import rate_limit
from backend.auth.security import get_current_user
from backend.schemas import ChatRequest
from backend.db.models import User

router = APIRouter()


@router.post("/chat/stream", dependencies=[Depends(rate_limit("chat", 20, 60))])
async def chat_stream_endpoint(request: ChatRequest, current_user: User = Depends(get_current_user)):
    """跟 Agent 对话 (流式)"""

    async def event_generator():
        try:
            session_id = request.session_id or "default_session"
            async for chunk in chat_with_agent_stream(request.message, current_user.username, session_id):
                yield chunk
        except Exception as e:
            error_data = {"type": "error", "content": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
