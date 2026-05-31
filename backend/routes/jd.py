import logging

from fastapi import APIRouter, Depends, HTTPException

from backend.middleware.rate_limit import rate_limit
from backend.auth.security import get_current_user
from backend.schemas import (
    JDDetailResponse,
    JDCreateRequest,
    JDCreateResponse,
    JDDeleteResponse,
    JDInfo,
    JDListResponse,
)
from backend.db.database import SessionLocal
from backend.db.models import User, JobDescription

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/jd", response_model=JDCreateResponse, dependencies=[Depends(rate_limit("upload", 10, 60))])
async def create_jd(request: JDCreateRequest, current_user: User = Depends(get_current_user)):
    """创建职位描述"""
    jd_text = (request.jd_text or "").strip()
    if not jd_text or len(jd_text) < 20:
        raise HTTPException(status_code=400, detail="JD 内容过短，请提供完整的职位描述")

    # 使用 LLM 解析
    from backend.agent.interview_tools import _get_llm, _parse_llm_json, JD_PARSE_PROMPT

    llm = _get_llm()
    structured_data = None
    title = request.title
    company = request.company

    if llm:
        try:
            prompt = JD_PARSE_PROMPT.format(text=jd_text[:5000])
            response = llm.invoke(prompt)
            resp_content = response.content if hasattr(response, "content") else str(response)
            structured_data = _parse_llm_json(resp_content)
            if structured_data:
                title = title or structured_data.get("title", "")
                company = company or structured_data.get("company", "")
        except Exception as e:
            logger.warning(f"JD 结构化解析失败: {e}")

    if not title:
        title = "未知职位"

    db = SessionLocal()
    try:
        jd = JobDescription(
            user_id=current_user.id,
            title=title,
            company=company,
            raw_text=jd_text[:10000],
            structured_data=structured_data,
        )
        db.add(jd)
        db.commit()
        db.refresh(jd)
        return JDCreateResponse(id=jd.id, title=title, company=company, message="JD 创建并解析成功")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"JD 保存失败: {e}")
    finally:
        db.close()


@router.get("/jd", response_model=JDListResponse, dependencies=[Depends(rate_limit("read", 30, 60))])
async def list_jds(current_user: User = Depends(get_current_user)):
    """获取当前用户的 JD 列表"""
    db = SessionLocal()
    try:
        jds = (
            db.query(JobDescription)
            .filter(JobDescription.user_id == current_user.id)
            .order_by(JobDescription.created_at.desc())
            .all()
        )
        return JDListResponse(
            job_descriptions=[
                JDInfo(
                    id=j.id,
                    title=j.title,
                    company=j.company,
                    structured_data=j.structured_data,
                    created_at=j.created_at.isoformat(),
                    updated_at=j.updated_at.isoformat(),
                )
                for j in jds
            ]
        )
    finally:
        db.close()


@router.get("/jd/{jd_id}", response_model=JDDetailResponse, dependencies=[Depends(rate_limit("read", 30, 60))])
async def get_jd(jd_id: int, current_user: User = Depends(get_current_user)):
    """获取 JD 详情"""
    db = SessionLocal()
    try:
        jd = (
            db.query(JobDescription)
            .filter(JobDescription.id == jd_id, JobDescription.user_id == current_user.id)
            .first()
        )
        if not jd:
            raise HTTPException(status_code=404, detail="JD 不存在")
        return JDDetailResponse(
            id=jd.id,
            title=jd.title,
            company=jd.company,
            raw_text=jd.raw_text,
            structured_data=jd.structured_data,
            created_at=jd.created_at.isoformat(),
            updated_at=jd.updated_at.isoformat(),
        )
    finally:
        db.close()


@router.delete("/jd/{jd_id}", response_model=JDDeleteResponse)
async def delete_jd(jd_id: int, current_user: User = Depends(get_current_user)):
    """删除 JD"""
    db = SessionLocal()
    try:
        jd = (
            db.query(JobDescription)
            .filter(JobDescription.id == jd_id, JobDescription.user_id == current_user.id)
            .first()
        )
        if not jd:
            raise HTTPException(status_code=404, detail="JD 不存在")
        db.delete(jd)
        db.commit()
        return JDDeleteResponse(id=jd_id, message="JD 已删除")
    finally:
        db.close()
