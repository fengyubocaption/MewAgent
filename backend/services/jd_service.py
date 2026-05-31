"""JD 管理服务 — 职位描述的创建、解析、CRUD。"""

import logging

from fastapi import HTTPException
from sqlalchemy.orm import Session

from backend.agent.interview_tools import _get_llm, _parse_llm_json, JD_PARSE_PROMPT
from backend.db.models import JobDescription
from backend.schemas import (
    JDDetailResponse,
    JDCreateRequest,
    JDCreateResponse,
    JDDeleteResponse,
    JDInfo,
    JDListResponse,
)

logger = logging.getLogger(__name__)


class JDService:
    """职位描述管理业务逻辑。"""

    def create_jd(self, db: Session, user_id: int, request: JDCreateRequest) -> JDCreateResponse:
        """创建职位描述。"""
        jd_text = (request.jd_text or "").strip()
        if not jd_text or len(jd_text) < 20:
            raise HTTPException(status_code=400, detail="JD 内容过短，请提供完整的职位描述")

        # 使用 LLM 解析
        structured_data, title, company = self._parse_structured_data(jd_text, request.title, request.company)

        if not title:
            title = "未知职位"

        try:
            jd = JobDescription(
                user_id=user_id,
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

    def list_jds(self, db: Session, user_id: int) -> JDListResponse:
        """获取当前用户的 JD 列表。"""
        jds = (
            db.query(JobDescription)
            .filter(JobDescription.user_id == user_id)
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

    def get_jd(self, db: Session, user_id: int, jd_id: int) -> JDDetailResponse:
        """获取 JD 详情。"""
        jd = (
            db.query(JobDescription)
            .filter(JobDescription.id == jd_id, JobDescription.user_id == user_id)
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

    def delete_jd(self, db: Session, user_id: int, jd_id: int) -> JDDeleteResponse:
        """删除 JD。"""
        jd = (
            db.query(JobDescription)
            .filter(JobDescription.id == jd_id, JobDescription.user_id == user_id)
            .first()
        )
        if not jd:
            raise HTTPException(status_code=404, detail="JD 不存在")
        db.delete(jd)
        db.commit()
        return JDDeleteResponse(id=jd_id, message="JD 已删除")

    @staticmethod
    def _parse_structured_data(jd_text: str, title: str | None, company: str | None):
        """使用 LLM 解析 JD 结构化数据。返回 (structured_data, title, company)。"""
        llm = _get_llm()
        structured_data = None

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

        return structured_data, title, company


# 模块级单例
jd_service = JDService()
