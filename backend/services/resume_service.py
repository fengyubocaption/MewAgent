"""简历管理服务 — 简历的上传、解析、CRUD。"""

import logging
import os
from pathlib import Path

from fastapi import HTTPException
from sqlalchemy.orm import Session

from backend.agent.interview_tools import _get_llm, _parse_llm_json, RESUME_PARSE_PROMPT
from backend.db.models import Resume
from backend.rag.document_loader import DocumentLoader
from backend.schemas import (
    ResumeDeleteResponse,
    ResumeDetailResponse,
    ResumeInfo,
    ResumeListResponse,
    ResumeUploadResponse,
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR.parent / "data"
RESUME_DIR = DATA_DIR / "resumes"


class ResumeService:
    """简历管理业务逻辑。"""

    def __init__(self):
        self.loader = DocumentLoader()

    def upload_resume(self, db: Session, user_id: int, filename: str, content: bytes) -> ResumeUploadResponse:
        """上传简历文件并解析。"""
        if not filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")

        file_lower = filename.lower()
        if not (file_lower.endswith(".pdf") or file_lower.endswith((".docx", ".doc"))):
            raise HTTPException(status_code=400, detail="仅支持 PDF 和 Word 格式的简历")

        # 防止路径遍历
        safe_name = os.path.basename(filename)
        if not safe_name or safe_name.startswith('.'):
            raise HTTPException(status_code=400, detail="无效的文件名")
        filename = safe_name

        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="简历文件过大，最大支持 10MB")

        os.makedirs(RESUME_DIR, exist_ok=True)

        # 保存文件
        file_path = RESUME_DIR / filename
        with open(file_path, "wb") as f:
            f.write(content)

        # 提取文本
        try:
            docs = self.loader.load_document(str(file_path), filename)
            raw_text = "\n".join(d.get("text", "") for d in docs)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"简历文件解析失败: {e}")

        if not raw_text.strip():
            raise HTTPException(status_code=500, detail="无法从简历中提取文本内容")

        # 使用 LLM 解析结构化数据
        structured_data = self._parse_structured_data(raw_text)

        # 存入数据库
        try:
            resume = Resume(
                user_id=user_id,
                filename=filename,
                file_path=str(file_path),
                raw_text=raw_text[:10000],
                structured_data=structured_data,
            )
            db.add(resume)
            db.commit()
            db.refresh(resume)
            return ResumeUploadResponse(id=resume.id, filename=filename, message="简历上传并解析成功")
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"简历保存失败: {e}")

    def list_resumes(self, db: Session, user_id: int) -> ResumeListResponse:
        """获取当前用户的简历列表。"""
        resumes = (
            db.query(Resume)
            .filter(Resume.user_id == user_id)
            .order_by(Resume.created_at.desc())
            .all()
        )
        return ResumeListResponse(
            resumes=[
                ResumeInfo(
                    id=r.id,
                    filename=r.filename,
                    structured_data=r.structured_data,
                    created_at=r.created_at.isoformat(),
                    updated_at=r.updated_at.isoformat(),
                )
                for r in resumes
            ]
        )

    def get_resume(self, db: Session, user_id: int, resume_id: int) -> ResumeDetailResponse:
        """获取简历详情。"""
        resume = (
            db.query(Resume)
            .filter(Resume.id == resume_id, Resume.user_id == user_id)
            .first()
        )
        if not resume:
            raise HTTPException(status_code=404, detail="简历不存在")
        return ResumeDetailResponse(
            id=resume.id,
            filename=resume.filename,
            raw_text=resume.raw_text,
            structured_data=resume.structured_data,
            created_at=resume.created_at.isoformat(),
            updated_at=resume.updated_at.isoformat(),
        )

    def delete_resume(self, db: Session, user_id: int, resume_id: int) -> ResumeDeleteResponse:
        """删除简历。"""
        resume = (
            db.query(Resume)
            .filter(Resume.id == resume_id, Resume.user_id == user_id)
            .first()
        )
        if not resume:
            raise HTTPException(status_code=404, detail="简历不存在")
        db.delete(resume)
        db.commit()
        return ResumeDeleteResponse(id=resume_id, message="简历已删除")

    @staticmethod
    def _parse_structured_data(raw_text: str) -> dict | None:
        """使用 LLM 解析简历结构化数据。"""
        llm = _get_llm()
        if not llm:
            return None
        try:
            prompt = RESUME_PARSE_PROMPT.format(text=raw_text[:5000])
            response = llm.invoke(prompt)
            resp_content = response.content if hasattr(response, "content") else str(response)
            return _parse_llm_json(resp_content)
        except Exception as e:
            logger.warning(f"简历结构化解析失败: {e}")
            return None


# 模块级单例
resume_service = ResumeService()
