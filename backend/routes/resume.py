from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.core.rate_limit import rate_limit
from backend.core.security import get_current_user
from backend.db.database import get_db
from backend.db.models import User
from backend.schemas import (
    ResumeDeleteResponse,
    ResumeDetailResponse,
    ResumeListResponse,
    ResumeUploadResponse,
)
from backend.services.resume_service import resume_service

router = APIRouter()


@router.post("/resume/upload", response_model=ResumeUploadResponse, dependencies=[Depends(rate_limit("upload", 5, 60))])
async def upload_resume(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """上传简历文件并解析"""
    content = await file.read(10 * 1024 * 1024 + 1)
    return resume_service.upload_resume(db, current_user.id, file.filename or "", content)


@router.get("/resume", response_model=ResumeListResponse, dependencies=[Depends(rate_limit("read", 30, 60))])
async def list_resumes(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """获取当前用户的简历列表"""
    return resume_service.list_resumes(db, current_user.id)


@router.get("/resume/{resume_id}", response_model=ResumeDetailResponse, dependencies=[Depends(rate_limit("read", 30, 60))])
async def get_resume(resume_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """获取简历详情"""
    return resume_service.get_resume(db, current_user.id, resume_id)


@router.delete("/resume/{resume_id}", response_model=ResumeDeleteResponse)
async def delete_resume(resume_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """删除简历"""
    return resume_service.delete_resume(db, current_user.id, resume_id)
