from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.core.rate_limit import rate_limit
from backend.core.security import get_current_user
from backend.db.database import get_db
from backend.db.models import User
from backend.schemas import (
    JDDetailResponse,
    JDCreateRequest,
    JDCreateResponse,
    JDDeleteResponse,
    JDListResponse,
)
from backend.services.jd_service import jd_service

router = APIRouter()


@router.post("/jd", response_model=JDCreateResponse, dependencies=[Depends(rate_limit("upload", 10, 60))])
async def create_jd(request: JDCreateRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """创建职位描述"""
    return jd_service.create_jd(db, current_user.id, request)


@router.get("/jd", response_model=JDListResponse, dependencies=[Depends(rate_limit("read", 30, 60))])
async def list_jds(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """获取当前用户的 JD 列表"""
    return jd_service.list_jds(db, current_user.id)


@router.get("/jd/{jd_id}", response_model=JDDetailResponse, dependencies=[Depends(rate_limit("read", 30, 60))])
async def get_jd(jd_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """获取 JD 详情"""
    return jd_service.get_jd(db, current_user.id, jd_id)


@router.delete("/jd/{jd_id}", response_model=JDDeleteResponse)
async def delete_jd(jd_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """删除 JD"""
    return jd_service.delete_jd(db, current_user.id, jd_id)
