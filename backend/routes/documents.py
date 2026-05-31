from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from backend.core.rate_limit import rate_limit
from backend.core.security import require_admin
from backend.schemas import (
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentUploadResponse,
)
from backend.services.document_service import document_service, MAX_UPLOAD_SIZE
from backend.db.models import User

router = APIRouter()


@router.get("/documents", response_model=DocumentListResponse, dependencies=[Depends(rate_limit("read", 30, 60))])
async def list_documents(_: User = Depends(require_admin)):
    """获取已上传的文档列表（管理员）"""
    try:
        return document_service.list_documents()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")


@router.post("/documents/upload", response_model=DocumentUploadResponse, dependencies=[Depends(rate_limit("upload", 3, 60))])
async def upload_document(file: UploadFile = File(...), _: User = Depends(require_admin)):
    """上传文档并进行 embedding（管理员）"""
    try:
        content = await file.read(MAX_UPLOAD_SIZE + 1)
        return document_service.upload_document(file.filename or "", content)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文档上传失败: {str(e)}")


@router.delete("/documents/{filename}", response_model=DocumentDeleteResponse)
async def delete_document(filename: str, _: User = Depends(require_admin)):
    """删除文档在 Milvus 中的向量（保留本地文件，管理员）"""
    try:
        return document_service.delete_document(filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除文档失败: {str(e)}")
