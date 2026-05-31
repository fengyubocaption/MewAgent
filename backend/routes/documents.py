import logging
import os
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from backend.middleware.rate_limit import rate_limit
from backend.auth.security import require_admin
from backend.schemas import (
    DocumentDeleteResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentUploadResponse,
)
from backend.rag.document_loader import DocumentLoader
from backend.milvus.embedding import embedding_service
from backend.milvus.milvus_client import MilvusManager
from backend.milvus.milvus_writer import MilvusWriter
from backend.rag.parent_chunk_store import ParentChunkStore
from backend.graph.graph_builder import GraphBuilder
from backend.graph.neo4j_client import get_neo4j_client
from backend.db.models import User

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR.parent / "data"
UPLOAD_DIR = DATA_DIR / "documents"

loader = DocumentLoader()
parent_chunk_store = ParentChunkStore()
milvus_manager = MilvusManager()
milvus_writer = MilvusWriter(embedding_service=embedding_service, milvus_manager=milvus_manager)

router = APIRouter()

MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB


@router.get("/documents", response_model=DocumentListResponse, dependencies=[Depends(rate_limit("read", 30, 60))])
async def list_documents(_: User = Depends(require_admin)):
    """获取已上传的文档列表（管理员）"""
    try:
        results = milvus_manager.query(
            output_fields=["filename", "file_type"],
            limit=10000,
        )

        file_stats = {}
        for item in results:
            filename = item.get("filename", "")
            file_type = item.get("file_type", "")
            if filename not in file_stats:
                file_stats[filename] = {
                    "filename": filename,
                    "file_type": file_type,
                    "chunk_count": 0,
                }
            file_stats[filename]["chunk_count"] += 1

        documents = [DocumentInfo(**stats) for stats in file_stats.values()]
        return DocumentListResponse(documents=documents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")


@router.post("/documents/upload", response_model=DocumentUploadResponse, dependencies=[Depends(rate_limit("upload", 3, 60))])
async def upload_document(file: UploadFile = File(...), _: User = Depends(require_admin)):
    """上传文档并进行 embedding（管理员）"""
    try:
        filename = file.filename or ""
        file_lower = filename.lower()
        if not filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")
        if not (
            file_lower.endswith(".pdf")
            or file_lower.endswith((".docx", ".doc"))
            or file_lower.endswith((".xlsx", ".xls"))
        ):
            raise HTTPException(status_code=400, detail="仅支持 PDF、Word 和 Excel 文档")

        # 防止路径遍历攻击
        safe_name = os.path.basename(filename)
        if not safe_name or safe_name.startswith('.'):
            raise HTTPException(status_code=400, detail="无效的文件名")
        filename = safe_name

        os.makedirs(UPLOAD_DIR, exist_ok=True)

        try:
            milvus_writer.delete_document_chunks(filename)
            parent_chunk_store.delete_by_filename(filename)
        except Exception:
            pass

        # 检查文件大小
        content = await file.read(MAX_UPLOAD_SIZE + 1)
        if len(content) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail=f"文件过大，最大支持 {MAX_UPLOAD_SIZE // 1024 // 1024}MB")

        file_path = UPLOAD_DIR / filename
        with open(file_path, "wb") as f:
            f.write(content)

        try:
            new_docs = loader.load_document(str(file_path), filename)
        except Exception as doc_err:
            raise HTTPException(status_code=500, detail=f"文档处理失败: {doc_err}")

        if not new_docs:
            raise HTTPException(status_code=500, detail="文档处理失败，未能提取内容")

        parent_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) in (1, 2)]
        leaf_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) == 3]
        if not leaf_docs:
            raise HTTPException(status_code=500, detail="文档处理失败，未生成可检索叶子分块")

        parent_chunk_store.upsert_documents(parent_docs)
        milvus_writer.write_documents(leaf_docs)

        # 构建知识图谱（如果启用）
        graph_enabled = os.getenv("GRAPH_ENABLED", "true").lower() != "false"
        if graph_enabled:
            try:
                graph_builder = GraphBuilder(get_neo4j_client())
                graph_builder.build_graph_for_document(
                    doc_id=filename,
                    filename=filename,
                    chunks=leaf_docs,
                )
            except Exception as graph_err:
                logger.warning(f"图谱构建失败（不影响文档上传）: {graph_err}")

        return DocumentUploadResponse(
            filename=filename,
            chunks_processed=len(leaf_docs),
            message=(
                f"成功上传并处理 {filename}，叶子分块 {len(leaf_docs)} 个，"
                f"父级分块 {len(parent_docs)} 个（存入 PostgreSQL）"
            ),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文档上传失败: {str(e)}")


@router.delete("/documents/{filename}", response_model=DocumentDeleteResponse)
async def delete_document(filename: str, _: User = Depends(require_admin)):
    """删除文档在 Milvus 中的向量（保留本地文件，管理员）"""
    try:
        result = milvus_writer.delete_document_chunks(filename)

        # 删除 PostgreSQL 中的关系型结构（L1/L2 父分块）
        parent_chunk_store.delete_by_filename(filename)

        # 删除知识图谱数据（如果启用）
        graph_enabled = os.getenv("GRAPH_ENABLED", "true").lower() != "false"
        if graph_enabled:
            try:
                graph_builder = GraphBuilder(get_neo4j_client())
                graph_builder.delete_graph_for_document(filename)
            except Exception as graph_err:
                logger.warning(f"图谱删除失败（不影响文档删除）: {graph_err}")

        return DocumentDeleteResponse(
            filename=filename,
            chunks_deleted=result.get("delete_count", 0) if isinstance(result, dict) else 0,
            message=f"成功删除文档 {filename} 的向量数据（本地文件已保留）",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除文档失败: {str(e)}")
