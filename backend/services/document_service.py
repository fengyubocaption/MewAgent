"""文档管理服务 — 知识库文档的上传、列表、删除。"""

import logging
import os
from pathlib import Path

from fastapi import HTTPException

from backend.graph.graph_builder import GraphBuilder
from backend.graph.neo4j_client import get_neo4j_client
from backend.milvus.embedding import embedding_service
from backend.milvus.milvus_client import MilvusManager
from backend.milvus.milvus_writer import MilvusWriter
from backend.rag.document_loader import DocumentLoader
from backend.rag.parent_chunk_store import ParentChunkStore
from backend.schemas import (
    DocumentDeleteResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentUploadResponse,
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR.parent / "data"
UPLOAD_DIR = DATA_DIR / "documents"

MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB


class DocumentService:
    """知识库文档管理业务逻辑。"""

    def __init__(self):
        self.loader = DocumentLoader()
        self.parent_chunk_store = ParentChunkStore()
        self.milvus_manager = MilvusManager()
        self.milvus_writer = MilvusWriter(
            embedding_service=embedding_service,
            milvus_manager=self.milvus_manager,
        )

    def list_documents(self) -> DocumentListResponse:
        """获取已上传的文档列表。"""
        results = self.milvus_manager.query(
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

    def upload_document(self, filename: str, content: bytes) -> DocumentUploadResponse:
        """上传文档并进行 embedding。"""
        if not filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")

        file_lower = filename.lower()
        if not (
            file_lower.endswith(".pdf")
            or file_lower.endswith((".docx", ".doc"))
            or file_lower.endswith((".xlsx", ".xls"))
        ):
            raise HTTPException(status_code=400, detail="仅支持 PDF、Word 和 Excel 文档")

        # 防止路径遍历攻击
        import os.path
        safe_name = os.path.basename(filename)
        if not safe_name or safe_name.startswith('.'):
            raise HTTPException(status_code=400, detail="无效的文件名")
        filename = safe_name

        if len(content) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail=f"文件过大，最大支持 {MAX_UPLOAD_SIZE // 1024 // 1024}MB")

        os.makedirs(UPLOAD_DIR, exist_ok=True)

        # 清理旧数据
        try:
            self.milvus_writer.delete_document_chunks(filename)
            self.parent_chunk_store.delete_by_filename(filename)
        except Exception:
            pass

        # 保存文件
        file_path = UPLOAD_DIR / filename
        with open(file_path, "wb") as f:
            f.write(content)

        # 提取文本
        try:
            new_docs = self.loader.load_document(str(file_path), filename)
        except Exception as doc_err:
            raise HTTPException(status_code=500, detail=f"文档处理失败: {doc_err}")

        if not new_docs:
            raise HTTPException(status_code=500, detail="文档处理失败，未能提取内容")

        # 分块
        parent_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) in (1, 2)]
        leaf_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) == 3]
        if not leaf_docs:
            raise HTTPException(status_code=500, detail="文档处理失败，未生成可检索叶子分块")

        # 写入存储
        self.parent_chunk_store.upsert_documents(parent_docs)
        self.milvus_writer.write_documents(leaf_docs)

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

    def delete_document(self, filename: str) -> DocumentDeleteResponse:
        """删除文档在 Milvus 中的向量（保留本地文件）。"""
        result = self.milvus_writer.delete_document_chunks(filename)
        self.parent_chunk_store.delete_by_filename(filename)

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


# 模块级单例
document_service = DocumentService()
