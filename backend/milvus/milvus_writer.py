"""文档向量化并写入 Milvus - 支持密集+稀疏向量"""
from milvus.embedding import EmbeddingService, embedding_service as _default_embedding_service
from milvus.milvus_client import MilvusManager


class MilvusWriter:
    """文档向量化并写入 Milvus 服务 - 支持混合检索"""

    def __init__(self, embedding_service: EmbeddingService = None, milvus_manager: MilvusManager = None):
        self.embedding_service = embedding_service or _default_embedding_service
        self.milvus_manager = milvus_manager or MilvusManager()

    def write_documents(self, documents: list[dict], batch_size: int = 50):
        """
        批量写入文档到 Milvus（同时生成密集和稀疏向量）
        :param documents: 文档列表
        :param batch_size: 批次大小
        """
        if not documents:
            return

        self.milvus_manager.init_collection()

        all_texts = [doc["text"] for doc in documents]
        self.embedding_service.increment_add_documents(all_texts)

        total = len(documents)
        for i in range(0, total, batch_size):
            batch = documents[i:i + batch_size]
            texts = [doc["text"] for doc in batch]
            
            # 同时生成密集向量和稀疏向量
            dense_embeddings, sparse_embeddings = self.embedding_service.get_all_embeddings(texts)

            insert_data = [
                {
                    "dense_embedding": dense_emb,
                    "sparse_embedding": sparse_emb,
                    "text": doc["text"],
                    "filename": doc["filename"],
                    "file_type": doc["file_type"],
                    "file_path": doc.get("file_path", ""),
                    "page_number": doc.get("page_number", 0),
                    "chunk_idx": doc.get("chunk_idx", 0),
                    "chunk_id": doc.get("chunk_id", ""),
                    "parent_chunk_id": doc.get("parent_chunk_id", ""),
                    "root_chunk_id": doc.get("root_chunk_id", ""),
                    "chunk_level": doc.get("chunk_level", 0),
                }
                for doc, dense_emb, sparse_emb in zip(batch, dense_embeddings, sparse_embeddings)
            ]

            self.milvus_manager.insert(insert_data)

    def delete_document_chunks(self, filename: str) -> dict:
        """删除文档的完整生命周期：先扣减 BM25，再删向量"""
        self.milvus_manager.init_collection()

        # 1. 先查出原文
        rows = self.milvus_manager.query_all(
            filter_expr=f'filename == "{filename}"',
            output_fields=["text"],
        )
        texts = [r.get("text") or "" for r in rows]

        if texts:
            # 2. 扣减 BM25 统计
            self.embedding_service.increment_remove_documents(texts)

        # 3. 真正执行 Milvus 删除
        result = self.milvus_manager.delete(f'filename == "{filename}"')
        return result