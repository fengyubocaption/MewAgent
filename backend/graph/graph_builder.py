"""知识图谱构建器 - 从文档中抽取实体和关系"""
import json
import logging
import os
from typing import Optional

from langchain.chat_models import init_chat_model

from backend.graph.neo4j_client import Neo4jClient, get_neo4j_client

logger = logging.getLogger(__name__)

API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")

# 实体抽取 Prompt
ENTITY_EXTRACTION_PROMPT = """从以下文本中抽取实体和关系，返回 JSON 格式。

文本：
{text}

输出格式（必须是合法 JSON）：
{{
    "entities": [
        {{"name": "实体名称", "type": "类型", "description": "简短描述"}}
    ],
    "relations": [
        {{"head": "实体A", "relation": "关系类型", "tail": "实体B"}}
    ]
}}

要求：
1. 只抽取文本中明确提及的实体，不要推测
2. 实体类型从以下选择：人、地点、组织、概念、产品、事件、技术、其他
3. 关系类型要具体，避免"相关"、"有关"等模糊描述
4. 实体名称要统一，同一实体使用相同名称
5. 如果没有实体或关系，返回空列表

只输出 JSON，不要其他内容。"""


class GraphBuilder:
    """知识图谱构建器"""

    def __init__(self, neo4j_client: Neo4jClient = None):
        self.neo4j = neo4j_client or get_neo4j_client()
        self._llm = None

    @property
    def llm(self):
        """延迟初始化 LLM"""
        if self._llm is None and API_KEY and MODEL:
            self._llm = init_chat_model(
                model=MODEL,
                model_provider="openai",
                api_key=API_KEY,
                base_url=BASE_URL,
                temperature=0.1,
            )
        return self._llm

    def extract_entities_and_relations(self, text: str) -> dict:
        """从单个文本块抽取实体和关系

        Returns:
            {
                "entities": [{"name", "type", "description"}],
                "relations": [{"head", "relation", "tail"}]
            }
        """
        if not self.llm or not text or len(text.strip()) < 10:
            return {"entities": [], "relations": []}

        try:
            prompt = ENTITY_EXTRACTION_PROMPT.format(text=text[:2000])  # 限制长度
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)

            # 提取 JSON
            content = content.strip()
            if content.startswith("```"):
                # 移除代码块标记
                lines = content.split("\n")
                content = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

            result = json.loads(content)

            # 验证格式
            entities = result.get("entities", [])
            relations = result.get("relations", [])

            # 过滤无效条目
            entities = [
                e for e in entities
                if isinstance(e, dict) and e.get("name") and e.get("type")
            ]
            relations = [
                r for r in relations
                if isinstance(r, dict) and r.get("head") and r.get("relation") and r.get("tail")
            ]

            return {"entities": entities, "relations": relations}

        except json.JSONDecodeError as e:
            logger.warning(f"实体抽取 JSON 解析失败: {e}")
            return {"entities": [], "relations": []}
        except Exception as e:
            logger.warning(f"实体抽取失败: {e}")
            return {"entities": [], "relations": []}

    def create_document_node(self, doc_id: str, filename: str) -> None:
        """创建文档节点"""
        query = """
        MERGE (d:Document {id: $doc_id})
        SET d.filename = $filename, d.upload_time = datetime()
        """
        self.neo4j.run_write(query, {"doc_id": doc_id, "filename": filename})

    def create_chunk_node(self, chunk_id: str, doc_id: str, text: str, level: int = 3, parent_id: str = None) -> None:
        """创建 Chunk 节点"""
        query = """
        MERGE (c:Chunk {id: $chunk_id})
        SET c.doc_id = $doc_id, c.text = $text, c.level = $level, c.parent_id = $parent_id
        WITH c
        MATCH (d:Document {id: $doc_id})
        MERGE (d)-[:CONTAINS]->(c)
        """
        self.neo4j.run_write(query, {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "text": text[:1000],  # 截断存储
            "level": level,
            "parent_id": parent_id,
        })

    def create_entity_node(self, name: str, entity_type: str, description: str, source_doc: str) -> None:
        """创建实体节点"""
        query = """
        MERGE (e:Entity {name: $name})
        SET e.type = $entity_type, e.description = $description, e.source_doc = $source_doc
        """
        self.neo4j.run_write(query, {
            "name": name,
            "entity_type": entity_type,
            "description": description or "",
            "source_doc": source_doc,
        })

    def create_relation(self, head: str, relation: str, tail: str, confidence: float = 1.0) -> None:
        """创建实体间关系"""
        query = """
        MATCH (e1:Entity {name: $head})
        MATCH (e2:Entity {name: $tail})
        MERGE (e1)-[r:RELATED_TO {relation: $relation}]->(e2)
        SET r.confidence = $confidence
        """
        self.neo4j.run_write(query, {
            "head": head,
            "relation": relation,
            "tail": tail,
            "confidence": confidence,
        })

    def link_chunk_to_entity(self, chunk_id: str, entity_name: str) -> None:
        """关联 Chunk 到实体"""
        query = """
        MATCH (c:Chunk {id: $chunk_id})
        MATCH (e:Entity {name: $entity_name})
        MERGE (c)-[r:MENTIONS]->(e)
        SET r.count = COALESCE(r.count, 0) + 1
        """
        self.neo4j.run_write(query, {
            "chunk_id": chunk_id,
            "entity_name": entity_name,
        })

    def build_graph_for_document(self, doc_id: str, filename: str, chunks: list[dict]) -> dict:
        """为文档构建知识图谱

        Args:
            doc_id: 文档 ID（通常用 filename）
            filename: 文件名
            chunks: L3 chunk 列表，每个包含 chunk_id, text, parent_chunk_id 等

        Returns:
            {"entity_count": int, "relation_count": int, "chunk_count": int}
        """
        # 创建文档节点
        self.create_document_node(doc_id, filename)

        entity_count = 0
        relation_count = 0
        all_entities = {}  # name -> {type, description} 去重

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            text = chunk.get("text", "")
            parent_id = chunk.get("parent_chunk_id")

            if not chunk_id or not text:
                continue

            # 创建 Chunk 节点
            self.create_chunk_node(chunk_id, doc_id, text, level=3, parent_id=parent_id)

            # 抽取实体和关系
            extraction = self.extract_entities_and_relations(text)

            # 收集实体（去重）
            for entity in extraction.get("entities", []):
                name = entity.get("name")
                if name and name not in all_entities:
                    all_entities[name] = {
                        "type": entity.get("type", "其他"),
                        "description": entity.get("description", ""),
                    }

            # 创建关系
            for rel in extraction.get("relations", []):
                head = rel.get("head")
                tail = rel.get("tail")
                relation = rel.get("relation")
                if head and tail and relation:
                    try:
                        self.create_relation(head, relation, tail)
                        relation_count += 1
                    except Exception as e:
                        logger.debug(f"创建关系失败 {head}-{relation}->{tail}: {e}")

        # 批量创建实体节点并关联到 chunk
        for name, info in all_entities.items():
            self.create_entity_node(name, info["type"], info["description"], filename)
            entity_count += 1

            # 关联到提及该实体的 chunk
            for chunk in chunks:
                chunk_id = chunk.get("chunk_id")
                text = chunk.get("text", "")
                if chunk_id and name in text:
                    try:
                        self.link_chunk_to_entity(chunk_id, name)
                    except Exception:
                        pass

        logger.info(f"文档 {filename} 图谱构建完成: {entity_count} 实体, {relation_count} 关系, {len(chunks)} chunks")

        return {
            "entity_count": entity_count,
            "relation_count": relation_count,
            "chunk_count": len(chunks),
        }

    def delete_graph_for_document(self, doc_id: str) -> None:
        """删除文档相关的图谱数据"""
        # 删除 Chunk 节点及相关关系
        query_chunks = """
        MATCH (c:Chunk {doc_id: $doc_id})
        DETACH DELETE c
        """
        self.neo4j.run_write(query_chunks, {"doc_id": doc_id})

        # 删除孤立实体（没有其他 Chunk 引用的）
        query_orphans = """
        MATCH (e:Entity)
        WHERE NOT (e)<-[:MENTIONS]-(:Chunk)
        DELETE e
        """
        self.neo4j.run_write(query_orphans)

        # 删除文档节点
        query_doc = """
        MATCH (d:Document {id: $doc_id})
        DETACH DELETE d
        """
        self.neo4j.run_write(query_doc, {"doc_id": doc_id})

        logger.info(f"文档 {doc_id} 图谱数据已删除")
