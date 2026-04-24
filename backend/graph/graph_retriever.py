"""知识图谱检索器 - 实体查询与多跳推理"""
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

# 实体识别 Prompt
ENTITY_RECOGNITION_PROMPT = """从用户问题中识别可能涉及的实体名称。

用户问题：{query}

请输出问题中提到的实体名称，JSON 数组格式：
["实体1", "实体2", ...]

要求：
1. 只输出 JSON 数组，不要其他内容
2. 提取问题中明确提到的名称，包括：
   - 产品名称（如：iPhone、幻影V3、小米扫地机器人等）
   - 人名、公司名、地点名
   - 技术术语、专有名词
3. 保持原样输出，不要添加空格或修改
4. 如果没有明显实体，输出空数组 []

示例：
问题："张三在哪家公司工作？"
输出：["张三"]

问题："A公司和B公司有什么合作关系？"
输出：["A公司", "B公司"]

问题："幻影V3包含什么？"
输出：["幻影V3"]

问题："iPhone 15 Pro 的价格是多少？"
输出：["iPhone 15 Pro"]

问题："如何使用这个功能？"
输出：[]"""


class GraphRetriever:
    """知识图谱检索器"""

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

    def extract_query_entities(self, query: str) -> list[str]:
        """从用户问题中识别实体名称"""
        if not self.llm or not query:
            return []

        try:
            prompt = ENTITY_RECOGNITION_PROMPT.format(query=query)
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)

            # 提取 JSON
            content = content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

            entities = json.loads(content)

            if isinstance(entities, list):
                return [str(e) for e in entities if e]
            return []

        except Exception as e:
            logger.debug(f"实体识别失败: {e}")
            return []

    def find_entity(self, name: str) -> Optional[dict]:
        """查找实体节点"""
        query = """
        MATCH (e:Entity {name: $name})
        RETURN e.name as name, e.type as type, e.description as description, e.source_doc as source_doc
        """
        results = self.neo4j.run_query(query, {"name": name})
        return results[0] if results else None

    def get_entity_neighbors(self, entity_name: str, max_depth: int = 2) -> list[dict]:
        """获取实体的邻居节点（多跳查询）"""
        query = f"""
        MATCH path = (e:Entity {{name: $name}})-[:RELATED_TO*1..{max_depth}]-(neighbor:Entity)
        RETURN DISTINCT neighbor.name as name, neighbor.type as type,
               neighbor.description as description, length(path) as distance
        ORDER BY distance
        LIMIT 20
        """
        return self.neo4j.run_query(query, {"name": entity_name})

    def get_related_entities(self, entity_name: str) -> list[dict]:
        """获取与实体直接相关的其他实体"""
        query = """
        MATCH (e:Entity {name: $name})-[r:RELATED_TO]-(other:Entity)
        RETURN other.name as name, other.type as type,
               r.relation as relation, r.confidence as confidence
        ORDER BY r.confidence DESC
        LIMIT 10
        """
        return self.neo4j.run_query(query, {"name": entity_name})

    def multi_hop_query(self, entity_a: str, entity_b: str, max_hops: int = 3) -> list[dict]:
        """查询两个实体之间的关系路径"""
        query = f"""
        MATCH path = shortestPath(
            (a:Entity {{name: $entity_a}})-[:RELATED_TO*1..{max_hops}]-(b:Entity {{name: $entity_b}})
        )
        RETURN [node in nodes(path) | node.name] as entities,
               [rel in relationships(path) | rel.relation] as relations
        """
        results = self.neo4j.run_query(query, {"entity_a": entity_a, "entity_b": entity_b})

        paths = []
        for result in results:
            entities = result.get("entities", [])
            relations = result.get("relations", [])

            # 构建路径描述
            path_steps = []
            for i, entity in enumerate(entities):
                path_steps.append(entity)
                if i < len(relations):
                    path_steps.append(f"--[{relations[i]}]-->")

            paths.append({
                "entities": entities,
                "relations": relations,
                "path_description": " ".join(path_steps),
            })

        return paths

    def get_chunks_by_entities(self, entity_names: list[str], limit: int = 10) -> list[dict]:
        """根据实体列表获取关联的 chunks，支持模糊匹配"""
        if not entity_names:
            return []

        # 先尝试精确匹配，同时关联 Document 节点获取 filename
        query = """
        MATCH (e:Entity)-[m:MENTIONS]-(c:Chunk)
        WHERE e.name IN $entity_names
        OPTIONAL MATCH (d:Document {id: c.doc_id})
        RETURN DISTINCT c.id as chunk_id, c.doc_id as doc_id, c.text as text,
               c.parent_id as parent_chunk_id, c.level as chunk_level,
               d.filename as filename,
               collect(e.name) as matched_entities, sum(m.count) as relevance_score
        ORDER BY relevance_score DESC
        LIMIT $limit
        """
        results = self.neo4j.run_query(query, {"entity_names": entity_names, "limit": limit})

        # 如果精确匹配无结果，尝试模糊匹配
        if not results:
            logger.debug(f"精确匹配无结果，尝试模糊匹配: {entity_names}")
            fuzzy_query = """
            MATCH (e:Entity)-[m:MENTIONS]-(c:Chunk)
            WHERE any(name IN $entity_names WHERE e.name CONTAINS replace(name, ' ', '') OR replace(e.name, ' ', '') CONTAINS name)
            OPTIONAL MATCH (d:Document {id: c.doc_id})
            RETURN DISTINCT c.id as chunk_id, c.doc_id as doc_id, c.text as text,
                   c.parent_id as parent_chunk_id, c.level as chunk_level,
                   d.filename as filename,
                   collect(e.name) as matched_entities, sum(m.count) as relevance_score
            ORDER BY relevance_score DESC
            LIMIT $limit
            """
            results = self.neo4j.run_query(fuzzy_query, {"entity_names": entity_names, "limit": limit})

        return results

    def retrieve_by_query(self, query: str, top_k: int = 5) -> dict:
        """根据用户问题检索相关 chunks

        Returns:
            {
                "entities": [识别到的实体名称],
                "chunks": [{chunk_id, text, doc_id, matched_entities, ...}],
                "graph_enabled": bool,
            }
        """
        # 检查 Neo4j 连接
        if not self.neo4j.is_connected():
            logger.warning("Neo4j 未连接，跳过图谱检索")
            return {"entities": [], "chunks": [], "graph_enabled": False}

        try:
            # 1. 识别问题中的实体
            entities = self.extract_query_entities(query)

            if not entities:
                return {"entities": [], "chunks": [], "graph_enabled": True}

            logger.debug(f"识别到实体: {entities}")

            # 2. 根据实体获取 chunks
            chunks = self.get_chunks_by_entities(entities, limit=top_k * 2)

            # 3. 为每个 chunk 添加来源标记
            for chunk in chunks:
                chunk["source"] = "graph"
                chunk["graph_score"] = chunk.pop("relevance_score", 0)

            return {
                "entities": entities,
                "chunks": chunks[:top_k],
                "graph_enabled": True,
            }

        except Exception as e:
            logger.warning(f"图谱检索失败: {e}")
            return {"entities": [], "chunks": [], "graph_enabled": True, "error": str(e)}
