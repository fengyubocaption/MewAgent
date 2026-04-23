# Graph RAG 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有向量+BM25 检索基础上，引入 Neo4j 知识图谱，增强实体关系推理和多跳查询能力。

**Architecture:** Neo4j 作为图数据库存储实体和关系；文档上传时 LLM 抽取实体构建图谱；检索时向量与图谱并行检索，结果融合去重后参与 auto-merging。

**Tech Stack:** Neo4j 5.x, neo4j Python driver, LangChain LLM

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `docker-compose.yml` | Modify | 新增 Neo4j 服务 |
| `backend/graph/__init__.py` | Create | 模块初始化 |
| `backend/graph/neo4j_client.py` | Create | Neo4j 连接管理、基础 CRUD |
| `backend/graph/graph_builder.py` | Create | 文档上传时构建图谱 |
| `backend/graph/graph_retriever.py` | Create | 图谱查询 + 多跳推理 |
| `backend/rag/rag_utils.py` | Modify | 新增融合检索函数 |
| `backend/rag/rag_pipeline.py` | Modify | retrieve_initial 改用融合检索 |
| `backend/routes/api.py` | Modify | 文档上传/删除时调用图谱构建/清理 |
| `.env.example` | Modify | 新增 Neo4j 环境变量示例 |

---

## Phase 1: Neo4j 部署 + 基础模块

### Task 1: 添加 Neo4j 到 Docker Compose

**Files:**
- Modify: `docker-compose.yml`

- [ ] **Step 1: 添加 Neo4j 服务配置**

在 `docker-compose.yml` 的 `services:` 下添加 Neo4j 服务：

```yaml
  neo4j:
    container_name: supermew-neo4j
    image: neo4j:5.15-community
    ports:
      - "7474:7474"   # HTTP
      - "7687:7687"   # Bolt
    environment:
      - NEO4J_AUTH=neo4j/supermew2024
      - NEO4J_PLUGINS=["apoc"]
      - NEO4J_dbms_memory_heap_initial__size=512m
      - NEO4J_dbms_memory_heap_max__size=1G
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/neo4j:/data
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/neo4j_logs:/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7474"]
      interval: 10s
      timeout: 5s
      retries: 10
    networks:
      - default
```

- [ ] **Step 2: 启动 Neo4j 服务**

```bash
docker compose up -d neo4j
```

- [ ] **Step 3: 验证 Neo4j 运行状态**

```bash
docker compose ps neo4j
curl -I http://localhost:7474
```

Expected: HTTP 200

- [ ] **Step 4: Commit**

```bash
git add docker-compose.yml
git commit -m "feat(infra): 添加 Neo4j 服务到 docker-compose"
```

---

### Task 2: 创建 graph 模块目录结构

**Files:**
- Create: `backend/graph/__init__.py`
- Create: `backend/graph/neo4j_client.py`

- [ ] **Step 1: 创建目录和 __init__.py**

```bash
mkdir -p backend/graph
```

`backend/graph/__init__.py`:
```python
"""知识图谱模块 - Neo4j 实体关系存储与检索"""

from backend.graph.neo4j_client import Neo4jClient, get_neo4j_client
from backend.graph.graph_builder import GraphBuilder
from backend.graph.graph_retriever import GraphRetriever

__all__ = [
    "Neo4jClient",
    "get_neo4j_client",
    "GraphBuilder",
    "GraphRetriever",
]
```

- [ ] **Step 2: 实现 neo4j_client.py 基础连接**

`backend/graph/neo4j_client.py`:
```python
"""Neo4j 客户端 - 连接管理与基础 CRUD 操作"""
import os
import logging
from typing import Optional

from neo4j import GraphDatabase, AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

logger = logging.getLogger(__name__)

# 默认配置
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "supermew2024")


class Neo4jClient:
    """Neo4j 同步客户端"""

    def __init__(self, uri: str = None, user: str = None, password: str = None):
        self.uri = uri or NEO4J_URI
        self.user = user or NEO4J_USER
        self.password = password or NEO4J_PASSWORD
        self._driver = None

    def connect(self):
        """建立连接"""
        if self._driver is None:
            try:
                self._driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.user, self.password)
                )
                # 验证连接
                self._driver.verify_connectivity()
                logger.info(f"Neo4j 连接成功: {self.uri}")
            except (ServiceUnavailable, AuthError) as e:
                logger.error(f"Neo4j 连接失败: {e}")
                raise
        return self._driver

    def close(self):
        """关闭连接"""
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j 连接已关闭")

    @property
    def driver(self):
        """获取驱动实例"""
        if self._driver is None:
            self.connect()
        return self._driver

    def is_connected(self) -> bool:
        """检查连接状态"""
        try:
            if self._driver:
                self._driver.verify_connectivity()
                return True
        except Exception:
            pass
        return False

    def run_query(self, query: str, parameters: dict = None) -> list[dict]:
        """执行 Cypher 查询，返回结果列表"""
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [dict(record) for record in result]

    def run_write(self, query: str, parameters: dict = None) -> None:
        """执行写入操作"""
        with self.driver.session() as session:
            session.run(query, parameters or {})


# 全局单例
_neo4j_client: Optional[Neo4jClient] = None


def get_neo4j_client() -> Neo4jClient:
    """获取 Neo4j 客户端单例"""
    global _neo4j_client
    if _neo4j_client is None:
        _neo4j_client = Neo4jClient()
    return _neo4j_client


def init_neo4j_schema():
    """初始化图谱 Schema（索引）"""
    client = get_neo4j_client()
    
    # 创建索引
    indexes = [
        "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)",
        "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.type)",
        "CREATE INDEX chunk_id_idx IF NOT EXISTS FOR (c:Chunk) ON (c.id)",
        "CREATE INDEX doc_id_idx IF NOT EXISTS FOR (d:Document) ON (d.id)",
    ]
    
    for index_query in indexes:
        try:
            client.run_write(index_query)
        except Exception as e:
            logger.warning(f"创建索引失败（可能已存在）: {e}")
    
    logger.info("Neo4j Schema 初始化完成")
```

- [ ] **Step 3: 添加 neo4j 依赖**

```bash
uv add neo4j
```

- [ ] **Step 4: 测试 Neo4j 连接**

```bash
uv run python -c "
from backend.graph.neo4j_client import get_neo4j_client
client = get_neo4j_client()
print('Connected:', client.is_connected())
client.run_write('CREATE (n:Test {name: \"test\"})')
result = client.run_query('MATCH (n:Test) RETURN n.name as name')
print('Query result:', result)
client.run_write('MATCH (n:Test) DELETE n')
print('Test passed!')
"
```

Expected: `Connected: True`, `Query result: [{'name': 'test'}]`, `Test passed!`

- [ ] **Step 5: Commit**

```bash
git add backend/graph/__init__.py backend/graph/neo4j_client.py pyproject.toml uv.lock
git commit -m "feat(graph): Neo4j 客户端模块 + 基础 Schema"
```

---

### Task 3: 添加 Neo4j 环境变量

**Files:**
- Modify: `.env.example` (如果存在)
- Create: `.env` 更新 (用户手动)

- [ ] **Step 1: 添加环境变量到配置文件**

如果项目有 `.env.example`，添加以下内容：

```env
# ===== Neo4j 知识图谱 =====
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=supermew2024
GRAPH_ENABLED=true
```

- [ ] **Step 2: Commit**

```bash
git add .env.example
git commit -m "docs: 添加 Neo4j 环境变量配置"
```

---

## Phase 2: 实体抽取 + 图谱构建

### Task 4: 实现实体抽取函数

**Files:**
- Create: `backend/graph/graph_builder.py`

- [ ] **Step 1: 实现 graph_builder.py**

`backend/graph/graph_builder.py`:
```python
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
```

- [ ] **Step 2: 测试实体抽取**

```bash
uv run python -c "
from backend.graph.graph_builder import GraphBuilder

builder = GraphBuilder()
text = '''
张三担任A公司的技术总监，负责AI产品研发。
A公司总部位于北京，与B公司在智能硬件领域有深度合作。
'''
result = builder.extract_entities_and_relations(text)
print('Entities:', result['entities'])
print('Relations:', result['relations'])
"
```

Expected: 输出包含张三、A公司、B公司、北京等实体

- [ ] **Step 3: Commit**

```bash
git add backend/graph/graph_builder.py
git commit -m "feat(graph): 实体抽取 + 图谱构建器"
```

---

### Task 5: 集成图谱构建到文档上传流程

**Files:**
- Modify: `backend/routes/api.py`

- [ ] **Step 1: 修改 upload_document 函数**

在 `backend/routes/api.py` 的 `upload_document` 函数中，添加图谱构建逻辑。

找到文件末尾的 `upload_document` 函数，在 `milvus_writer.write_documents(leaf_docs)` 之后添加：

```python
# 在文件顶部添加 import
from backend.graph.graph_builder import GraphBuilder
from backend.graph.neo4j_client import get_neo4j_client
import os

# 在 upload_document 函数中，milvus_writer.write_documents(leaf_docs) 之后添加：

        # 构建知识图谱（如果启用）
        graph_enabled = os.getenv("GRAPH_ENABLED", "true").lower() != "false"
        graph_result = None
        if graph_enabled:
            try:
                graph_builder = GraphBuilder(get_neo4j_client())
                graph_result = graph_builder.build_graph_for_document(
                    doc_id=filename,
                    filename=filename,
                    chunks=leaf_docs,
                )
            except Exception as graph_err:
                logger.warning(f"图谱构建失败（不影响文档上传）: {graph_err}")
                graph_result = None
```

- [ ] **Step 2: 修改 delete_document 函数**

在 `backend/routes/api.py` 的 `delete_document` 函数中，添加图谱删除逻辑。

在 `parent_chunk_store.delete_by_filename(filename)` 之后添加：

```python
        # 删除知识图谱数据（如果启用）
        graph_enabled = os.getenv("GRAPH_ENABLED", "true").lower() != "false"
        if graph_enabled:
            try:
                graph_builder = GraphBuilder(get_neo4j_client())
                graph_builder.delete_graph_for_document(filename)
            except Exception as graph_err:
                logger.warning(f"图谱删除失败（不影响文档删除）: {graph_err}")
```

- [ ] **Step 3: 添加 logger 导入（如果没有）**

在文件顶部确认有：
```python
import logging
logger = logging.getLogger(__name__)
```

- [ ] **Step 4: Commit**

```bash
git add backend/routes/api.py
git commit -m "feat(api): 文档上传/删除时构建/清理知识图谱"
```

---

## Phase 3: 图谱检索 + 多跳查询

### Task 6: 实现图谱检索器

**Files:**
- Create: `backend/graph/graph_retriever.py`

- [ ] **Step 1: 实现 graph_retriever.py**

`backend/graph/graph_retriever.py`:
```python
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
2. 提取问题中明确提到的名称（人名、公司名、产品名、地点等）
3. 如果没有明显实体，输出空数组 []

示例：
问题："张三在哪家公司工作？"
输出：["张三"]

问题："A公司和B公司有什么合作关系？"
输出：["A公司", "B公司"]

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
        """根据实体列表获取关联的 chunks"""
        if not entity_names:
            return []

        query = """
        MATCH (e:Entity)-[m:MENTIONS]-(c:Chunk)
        WHERE e.name IN $entity_names
        RETURN DISTINCT c.id as chunk_id, c.doc_id as doc_id, c.text as text,
               collect(e.name) as matched_entities, sum(m.count) as relevance_score
        ORDER BY relevance_score DESC
        LIMIT $limit
        """
        return self.neo4j.run_query(query, {"entity_names": entity_names, "limit": limit})

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
```

- [ ] **Step 2: 测试图谱检索**

```bash
uv run python -c "
from backend.graph.graph_retriever import GraphRetriever
from backend.graph.neo4j_client import get_neo4j_client

retriever = GraphRetriever(get_neo4j_client())
entities = retriever.extract_query_entities('A公司和B公司有什么合作关系？')
print('Entities:', entities)
"
```

Expected: `Entities: ['A公司', 'B公司']`（或类似结果）

- [ ] **Step 3: Commit**

```bash
git add backend/graph/graph_retriever.py
git commit -m "feat(graph): 图谱检索器 + 多跳查询"
```

---

## Phase 4: 向量+图谱融合检索

### Task 7: 实现融合检索函数

**Files:**
- Modify: `backend/rag/rag_utils.py`

- [ ] **Step 1: 添加融合检索函数**

在 `backend/rag/rag_utils.py` 文件末尾添加：

```python
# ===== 向量 + 图谱融合检索 =====

import asyncio
import os

GRAPH_ENABLED = os.getenv("GRAPH_ENABLED", "true").lower() != "false"


async def _graph_retrieve_async(query: str, top_k: int) -> dict:
    """异步图谱检索"""
    from backend.graph.graph_retriever import GraphRetriever
    from backend.graph.neo4j_client import get_neo4j_client
    
    try:
        retriever = GraphRetriever(get_neo4j_client())
        return retriever.retrieve_by_query(query, top_k)
    except Exception as e:
        logger.warning(f"图谱检索异常: {e}")
        return {"entities": [], "chunks": [], "graph_enabled": False, "error": str(e)}


def retrieve_documents_with_graph(query: str, top_k: int = 5) -> Dict[str, Any]:
    """向量 + 图谱 并行融合检索
    
    Returns:
        {
            "docs": [...],
            "meta": {
                ...原有 meta 信息...,
                "graph_enabled": bool,
                "graph_entities": list[str],
                "graph_chunk_count": int,
            }
        }
    """
    # 1. 向量检索（原有逻辑）
    vector_result = retrieve_documents(query, top_k=top_k * 2)
    vector_docs = vector_result.get("docs", [])
    meta = vector_result.get("meta", {})
    
    meta["graph_enabled"] = False
    meta["graph_entities"] = []
    meta["graph_chunk_count"] = 0
    
    if not GRAPH_ENABLED:
        return {"docs": vector_docs[:top_k], "meta": meta}
    
    # 2. 图谱检索
    try:
        graph_result = _graph_retrieve_async(query, top_k)
        # 由于我们在同步上下文中，需要用 asyncio.run 或检查是否在事件循环中
        try:
            loop = asyncio.get_running_loop()
            # 已在异步上下文，创建任务
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                graph_result = loop.run_in_executor(
                    pool, 
                    lambda: GraphRetriever(get_neo4j_client()).retrieve_by_query(query, top_k)
                )
        except RuntimeError:
            # 不在异步上下文，直接同步调用
            from backend.graph.graph_retriever import GraphRetriever
            from backend.graph.neo4j_client import get_neo4j_client
            retriever = GraphRetriever(get_neo4j_client())
            graph_result = retriever.retrieve_by_query(query, top_k)
    except Exception as e:
        logger.warning(f"图谱检索失败: {e}")
        graph_result = {"entities": [], "chunks": []}
    
    graph_docs = graph_result.get("chunks", [])
    meta["graph_enabled"] = graph_result.get("graph_enabled", True)
    meta["graph_entities"] = graph_result.get("entities", [])
    meta["graph_chunk_count"] = len(graph_docs)
    
    # 3. 去重融合
    seen = set()
    merged = []
    
    # 先加入向量检索结果（优先级高）
    for doc in vector_docs:
        key = doc.get("chunk_id") or (doc.get("filename"), doc.get("text", "")[:100])
        if key not in seen:
            seen.add(key)
            doc["source"] = "vector"
            merged.append(doc)
    
    # 再加入图谱检索结果
    for doc in graph_docs:
        key = doc.get("chunk_id") or (doc.get("doc_id"), doc.get("text", "")[:100])
        if key not in seen:
            seen.add(key)
            doc["source"] = "graph"
            merged.append(doc)
    
    # 4. 排序（优先向量 rerank_score，其次图谱 graph_score）
    def sort_key(doc):
        score = doc.get("rerank_score") or doc.get("score") or doc.get("graph_score") or 0
        return -score if isinstance(score, (int, float)) else 0
    
    merged.sort(key=sort_key)
    
    return {"docs": merged[:top_k], "meta": meta}
```

- [ ] **Step 2: 添加必要的 import**

在文件顶部添加：
```python
from backend.graph.graph_retriever import GraphRetriever
from backend.graph.neo4j_client import get_neo4j_client
```

- [ ] **Step 3: Commit**

```bash
git add backend/rag/rag_utils.py
git commit -m "feat(rag): 向量+图谱融合检索函数"
```

---

### Task 8: 集成融合检索到 RAG Pipeline

**Files:**
- Modify: `backend/rag/rag_pipeline.py`

- [ ] **Step 1: 修改 retrieve_initial 函数**

找到 `retrieve_initial` 函数，将：
```python
    retrieved = retrieve_documents(query, top_k=top_k)
```

替换为：
```python
    from backend.rag.rag_utils import retrieve_documents_with_graph
    retrieved = retrieve_documents_with_graph(query, top_k=top_k)
```

- [ ] **Step 2: 更新 emit_rag_step 输出**

在 `retrieve_initial` 函数中，找到 `emit_rag_step` 调用处，添加图谱信息输出。

在现有的 emit_rag_step 之后添加：
```python
    # 图谱检索信息
    if retrieve_meta.get("graph_enabled"):
        emit_rag_step(
            "🕸️",
            "知识图谱检索",
            f"识别实体: {', '.join(retrieve_meta.get('graph_entities', [])) or '无'}，"
            f"图谱召回: {retrieve_meta.get('graph_chunk_count', 0)} 片段",
        )
```

- [ ] **Step 3: 更新 rag_trace 记录**

在 `rag_trace` 字典中添加图谱信息：
```python
        "graph_enabled": retrieve_meta.get("graph_enabled"),
        "graph_entities": retrieve_meta.get("graph_entities"),
        "graph_chunk_count": retrieve_meta.get("graph_chunk_count"),
```

- [ ] **Step 4: Commit**

```bash
git add backend/rag/rag_pipeline.py
git commit -m "feat(rag): RAG Pipeline 集成图谱融合检索"
```

---

### Task 9: 端到端测试

**Files:**
- 无代码修改，手动测试

- [ ] **Step 1: 启动所有服务**

```bash
docker compose up -d
uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

- [ ] **Step 2: 上传测试文档**

使用 Swagger UI (`http://127.0.0.1:8000/docs`) 或 curl 上传一个包含实体的测试文档。

- [ ] **Step 3: 验证图谱构建**

访问 Neo4j Browser (`http://localhost:7474`)，运行：
```cypher
MATCH (e:Entity) RETURN e LIMIT 25
```

Expected: 看到从文档中抽取的实体节点

- [ ] **Step 4: 测试实体查询**

在前端或 API 提问涉及实体关系的问题，检查响应中的 `rag_trace` 是否包含 `graph_entities` 和 `graph_chunk_count`。

- [ ] **Step 5: 测试多跳查询（手动 Cypher）**

在 Neo4j Browser 中测试：
```cypher
MATCH path = (a:Entity)-[:RELATED_TO*1..3]-(b:Entity)
RETURN path
LIMIT 10
```

---

### Task 10: 最终提交与文档更新

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

- [ ] **Step 1: 更新 CLAUDE.md**

在 `## Architecture` 部分添加图谱模块描述：
```markdown
backend/graph/           — Neo4j 客户端、图谱构建、图谱检索
```

在 `## Key Patterns` 部分添加：
```markdown
- **Graph RAG**: Neo4j 知识图谱 + 向量检索融合，实体关系推理，多跳查询
```

- [ ] **Step 2: 更新 README.md**

在环境变量部分添加 Neo4j 配置说明。

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: 更新文档 - Graph RAG 功能"
```

---

## 完成检查

- [ ] Neo4j 服务正常运行
- [ ] 文档上传时自动构建知识图谱
- [ ] 图谱检索能识别问题中的实体
- [ ] 向量+图谱融合检索正常工作
- [ ] 文档删除时清理图谱数据
- [ ] 图谱服务不可用时自动降级为纯向量检索
