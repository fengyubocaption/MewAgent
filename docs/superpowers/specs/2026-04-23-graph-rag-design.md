# Graph RAG 设计文档

> **目标**: 在现有向量+BM25 检索基础上，引入 Neo4j 知识图谱，增强实体关系推理和多跳查询能力。

## 核心决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 图数据库 | Neo4j | 成熟稳定，Cypher 查询强大，生态完善 |
| 实体抽取 | LLM 抽取 | 灵活，能处理复杂语义，无需训练 |
| 构建时机 | 文档上传时离线构建 | 查询时无延迟，图谱始终就绪 |
| 检索融合 | 并行检索 + 融合 | 召回全面，通用场景适用 |

---

## 架构设计

### 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                      文档上传流程                            │
├─────────────────────────────────────────────────────────────┤
│  PDF/Word ──→ 三级分块 ──→ L3 chunks                        │
│                  │              │                           │
│                  │              ├──→ 向量化 ──→ Milvus      │
│                  │              │                           │
│                  │              └──→ 实体抽取(LLM) ──→ Neo4j│
│                  │                          │                │
│                  └──→ L1/L2 存储 ──→ PostgreSQL            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      检索流程                               │
├─────────────────────────────────────────────────────────────┤
│  用户问题                                                   │
│      │                                                      │
│      ├──→ 向量+BM25 检索 ──→ Milvus ──→ 候选 chunks A      │
│      │                                                      │
│      ├──→ 实体识别(LLM) ──→ 图谱查询 ──→ Neo4j ──→ chunks B│
│      │                                                      │
│      └──→ 融合去重 ──→ Auto-merging ──→ 最终上下文         │
└─────────────────────────────────────────────────────────────┘
```

### 与现有系统集成点

| 现有组件 | 集成方式 |
|----------|----------|
| 三级分块 | L3 chunk 作为图谱节点，保留 chunk_id 关联 |
| Auto-merging | 图谱检索结果通过 chunk_id 参与 auto-merging |
| Agentic RAG | 多轮检索能力保持，`search_knowledge_base` 内部调用融合检索 |
| Milvus | 并行检索，结果融合去重 |

---

## Neo4j 图谱 Schema

### 节点类型

```cypher
// 实体节点
CREATE (:Entity {
    name: string,        // 实体名称
    type: string,        // 类型：人/地点/概念/产品/事件...
    description: string, // 简要描述
    source_doc: string   // 来源文档
})

// Chunk 节点（L3 叶子块）
CREATE (:Chunk {
    id: string,          // chunk_id
    text: string,        // 文本内容
    level: int,          // 固定为 3
    doc_id: string,      // 所属文档 ID
    parent_id: string    // 父块 ID (L2)
})

// 文档节点
CREATE (:Document {
    id: string,          // 文档 ID
    filename: string,    // 文件名
    upload_time: datetime
})
```

### 关系类型

```cypher
// Chunk 提及实体-[:MENTIONS {count: int}]->(e:Entity)

// 实体间关系
(e1:Entity)-[:RELATED_TO {
    relation: string,    // 关系类型：属于/位于/包含/合作...
    confidence: float    // 置信度
}]->(e2:Entity)

// 文档包含 Chunk
(d:Document)-[:CONTAINS]->(c:Chunk)

// 同一父块下的 Chunk（用于多跳扩展）
(c1:Chunk)-[:SAME_PARENT]->(c2:Chunk)
```

### 索引设计

```cypher
// 实体名称索引（快速查找实体）
CREATE INDEX entity_name_idx FOR (e:Entity) ON (e.name);

// 实体类型索引（按类型过滤）
CREATE INDEX entity_type_idx FOR (e:Entity) ON (e.type);

// Chunk ID 索引（回溯原始文本）
CREATE INDEX chunk_id_idx FOR (c:Chunk) ON (c.id);

// 文档 ID 索引
CREATE INDEX doc_id_idx FOR (d:Document) ON (d.id);
```

---

## 模块设计

### 新增目录结构

```
backend/
└── graph/
    ├── __init__.py
    ├── neo4j_client.py      # Neo4j 连接管理、基础 CRUD
    ├── graph_builder.py     # 文档上传时构建图谱
    └── graph_retriever.py   # 图谱查询 + 多跳推理
```

### neo4j_client.py

负责 Neo4j 连接管理和基础操作：

```python
class Neo4jClient:
    """Neo4j 连接管理器"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    async def create_entity(self, name: str, entity_type: str, description: str, source_doc: str) -> str:
        """创建实体节点，返回实体 ID"""
        
    async def create_relation(self, head: str, relation: str, tail: str, confidence: float = 1.0):
        """创建实体间关系"""
        
    async def link_chunk_to_entity(self, chunk_id: str, entity_name: str):
        """关联 chunk 到实体"""
        
    async def find_entity(self, name: str) -> Optional[dict]:
        """按名称查找实体"""
        
    async def get_entity_neighbors(self, entity_name: str, max_depth: int = 2) -> list[dict]:
        """获取实体的邻居节点（多跳查询）"""
        
    async def multi_hop_query(self, entity_a: str, entity_b: str, max_hops: int = 3) -> list[dict]:
        """查询两个实体之间的关系路径"""
        
    async def get_chunks_by_entities(self, entity_names: list[str]) -> list[str]:
        """根据实体列表获取关联的 chunk_ids"""
```

### graph_builder.py

文档上传时的图谱构建逻辑：

```python
class GraphBuilder:
    """知识图谱构建器"""
    
    def __init__(self, neo4j_client: Neo4jClient, llm_client):
        self.neo4j = neo4j_client
        self.llm = llm_client
    
    async def extract_entities_and_relations(self, chunk: dict) -> dict:
        """LLM 从单个 chunk 抽取实体和关系"""
        prompt = f"""
从以下文本中抽取实体和关系，返回 JSON 格式：

文本：{chunk['text']}

输出格式：
{{
    "entities": [
        {{"name": "实体名", "type": "人/地点/概念/产品/事件...", "description": "简述"}}
    ],
    "relations": [
        {{"head": "实体A", "relation": "关系类型", "tail": "实体B"}}
    ]
}}

要求：
1. 只抽取文本中明确提及的实体，不要推测
2. 关系类型要具体，避免"相关"等模糊描述
3. 实体名称要统一，避免同义不同名
"""
        # 调用 LLM，解析返回结果
        
    async def build_graph_for_document(self, doc_id: str, filename: str, chunks: list[dict]):
        """为文档构建知识图谱
        
        流程：
        1. 创建 Document 节点
        2. 为每个 L3 chunk 创建 Chunk 节点
        3. LLM 抽取实体和关系
        4. 创建 Entity 节点和关系
        5. 关联 Chunk 到 Entity
        6. 建立同父块 Chunk 间的关系
        """
        
    async def delete_graph_for_document(self, doc_id: str):
        """删除文档相关的图谱数据"""
```

### graph_retriever.py

图谱检索和多跳推理：

```python
class GraphRetriever:
    """知识图谱检索器"""
    
    def __init__(self, neo4j_client: Neo4jClient, llm_client):
        self.neo4j = neo4j_client
        self.llm = llm_client
    
    async def extract_query_entities(self, query: str) -> list[str]:
        """从用户问题中识别实体
        
        例如："A公司和B公司的合作项目有哪些"
        返回：["A公司", "B公司"]
        """
        
    async def retrieve_by_entities(self, entity_names: list[str], top_k: int = 5) -> list[dict]:
        """根据实体检索相关 chunks
        
        流程：
        1. 在图谱中查找实体节点
        2. 获取关联的 Chunk 节点
        3. 返回 chunk 详情
        """
        
    async def multi_hop_retrieve(self, entity_a: str, entity_b: str, max_hops: int = 3) -> dict:
        """多跳推理检索
        
        例如："A和B之间有什么关系？"
        
        返回：
        {
            "path": [(entity1, relation, entity2), ...],
            "chunks": [...],
            "explanation": "A 通过 C 与 B 相连..."
        }
        """
        
    async def expand_by_relations(self, entity_name: str, relation_types: list[str] = None) -> list[dict]:
        """根据关系类型扩展检索
        
        例如：查找某公司的所有"合作伙伴"
        """
```

---

## 检索融合设计

### 融合流程

```python
# backend/rag/rag_utils.py 新增

async def hybrid_retrieve_with_graph(
    query: str, 
    top_k: int = 5,
    enable_graph: bool = True
) -> tuple[list[dict], dict]:
    """向量 + 图谱 并行检索融合
    
    返回：(检索结果, 融合trace)
    """
    
    trace = {
        "vector_count": 0,
        "graph_count": 0,
        "merged_count": 0,
        "graph_entities": [],
    }
    
    if not enable_graph:
        # 降级为纯向量检索
        results = await retrieve_documents(query, top_k)
        trace["vector_count"] = len(results)
        return results, trace
    
    # 并行执行
    vector_task = asyncio.create_task(retrieve_documents(query, top_k * 2))
    graph_task = asyncio.create_task(_graph_retrieve(query, top_k))
    
    vector_results, graph_results = await asyncio.gather(
        vector_task, graph_task, return_exceptions=True
    )
    
    # 处理异常降级
    if isinstance(vector_results, Exception):
        vector_results = []
    if isinstance(graph_results, Exception):
        graph_results = []
    
    trace["vector_count"] = len(vector_results)
    trace["graph_count"] = len(graph_results.get("chunks", [])) if graph_results else 0
    trace["graph_entities"] = graph_results.get("entities", []) if graph_results else []
    
    # 去重融合（基于 chunk_id）
    seen = set()
    merged = []
    for doc in vector_results:
        key = doc.get("chunk_id") or (doc.get("filename"), doc.get("text")[:100])
        if key not in seen:
            seen.add(key)
            merged.append(doc)
    
    for doc in graph_results.get("chunks", []):
        key = doc.get("chunk_id") or (doc.get("filename"), doc.get("text")[:100])
        if key not in seen:
            seen.add(key)
            merged.append(doc)
    
    # 按分数排序（向量用 rerank_score，图谱用图谱相关性分数）
    merged.sort(key=lambda x: x.get("rerank_score", x.get("graph_score", 0)), reverse=True)
    
    trace["merged_count"] = len(merged[:top_k])
    
    return merged[:top_k], trace


async def _graph_retrieve(query: str, top_k: int) -> dict:
    """图谱检索内部实现"""
    from backend.graph.graph_retriever import GraphRetriever
    from backend.graph.neo4j_client import get_neo4j_client
    
    neo4j_client = get_neo4j_client()
    retriever = GraphRetriever(neo4j_client, get_llm_client())
    
    # 1. 识别问题中的实体
    entities = await retriever.extract_query_entities(query)
    
    if not entities:
        return {"chunks": [], "entities": []}
    
    # 2. 根据实体检索 chunks
    chunks = await retriever.retrieve_by_entities(entities, top_k)
    
    return {
        "chunks": chunks,
        "entities": entities,
    }
```

### 与 Auto-merging 集成

```python
# 图谱检索结果也参与 auto-merging
# 获取到 chunk_id 后，通过 parent_chunk_store 查找父块

async def auto_merge_graph_results(chunks: list[dict], threshold: float = 0.7) -> list[dict]:
    """对图谱检索结果应用 auto-merging"""
    from backend.milvus.parent_chunk_store import DocStore
    
    doc_store = DocStore()
    merged = []
    
    for chunk in chunks:
        # 如果相似度高，尝试合并到父块
        if chunk.get("score", 0) >= threshold:
            parent_id = chunk.get("parent_chunk_id")
            if parent_id:
                parent = doc_store.get(parent_id)
                if parent:
                    chunk["merged_text"] = parent.get("text", chunk.get("text", ""))
        merged.append(chunk)
    
    return merged
```

---

## 文档上传流程改动

### api.py 改动

```python
# backend/routes/api.py

@router.post("/documents/upload")
async def upload_document(file: UploadFile, current_user: User = Depends(require_admin)):
    # ... 现有的文件处理、分块逻辑 ...
    
    # 新增：异步构建知识图谱
    from backend.graph.graph_builder import GraphBuilder
    graph_builder = GraphBuilder(get_neo4j_client(), get_llm_client())
    
    # 可以选择同步或异步构建
    # 方案A：同步等待（简单但上传慢）
    await graph_builder.build_graph_for_document(doc_id, filename, l3_chunks)
    
    # 方案B：后台任务（推荐）
    # asyncio.create_task(graph_builder.build_graph_for_document(doc_id, filename, l3_chunks))
    
    # ... 返回响应 ...
```

### 文档删除时的图谱清理

```python
@router.delete("/documents/{filename}")
async def delete_document(filename: str, current_user: User = Depends(require_admin)):
    # ... 现有的删除逻辑 ...
    
    # 新增：删除图谱数据
    graph_builder = GraphBuilder(get_neo4j_client(), get_llm_client())
    await graph_builder.delete_graph_for_document(doc_id)
    
    # ... 返回响应 ...
```

---

## 环境变量

```env
# Neo4j 配置
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# 图谱构建配置
GRAPH_ENABLED=true
GRAPH_ENTITY_TYPES=人,地点,概念,产品,事件,组织
GRAPH_MAX_HOPS=3
```

---

## Docker Compose 改动

```yaml
# docker-compose.yml 新增

services:
  neo4j:
    image: neo4j:5.15-community
    container_name: supermew-neo4j
    ports:
      - "7474:7474"   # HTTP
      - "7687:7687"   # Bolt
    environment:
      - NEO4J_AUTH=neo4j/your_password
      - NEO4J_PLUGINS=["apoc"]
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7474"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  neo4j_data:
  neo4j_logs:
```

---

## RAG Pipeline 改动

```python
# backend/rag/rag_pipeline.py

async def retrieve_initial(state: RAGState) -> dict:
    """初始检索节点 - 改为融合检索"""
    query = state["query"]
    top_k = state.get("top_k", 5)
    
    # 使用融合检索替代纯向量检索
    docs, graph_trace = await hybrid_retrieve_with_graph(query, top_k)
    
    return {
        "docs": docs,
        "graph_trace": graph_trace,
    }
```

---

## 错误处理与降级策略

```python
# 图谱服务不可用时的降级

async def hybrid_retrieve_with_graph(query: str, top_k: int = 5) -> tuple[list[dict], dict]:
    try:
        # 尝试图谱检索
        graph_results = await _graph_retrieve(query, top_k)
    except Neo4jError as e:
        logger.warning(f"Neo4j 查询失败，降级为纯向量检索: {e}")
        graph_results = {"chunks": [], "entities": []}
    except LLMError as e:
        logger.warning(f"实体抽取失败，跳过图谱检索: {e}")
        graph_results = {"chunks": [], "entities": []}
    
    # 继续融合逻辑...
```

---

## 性能考量

| 操作 | 预估延迟 | 优化方案 |
|------|----------|----------|
| LLM 实体抽取（单 chunk） | 1-3s | 批量处理 + 缓存 |
| 图谱查询（单实体） | 10-50ms | 索引优化 |
| 多跳查询（2-3跳） | 50-200ms | 限制路径数 |
| 向量+图谱并行检索 | max(向量, 图谱) | asyncio.gather |

---

## 实施计划

| 阶段 | 内容 | 预估工作量 |
|------|------|-----------|
| **Phase 1** | Neo4j 部署 + 基础 Schema + 连接模块 | 1-2 天 |
| **Phase 2** | 实体抽取 Pipeline + 图谱构建 | 2-3 天 |
| **Phase 3** | 图谱检索 + 多跳查询 | 2-3 天 |
| **Phase 4** | 向量+图谱融合 + 集成测试 | 1-2 天 |
| **Phase 5** | 性能优化 + 监控 | 1 天 |

**总计：7-11 天**

---

## 测试计划

### 单元测试

- [ ] Neo4j 连接与 CRUD 操作
- [ ] 实体抽取函数
- [ ] 图谱构建流程
- [ ] 图谱查询（单实体、多跳）
- [ ] 融合去重逻辑

### 集成测试

- [ ] 文档上传 → 图谱构建 → 验证图谱结构
- [ ] 实体查询 → 验证返回 chunks
- [ ] 多跳查询 → 验证路径正确性
- [ ] 向量+图谱融合 → 验证去重和排序

### 端到端测试

- [ ] 上传包含实体的文档，询问实体相关问题，验证回答正确
- [ ] 询问"A和B的关系"类型问题，验证多跳推理
- [ ] 图谱服务不可用时，验证降级为纯向量检索
