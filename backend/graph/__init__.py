"""知识图谱模块 - Neo4j 实体关系存储与检索"""

from backend.graph.neo4j_client import Neo4jClient, get_neo4j_client

# 延迟导入，避免循环依赖
__all__ = [
    "Neo4jClient",
    "get_neo4j_client",
]


def __getattr__(name):
    """延迟导入 GraphBuilder 和 GraphRetriever"""
    if name == "GraphBuilder":
        from backend.graph.graph_builder import GraphBuilder
        return GraphBuilder
    if name == "GraphRetriever":
        from backend.graph.graph_retriever import GraphRetriever
        return GraphRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
