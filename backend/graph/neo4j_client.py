"""Neo4j 客户端 - 连接管理与基础 CRUD 操作"""
import os
import logging
from typing import Optional

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

logger = logging.getLogger(__name__)

# 默认配置
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
if not NEO4J_PASSWORD:
    raise RuntimeError("NEO4J_PASSWORD environment variable is required")


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
