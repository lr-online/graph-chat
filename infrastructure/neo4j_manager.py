from neo4j import AsyncGraphDatabase
from typing import List, Dict, Optional
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Neo4jConnectionPool:
    """支持上下文管理器的 Neo4j 异步连接池管理类"""

    def __init__(self, uri: str, username: str, password: str, max_size: int = 5):
        self._uri = uri
        self._username = username
        self._password = password
        self._max_size = max_size
        self._pool = asyncio.Queue(max_size)
        self._driver = AsyncGraphDatabase.driver(uri, auth=(username, password))

    async def init_pool(self):
        """初始化连接池"""
        for _ in range(self._max_size):
            session = self._driver.session()
            await self._pool.put(session)
        logger.info(f"初始化 Neo4j 连接池，大小: {self._max_size}")

    async def acquire(self):
        """获取连接"""
        return await self._pool.get()

    async def release(self, session):
        """释放连接"""
        await self._pool.put(session)

    async def close(self):
        """关闭连接池"""
        while not self._pool.empty():
            session = await self._pool.get()
            await session.close()
        await self._driver.close()
        logger.info("Neo4j 连接池已关闭")

    async def __aenter__(self):
        """上下文管理器 - 获取连接"""
        self._current_session = await self.acquire()
        return self._current_session

    async def __aexit__(self, exc_type, exc_value, traceback):
        """上下文管理器 - 释放连接"""
        await self.release(self._current_session)


class Neo4jManager:
    """基于上下文管理连接池的 Neo4j 管理类"""

    def __init__(self, connection_pool: Neo4jConnectionPool):
        self._connection_pool = connection_pool

    async def execute_query(self, query: str, parameters: Optional[dict] = None) -> List[dict]:
        """
        执行查询并返回结果
        """
        async with self._connection_pool as session:
            logger.info(f"执行查询: {query} 参数: {parameters}")
            result = await session.run(query, parameters or {})
            data = []
            async for record in result:
                data.append(record.data())
            return data

    async def create_node(self, label: str, properties: Dict[str, str]):
        """创建节点"""
        query = f"CREATE (n:{label} $props) RETURN n"
        return await self.execute_query(query, {"props": properties})

    async def create_relationship(
        self,
        source_label: str,
        source_properties: dict,
        target_label: str,
        target_properties: dict,
        relationship_type: str,
        relationship_properties: dict = None,
    ):
        """创建关系"""
        relationship_properties = relationship_properties or {}
        source_conditions = " AND ".join([f"a.{key} = ${key}" for key in source_properties.keys()])
        target_conditions = " AND ".join([f"b.{key} = ${key}" for key in target_properties.keys()])
        relationship_props = ", ".join([f"{key}: ${key}" for key in relationship_properties.keys()])

        query = f"""
        MATCH (a:{source_label}), (b:{target_label})
        WHERE {source_conditions} AND {target_conditions}
        MERGE (a)-[r:{relationship_type} {{{relationship_props}}}]->(b)
        RETURN r
        """
        parameters = {**source_properties, **target_properties, **relationship_properties}
        return await self.execute_query(query, parameters)

    async def update_node_properties(self, label: str, properties: dict, new_properties: dict):
        """更新节点属性"""
        conditions = " AND ".join([f"n.{key} = ${key}" for key in properties.keys()])
        updates = ", ".join([f"n.{key} = ${key}" for key in new_properties.keys()])
        query = f"""
        MATCH (n:{label})
        WHERE {conditions}
        SET {updates}
        RETURN n
        """
        parameters = {**properties, **new_properties}
        return await self.execute_query(query, parameters)

    async def update_relationship_properties(
        self, source_label: str, target_label: str, relation: str, new_properties: dict
    ):
        """更新关系属性"""
        updates = ", ".join([f"r.{key} = ${key}" for key in new_properties.keys()])
        query = f"""
        MATCH (a:{source_label})-[r:{relation}]->(b:{target_label})
        SET {updates}
        RETURN r
        """
        return await self.execute_query(query, new_properties)

    async def get_all_nodes(self, label: Optional[str] = None) -> List[Dict]:
        """获取所有节点"""
        query = f"MATCH (n{':' + label if label else ''}) RETURN n"
        result = await self.execute_query(query)
        return [
            {
                "labels": record["n"].get("labels", []),
                "properties": record["n"],  # record["n"] 已经是节点属性的字典
            }
            for record in result
        ]

    async def get_relationships(self, source_label: str, target_label: str, relation: str) -> List[Dict]:
        """获取特定关系及其关联节点"""
        query = f"""
        MATCH (a:{source_label})-[r:{relation}]->(b:{target_label})
        RETURN id(a) as source_id, labels(a) as source_labels, a as source_properties,
               id(r) as relationship_id, type(r) as relationship_type, r as relationship_properties,
               id(b) as target_id, labels(b) as target_labels, b as target_properties
        """
        result = await self.execute_query(query)
        return [
            {
                "source": {
                    "id": record["source_id"],
                    "labels": record["source_labels"],
                    "properties": record["source_properties"],
                },
                "relationship": {
                    "id": record["relationship_id"],
                    "type": record["relationship_type"],
                    "properties": record["relationship_properties"],
                },
                "target": {
                    "id": record["target_id"],
                    "labels": record["target_labels"],
                    "properties": record["target_properties"],
                },
            }
            for record in result
        ]

    async def get_related_concepts(self, label: str, properties: dict, relation: str, max_distance: int = 2) -> List[
        Dict]:
        """获取节点的所有关联关系（支持多跳）"""
        conditions = " AND ".join([f"n.{key} = ${key}" for key in properties.keys()])
        query = f"""
        MATCH p = (n:{label})-[r:{relation}*1..{max_distance}]-(m)
        WHERE {conditions}
        RETURN n, r, m
        """
        result = await self.execute_query(query, properties)
        return [
            {
                "source": {
                    "id": record["n"].id,
                    "labels": list(record["n"].labels),
                    "properties": dict(record["n"]),
                },
                "relationships": [
                    {"type": rel.type, "properties": dict(rel)} for rel in record["r"]
                ],
                "target": {
                    "id": record["m"].id,
                    "labels": list(record["m"].labels),
                    "properties": dict(record["m"]),
                },
            }
            for record in result
        ]

    async def node_exists(self, label: str, properties: dict) -> bool:
        """检查节点是否存在"""
        conditions = " AND ".join([f"n.{key} = ${key}" for key in properties.keys()])
        query = f"""
        MATCH (n:{label})
        WHERE {conditions}
        RETURN COUNT(n) > 0 AS exists
        """
        result = await self.execute_query(query, properties)
        return result[0]["exists"] if result else False

    async def export_graph(self) -> Dict:
        """导出图谱结构用于前端可视化"""
        query = """
        MATCH (n)-[r]->(m)
        RETURN n, r, m
        """
        result = await self.execute_query(query)
        nodes, edges = {}, []
        for record in result:
            node_a = record["n"]
            node_b = record["m"]
            relationship = record["r"]

            # 节点信息
            nodes[node_a.id] = {**node_a}
            nodes[node_b.id] = {**node_b}

            # 边信息
            edges.append({
                "source": node_a.id,
                "target": node_b.id,
                "relation": relationship.type,
            })

        return {"nodes": nodes, "edges": edges}

    async def delete_node(self, label: str, properties: dict):
        """删除节点及其关联关系"""
        conditions = " AND ".join([f"n.{key} = ${key}" for key in properties.keys()])
        query = f"""
        MATCH (n:{label})
        WHERE {conditions}
        DETACH DELETE n
        """
        await self.execute_query(query, properties)

    async def delete_relationship(self, source_label: str, target_label: str, relation: str):
        """删除特定关系"""
        query = f"""
        MATCH (a:{source_label})-[r:{relation}]->(b:{target_label})
        DELETE r
        """
        await self.execute_query(query)

    async def delete_all_nodes(self):
        """删除所有节点及其关联关系"""
        query = "MATCH (n) DETACH DELETE n"
        await self.execute_query(query)

# 测试代码
if __name__ == '__main__':
    async def test_neo4j():
        pool = Neo4jConnectionPool(uri="bolt://localhost:7687", username="neo4j", password="qazwsx123!", max_size=5)
        await pool.init_pool()

        manager = Neo4jManager(connection_pool=pool)

        await manager.delete_all_nodes()
        await manager.create_node("Person", {"name": "Alice"})
        await manager.create_node("Person", {"name": "Bob"})
        await manager.create_relationship(
            "Person", {"name": "Alice"}, "Person", {"name": "Bob"}, "KNOWS", {"created_at": "2025-01-01"}
        )
        print(await manager.get_all_nodes())
        print(await manager.get_relationships("Person", "Person", "KNOWS"))

        await pool.close() # 关闭连接池，而正式部署：连接池会在整个服务生命周期中一直存在并管理连接

    asyncio.run(test_neo4j())