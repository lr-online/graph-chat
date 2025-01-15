import asyncio
import uuid
from typing import List, Dict, Optional

from loguru import logger
from neo4j import AsyncGraphDatabase


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
            logger.debug(f"执行查询: {query} 参数: {parameters}")
            result = await session.run(query, parameters or {})
            data = []
            async for record in result:
                data.append(record.data())
            return data

    async def create_node(self, label: str, properties: Dict[str, str]):
        """创建节点并为其分配 UUID"""
        properties["uuid"] = str(uuid.uuid4())  # 为节点生成唯一的 UUID
        query = f"CREATE (n:{label} $props) RETURN n"
        return await self.execute_query(query, {"props": properties})

    async def create_relationship(self, source: str, target: str, relation: str, props: dict = None):
        """在两个节点之间创建关系"""
        props = props or {}
        query = """
        MATCH (a:Concept {name: $source, uuid: $source_uuid}), 
              (b:Concept {name: $target, uuid: $target_uuid})
        MERGE (a)-[r:RELATION_TYPE]->(b)
        SET r += $props
        RETURN r
        """
        query = query.replace("RELATION_TYPE", relation)
        params = {
            "source": source,
            "source_uuid": props.get("source_uuid"),
            "target": target,
            "target_uuid": props.get("target_uuid"),
            "props": props,
        }

        logger.debug(f"Creating relationship with query: {query}, params: {params}")
        await self.execute_query(query, params)

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
            self, source_properties: dict, target_properties: dict, relation: str, new_properties: dict
    ):
        """更新关系属性"""
        source_conditions = " AND ".join([f"a.{key} = ${key}" for key in source_properties.keys()])
        target_conditions = " AND ".join([f"b.{key} = ${key}" for key in target_properties.keys()])
        updates = ", ".join([f"r.{key} = ${key}" for key in new_properties.keys()])

        query = f"""
        MATCH (a)-[r:{relation}]->(b)
        WHERE {source_conditions} AND {target_conditions}
        SET {updates}
        RETURN r
        """
        parameters = {**source_properties, **target_properties, **new_properties}
        return await self.execute_query(query, parameters)

    async def get_all_nodes(self, label: Optional[str] = None) -> List[Dict]:
        """获取所有节点"""
        query = f"MATCH (n{':' + label if label else ''}) RETURN n.uuid AS uuid, labels(n) AS labels, n AS properties"
        result = await self.execute_query(query)
        return [
            {
                "uuid": record["uuid"],
                "labels": record["labels"],
                "properties": record["properties"],
            }
            for record in result
        ]

    async def get_relationships(self, source_properties: dict, target_properties: dict, relation: str) -> List[Dict]:
        """获取特定关系及其关联节点"""
        source_conditions = " AND ".join([f"a.{key} = ${key}" for key in source_properties.keys()])
        target_conditions = " AND ".join([f"b.{key} = ${key}" for key in target_properties.keys()])
        query = f"""
        MATCH (a)-[r:{relation}]->(b)
        WHERE {source_conditions} AND {target_conditions}
        RETURN a.name AS source, b.name AS target, r.type AS relation
        """
        parameters = {**source_properties, **target_properties}
        result = await self.execute_query(query, parameters)
        return [
            {
                "source": record["source"],
                "target": record["target"],
                "relation": record["relation"]
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

    async def delete_node(self, label: str, properties: dict):
        """删除节点及其关联关系"""
        conditions = " AND ".join([f"n.{key} = ${key}" for key in properties.keys()])
        query = f"""
        MATCH (n:{label})
        WHERE {conditions}
        DETACH DELETE n
        """
        await self.execute_query(query, properties)

    async def delete_relationship(self, source_properties: dict, target_properties: dict, relation: str):
        """删除特定关系"""
        source_conditions = " AND ".join([f"a.{key} = ${key}" for key in source_properties.keys()])
        target_conditions = " AND ".join([f"b.{key} = ${key}" for key in target_properties.keys()])
        query = f"""
        MATCH (a)-[r:{relation}]->(b)
        WHERE {source_conditions} AND {target_conditions}
        DELETE r
        """
        parameters = {**source_properties, **target_properties}
        await self.execute_query(query, parameters)

    async def delete_all_nodes(self):
        """删除所有节点及其关联关系"""
        query = "MATCH (n) DETACH DELETE n"
        await self.execute_query(query)


# 测试代码
if __name__ == '__main__':
    async def test_neo4j():
        # 初始化连接池
        pool = Neo4jConnectionPool(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="qazwsx123!",
            max_size=5
        )
        await pool.init_pool()

        # 初始化 Neo4j 管理器
        manager = Neo4jManager(connection_pool=pool)

        try:
            print("========== 清空数据库 ==========")
            await manager.delete_all_nodes()

            print("========== 创建节点 ==========")
            alice_properties = {"name": "Alice"}
            bob_properties = {"name": "Bob"}

            # 创建节点
            await manager.create_node("Concept", alice_properties)
            print(f"创建节点: {alice_properties}")
            await manager.create_node("Concept", bob_properties)
            print(f"创建节点: {bob_properties}")

            print("\n========== 获取所有节点 ==========")
            nodes = await manager.get_all_nodes(label="Concept")
            for node in nodes:
                print(node)



        except Exception as e:
            print(f"Error during Neo4j operations: {e}")
        finally:
            print("\n========== 关闭连接池 ==========")
            await pool.close()

    asyncio.run(test_neo4j())
