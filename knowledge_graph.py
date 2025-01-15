from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Set

import networkx as nx
from loguru import logger

from infrastructure.neo4j_manager import Neo4jManager
from utils.formate_for_neo4j import sanitize_relation
from utils.sync_to_async import sync_or_async


class BaseKnowledgeGraph(ABC):
    """知识图谱抽象基类"""

    @sync_or_async
    @abstractmethod
    def update_concept(self, concept_name: str, new_attrs: Dict[str, str]) -> str:
        pass

    @sync_or_async
    @abstractmethod
    def delete_concept(self, concept_name: str) -> str:
        pass

    @sync_or_async
    @abstractmethod
    def update_relation(self, source: str, target: str, new_relation: str) -> str:
        pass

    @sync_or_async
    @abstractmethod
    def delete_relation(self, source: str, target: str) -> str:
        pass

    @sync_or_async
    @abstractmethod
    def add_knowledge(
            self, concepts: List[Dict[str, str]], relations: List[Dict[str, str]] = None, context: str = ""
    ) -> List[str]:
        pass

    @sync_or_async
    @abstractmethod
    def get_related_concepts(self, concept: str, max_distance: int = 2) -> Set[str]:
        pass

    @sync_or_async
    @abstractmethod
    def get_concept_info(self, concept: str) -> Optional[Dict]:
        pass

    @sync_or_async
    @abstractmethod
    def get_graph_data(self) -> dict:
        pass


class MemoryKnowledgeGraph(BaseKnowledgeGraph):
    """
    知识图谱管理类
    负责管理概念节点和关系，支持增删改查操作
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        logger.info("初始化知识图谱")

    def update_concept(self, concept_name: str, new_attrs: Dict[str, str]) -> str:
        """更新概念的属性"""
        if concept_name not in self.graph:
            return f"概念不存在: {concept_name}"

        old_attrs = dict(self.graph.nodes[concept_name])
        self.graph.nodes[concept_name].update(new_attrs)

        log = f"更新概念 {concept_name}:\n"
        for key, new_value in new_attrs.items():
            old_value = old_attrs.get(key, "无")
            if old_value != new_value:
                log += f"- {key}: {old_value} -> {new_value}\n"

        return log

    def delete_concept(self, concept_name: str) -> str:
        """删除概念及其相关的边"""
        if concept_name not in self.graph:
            return f"概念不存在: {concept_name}"

        # 获取相关的边用于日志
        related_edges = [
            (s, t, self.graph.edges[s, t]["relation"])
            for s, t in self.graph.edges
            if s == concept_name or t == concept_name
        ]

        # 删除节点（会自动删除相关的边）
        self.graph.remove_node(concept_name)

        # 生成日志
        log = f"删除概念: {concept_name}\n"
        log += "删除的关系:\n"
        for source, target, relation in related_edges:
            log += f"- {source} --({relation})--> {target}\n"

        return log

    def update_relation(self, source: str, target: str, new_relation: str) -> str:
        """更新两个概念之间的关系"""
        if not self.graph.has_edge(source, target):
            return f"关系不存在: {source} -> {target}"

        old_relation = self.graph.edges[source, target]["relation"]
        self.graph.edges[source, target]["relation"] = new_relation

        log = f"更新关系: {source} -> {target}\n"
        log += f"- 旧关系: {old_relation}\n"
        log += f"- 新关系: {new_relation}"

        return log

    def delete_relation(self, source: str, target: str) -> str:
        """删除两个概念之间的关系"""
        if not self.graph.has_edge(source, target):
            return f"关系不存在: {source} -> {target}"

        relation = self.graph.edges[source, target]["relation"]
        self.graph.remove_edge(source, target)

        log = f"删除关系: {source} --({relation})--> {target}"

        return log

    def add_knowledge(
            self, concepts: List[Dict[str, str]], relations: List[Dict[str, str]] = None, context: str = ""
    ) -> List[str]:
        """添加知识到图谱中

        Args:
            concepts: 概念列表，每个概念包含name、type和description
            context: 概念的上下文信息
            relations: 关系列表，每个关系包含source、target和relation
        Returns:
            新添加节点的日志列表
        """
        logger.info(f"添加新知识 - 概念数量: {len(concepts)}")
        new_nodes = []
        for concept in concepts:
            concept_name = concept["concept"]
            if concept_name not in self.graph:
                # 添加新概念
                self.graph.add_node(
                    concept_name,
                    type=concept["type"],
                    description=concept["description"],
                    context=context,
                )
                log = f"新概念: {concept_name} ({concept['type']}) - {concept['description']}"
                new_nodes.append(log)
                logger.debug(log)
            else:
                # 更新现有概念
                update_log = self.update_concept(
                    concept_name,
                    {"type": concept["type"], "description": concept["description"]},
                )
                new_nodes.append(update_log)
                logger.debug(update_log)

        return new_nodes

    def add_relation(self, source: str, target: str, relation: str) -> Optional[str]:
        """添加概念之间的关系，返回新添加的关系描述"""
        # 检查概念是否存在
        if source not in self.graph or target not in self.graph:
            return f"错误: 概念不存在 ({source} 或 {target})"

        if not self.graph.has_edge(source, target):
            self.graph.add_edge(source, target, relation=relation)
            return f"新关系: {source} --({relation})--> {target}"
        else:
            # 如果关系已存在但不同，则更新
            old_relation = self.graph.edges[source, target]["relation"]
            if old_relation != relation:
                return self.update_relation(source, target, relation)
        return None

    def get_related_concepts(self, concept: str, max_distance: int = 2) -> Set[str]:
        """获取与给定概念相关的概念"""
        if concept not in self.graph:
            return set()
        related = set()
        for node in nx.single_source_shortest_path_length(
                self.graph, concept, cutoff=max_distance
        ):
            related.add(node)
        return related

    def get_concept_info(self, concept: str) -> Optional[Dict]:
        """获取概念的详细信息"""
        if concept not in self.graph:
            return None
        return dict(self.graph.nodes[concept])

    def get_graph_data(self) -> dict:
        """获取图谱的节点和边数据"""
        try:
            nodes = []
            edges = []

            # 获取节点信息
            for node, attrs in self.graph.nodes(data=True):
                nodes.append({
                    "name": node,
                    "type": attrs.get("type", "default"),
                    "description": attrs.get("description", ""),
                    "context": attrs.get("context", ""),
                })

            # 获取边信息
            for source, target, attrs in self.graph.edges(data=True):
                edges.append({
                    "source": source,
                    "target": target,
                    "relation": attrs.get("relation", ""),
                })

            return {
                "nodes": nodes,
                "edges": edges,
            }
        except Exception as e:
            logger.error(f"获取内存知识图谱数据失败: {e}")
            return {"nodes": [], "edges": []}


class Neo4jKnowledgeGraph(BaseKnowledgeGraph):
    """基于 Neo4j 的知识图谱管理类"""

    def __init__(self, neo4j_manager: Neo4jManager):
        self.neo4j_manager = neo4j_manager

        logger.info("初始化 Neo4j 知识图谱")

    async def update_concept(self, concept_name: str, new_attrs: Dict[str, str]) -> str:
        """更新概念的属性"""
        exists = await self.neo4j_manager.node_exists("Concept", {"name": concept_name})
        if not exists:
            return f"概念不存在: {concept_name}"

        await self.neo4j_manager.update_node_properties("Concept", {"name": concept_name}, new_attrs)

        log = f"更新概念 {concept_name}:\n"
        for key, value in new_attrs.items():
            log += f"- {key}: {value}\n"
        return log

    async def delete_concept(self, concept_name: str) -> str:
        """删除概念及其相关的边"""
        exists = await self.neo4j_manager.node_exists("Concept", {"name": concept_name})
        if not exists:
            return f"概念不存在: {concept_name}"

        await self.neo4j_manager.delete_node("Concept", {"name": concept_name})
        return f"删除概念: {concept_name}"

    async def update_relation(self, source: str, target: str, new_relation: str) -> str:
        """更新两个概念之间的关系"""
        exists = await self.neo4j_manager.get_relationships(
            {"name": source}, {"name": target}, "RELATED"
        )
        if not exists:
            return f"关系不存在: {source} -> {target}"

        await self.neo4j_manager.update_relationship_properties(
            {"name": source}, {"name": target}, "RELATED", {"type": new_relation}
        )
        return f"更新关系: {source} -> {target}\n- 新关系: {new_relation}"

    async def delete_relation(self, source: str, target: str) -> str:
        """删除两个概念之间的关系"""
        exists = await self.neo4j_manager.get_relationships(
            {"name": source}, {"name": target}, "RELATED"
        )
        if not exists:
            return f"关系不存在: {source} -> {target}"

        await self.neo4j_manager.delete_relationship({"name": source}, {"name": target}, "RELATED")
        return f"删除关系: {source} -> {target}"

    async def add_knowledge(self, concepts: List[Dict[str, str]], relations: List[Dict[str, str]] = None,
                            context: str = "") -> List[str]:
        """添加知识到图谱中，包括节点和关系"""
        logs = []

        # 添加概念节点
        for concept in concepts:
            exists = await self.neo4j_manager.node_exists("Concept", {"name": concept["concept"]})
            if not exists:
                await self.neo4j_manager.create_node(
                    "Concept",
                    {
                        "name": concept["concept"],
                        "type": concept["type"],
                        "description": concept["description"],
                        "context": context,
                    },
                )
                log = f"新概念: {concept['concept']} ({concept['type']}) - {concept['description']}"
                logs.append(log)
            else:
                update_log = await self.update_concept(
                    concept["concept"],
                    {"type": concept["type"], "description": concept["description"]},
                )
                logs.append(update_log)

        # 添加关系
        if relations:
            for relation in relations:
                relation_type = sanitize_relation(relation["relation"])
                log = await self.add_relation(
                    relation["source"], relation["target"], relation_type
                )
                if log:
                    logs.append(log)

        return logs

    async def add_relation(
            self, source: str, target: str, relation: str, properties: Dict[str, str] = None
    ) -> Optional[str]:
        """添加概念之间的关系，支持属性更新"""
        properties = properties or {}
        sanitized_relation = sanitize_relation(relation)

        # 检查源和目标节点是否存在
        source_exists = await self.neo4j_manager.node_exists("Concept", {"name": source})
        target_exists = await self.neo4j_manager.node_exists("Concept", {"name": target})
        if not (source_exists and target_exists):
            return f"错误: 概念不存在 ({source} 或 {target})"

        # 检查关系是否存在
        relationships = await self.neo4j_manager.get_relationships(
            {"name": source}, {"name": target}, sanitized_relation
        )
        if relationships:
            # 如果关系已存在，则更新其属性
            if properties:
                updates = ", ".join([f"r.{key} = ${key}" for key in properties.keys()])
                query = f"""
                    MATCH (a:Concept)-[r:{sanitized_relation}]->(b:Concept)
                    WHERE a.name = $source AND b.name = $target
                    SET {updates}
                    RETURN r
                    """
                parameters = {**properties, "source": source, "target": target}
                await self.neo4j_manager.execute_query(query, parameters)
                return f"更新关系: {source} --({sanitized_relation})--> {target}"
            else:
                return f"关系已存在，无需更新: {source} --({sanitized_relation})--> {target}"
        else:
            # 如果关系不存在，创建新关系
            query = f"""
                MATCH (a:Concept {{name: $source}}), (b:Concept {{name: $target}})
                MERGE (a)-[r:{sanitized_relation}]->(b)
                SET r += $props
                RETURN r
                """
            parameters = {"source": source, "target": target, "props": properties}
            await self.neo4j_manager.execute_query(query, parameters)
            return f"新关系: {source} --({sanitized_relation})--> {target}"

    async def get_related_concepts(self, concept: str, max_distance: int = 2) -> Set[str]:
        """获取与给定概念相关的概念"""
        related_concepts = set()
        query = f"""
            MATCH p = (n:Concept)-[*1..{max_distance}]-(m:Concept)
            WHERE n.name = $name
            RETURN DISTINCT m.name AS name
            """
        result = await self.neo4j_manager.execute_query(query, {"name": concept})
        for record in result:
            related_concepts.add(record["name"])
        return related_concepts

    async def get_concept_info(self, concept: str) -> Optional[Dict]:
        """获取概念的详细信息"""
        query = "MATCH (n:Concept {name: $name}) RETURN n"
        result = await self.neo4j_manager.execute_query(query, {"name": concept})
        return result[0]["n"] if result else None

    async def get_graph_data(self) -> dict:
        """获取知识图谱数据"""
        try:
            # 查询所有节点和关系
            query = """
                MATCH (n:Concept)-[r]->(m:Concept)
                RETURN n.name AS source, m.name AS target, r.type AS relation, 
                       properties(n) AS source_properties, properties(m) AS target_properties
                """
            result = await self.neo4j_manager.execute_query(query)

            # 构造节点和边的集合
            nodes = {}
            edges = []

            for record in result:
                # 添加节点
                source_name = record["source"]
                target_name = record["target"]

                if source_name not in nodes:
                    nodes[source_name] = {
                        "name": source_name,
                        "type": record["source_properties"].get("type", "default"),
                        "description": record["source_properties"].get("description", ""),
                        "context": record["source_properties"].get("context", ""),
                    }

                if target_name not in nodes:
                    nodes[target_name] = {
                        "name": target_name,
                        "type": record["target_properties"].get("type", "default"),
                        "description": record["target_properties"].get("description", ""),
                        "context": record["target_properties"].get("context", ""),
                    }

                # 添加边
                edges.append(
                    {
                        "source": source_name,
                        "target": target_name,
                        "relation": record["relation"],
                    }
                )

            # 将 nodes 转换为列表格式
            return {
                "nodes": list(nodes.values()),
                "edges": edges,
            }

        except Exception as e:
            logger.error(f"获取知识图谱数据失败: {e}")
            return {"nodes": [], "edges": []}

    async def get_graph_data_front(self) -> dict:
        """获取知识图谱数据，用于前端可视化"""
        try:
            # 查询所有节点和关系
            query = """
                MATCH (n:Concept)-[r]->(m:Concept)
                RETURN n.name AS source, m.name AS target, type(r) AS relation, 
                       properties(n) AS source_properties, properties(m) AS target_properties
                """
            result = await self.neo4j_manager.execute_query(query)

            # 构造节点和边的集合
            nodes = {}
            edges = []

            for record in result:
                # 添加节点
                source_name = record["source"]
                target_name = record["target"]

                # 确保节点存储为字典形式，key 为节点 id
                if source_name not in nodes:
                    nodes[source_name] = {
                        "type": record["source_properties"].get("type", "default"),
                        "description": record["source_properties"].get("description", ""),
                        "context": record["source_properties"].get("context", ""),
                    }

                if target_name not in nodes:
                    nodes[target_name] = {
                        "type": record["target_properties"].get("type", "default"),
                        "description": record["target_properties"].get("description", ""),
                        "context": record["target_properties"].get("context", ""),
                    }

                # 添加边
                edges.append(
                    {
                        "source": source_name,
                        "target": target_name,
                        "relation": record["relation"],
                    }
                )

            # 去重边
            # edges = [dict(t) for t in {tuple(edge.items()) for edge in edges}]
            edges = [dict(edge) for edge in {frozenset(edge.items()) for edge in edges}]
            logger.debug(f"Nodes: {nodes}")
            logger.debug(f"Edges: {edges}")

            r = {
                "nodes": nodes,  # 保持节点为字典格式
                "edges": edges,
            }

            return r
        except Exception as e:
            logger.error(f"获取知识图谱数据失败: {e}")
            return {"nodes": {}, "edges": []}
