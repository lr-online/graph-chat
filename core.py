import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
from loguru import logger
from openai import AsyncOpenAI


class Message:
    def __init__(self, content: str, role: str):
        self.id = f"msg_{int(datetime.now().timestamp())}"
        self.timestamp = datetime.now().isoformat()
        self.content = content
        self.role = role
        self.topic: Optional[str] = None
        self.importance: Optional[int] = None
        self.embedding: Optional[List[float]] = None
        self.related_nodes: List[str] = []

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "content": self.content,
            "role": self.role,
            "topic": self.topic,
            "importance": self.importance,
            "related_nodes": self.related_nodes,
        }


class KnowledgeGraph:
    def __init__(self, save_path: str = "knowledge_graph.json"):
        self.graph = nx.DiGraph()
        self.save_path = save_path

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
        self, concepts: List[Dict[str, str]], context: str = ""
    ) -> List[str]:
        """添加知识到图谱中，返回新添加的节点列表"""
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
                new_nodes.append(
                    f"新概念: {concept_name} ({concept['type']}) - {concept['description']}"
                )
            else:
                # 更新现有概念
                update_log = self.update_concept(
                    concept_name,
                    {"type": concept["type"], "description": concept["description"]},
                )
                new_nodes.append(update_log)

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
        """获取知识图谱数据，用于前端可视化"""
        try:
            nodes = {}
            for node in self.graph.nodes:
                attrs = self.graph.nodes[node]
                nodes[node] = {
                    "type": attrs.get("type", "default"),
                    "description": attrs.get("description", ""),
                    "context": attrs.get("context", ""),
                }

            return {
                "nodes": nodes,
                "edges": [
                    {
                        "source": source,
                        "target": target,
                        "relation": self.graph.edges[source, target]["relation"],
                    }
                    for source, target in self.graph.edges
                ],
            }
        except Exception as e:
            logger.error(f"获取知识图谱数据失败: {e}")
            return {"nodes": {}, "edges": []}


class MemoryManager:
    def __init__(self):
        self.messages: List[Message] = []
        self.oai_client = AsyncOpenAI()
        self.knowledge_graph = KnowledgeGraph("knowledge_graph.json")
        self.last_knowledge_extraction: Optional[datetime] = None

    async def add_message(self, content: str, role: str) -> Message:
        message = Message(content, role)
        await self._analyze_message(message)
        self.messages.append(message)

        await self.extract_knowledge()
        return message

    async def _get_embedding(self, text: str) -> List[float]:
        """获取文本的embedding向量"""
        response = await self.oai_client.embeddings.create(
            model="text-embedding-ada-002", input=text
        )
        return response.data[0].embedding

    def _calculate_similarity(self, v1: List[float], v2: List[float]) -> float:
        """计算两个向量的余弦相似度"""
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm_v1 = sum(x * x for x in v1) ** 0.5
        norm_v2 = sum(x * x for x in v2) ** 0.5
        return dot_product / (norm_v1 * norm_v2)

    async def _analyze_message(self, message: Message):
        """分析消息的主题、重要性和embedding"""
        try:
            window = self._get_analysis_window()
            window.append(message)

            analysis_prompt = f"""分析以下对话的主题和重要性。
对话历史:
{self._format_messages_for_prompt(window)}

请返回JSON格式:
{{
    "topic": "主题名称",
    "importance": 重要性分数(1-5,5最重要)
}}"""

            response = await self.oai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": analysis_prompt}],
                response_format={"type": "json_object"},
            )

            analysis = json.loads(response.choices[0].message.content)
            message.topic = analysis["topic"]
            message.importance = analysis["importance"]
            message.embedding = await self._get_embedding(message.content)
            logger.debug(
                f"消息分析完成 - 主题: {message.topic}, 重要性: {message.importance}"
            )
        except Exception as e:
            logger.error(f"消息分析错误: {e}")
            message.topic = "未分类"
            message.importance = 1
            message.embedding = None

    def _get_analysis_window(self, max_messages: int = 4) -> List[Message]:
        return self.messages[-max_messages:] if len(self.messages) > 0 else []

    def _format_messages_for_prompt(self, messages: List[Message]) -> str:
        return "\n".join([f"{msg.role}: {msg.content}" for msg in messages])

    def get_chat_history(self) -> List[Dict]:
        return [msg.to_dict() for msg in self.messages]

    async def get_relevant_history(self, query: str, top_k: int = 3) -> List[Dict]:
        try:
            query_embedding = await self._get_embedding(query)
            similarities = []
            for msg in self.messages:
                if msg.embedding is not None:
                    similarity = self._calculate_similarity(
                        query_embedding, msg.embedding
                    )
                    similarities.append((msg, similarity))

            similarities.sort(key=lambda x: x[1], reverse=True)
            return [
                {**msg.to_dict(), "similarity": sim}
                for msg, sim in similarities[:top_k]
            ]
        except Exception as e:
            print(f"Error getting relevant history: {e}")
            return []

    async def extract_knowledge(self) -> None:
        """从对话历史中提取知识并构建知识图谱"""
        try:
            recent_messages = self.messages[-5:]
            conversation_history = self._format_messages_for_prompt(recent_messages)

            # 获取当前图谱中的概念
            current_concepts = list(self.knowledge_graph.graph.nodes)

            extraction_prompt = f"""分析以下对话，提取关键概念和它们之间的关系。

当前知识库中已有的概念：
{', '.join(current_concepts)}

对话历史：
{conversation_history}

请提取以下信息并以JSON格式返回。注意：
1. 如果发现的概念与已有概念相关，应该更新或完善已有概念的描述
2. 如果发现新的概念，应该与已有概念建立关联
3. 如果发现现有概念之间的新关系，应该添加这些关系

请返回：
1. concepts: 关键概念列表，每个概念包含：
   - concept: 概念名称
   - type: 概念类型（如：技术、工具、方法等）
   - description: 简短描述
2. relations: 概念之间的关系列表，每个关系包含：
   - source: 源概念
   - target: 目标概念
   - relation: 关系类型（如：是...的一部分、用于、需要等）

格式如下：
{{
    "concepts": [
        {{"concept": "Python", "type": "编程语言", "description": "一种高级编程语言"}}
    ],
    "relations": [
        {{"source": "Python", "target": "数据分析", "relation": "用于"}}
    ]
}}"""

            response = await self.oai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": extraction_prompt}],
                response_format={"type": "json_object"},
            )

            knowledge = json.loads(response.choices[0].message.content)

            logger.info("开始更新知识图谱")

            # 添加新概念
            new_nodes = self.knowledge_graph.add_knowledge(knowledge["concepts"])
            for log in new_nodes:
                logger.debug(log)

            # 添加新关系
            for relation in knowledge["relations"]:
                log = self.knowledge_graph.add_relation(
                    relation["source"], relation["target"], relation["relation"]
                )
                if log:
                    logger.debug(log)

            # 更新消息的相关节点
            for msg in recent_messages:
                msg.related_nodes = []
                for concept in knowledge["concepts"]:
                    if concept["concept"].lower() in msg.content.lower():
                        msg.related_nodes.append(concept["concept"])

            logger.info(
                f"知识图谱更新完成 - 节点数: {self.knowledge_graph.graph.number_of_nodes()}, 边数: {self.knowledge_graph.graph.number_of_edges()}"
            )

        except Exception as e:
            logger.error(f"知识提取错误: {e}")


class Agent:
    def __init__(self, system_prompt: str = "你是一个有帮助的AI助手"):
        self.memory = MemoryManager()
        self.system_prompt = system_prompt

    async def reply(self, user_input: str):
        """处理用户输入并生成回复"""
        # 记录用户消息
        user_message = await self.memory.add_message(user_input, "user")

        # 获取相关上下文
        relevant_history = await self.memory.get_relevant_history(user_input, top_k=3)
        relevant_context = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in relevant_history]
        )

        # 获取相关知识
        related_concepts = set()
        for node in self.memory.knowledge_graph.graph.nodes():
            if node.lower() in user_input.lower():
                related_concepts.update(
                    self.memory.knowledge_graph.get_related_concepts(node)
                )

        # 构建完整的system prompt
        full_system_prompt = f"""{self.system_prompt}

            相关的历史对话：
            {relevant_context}

            相关的知识概念：
            {', '.join(related_concepts) if related_concepts else '无'}

            请基于这些上下文来回答用户的问题。如果历史信息或相关概念对回答有帮助，可以参考它们，但不要直接重复这些内容。
        """

        # 生成回复
        messages = [
            {"role": "system", "content": full_system_prompt},
            {"role": "user", "content": user_input},
        ]

        response = await self.memory.oai_client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, stream=True
        )
        llm_response = ""
        async for chunk in response:
            chunk_content = chunk.choices[0].delta.content or ""
            yield chunk_content
            llm_response += chunk_content

        # 记录助手消息
        assistant_message = await self.memory.add_message(llm_response, "assistant")


if __name__ == "__main__":
    logger.info("启动智能对话系统")
    agent = Agent()

    while True:
        user_input = input("\n用户: ")
        if user_input.lower() == "quit":
            break

        # 获取助手回复
        response, debug_info = agent.reply(user_input)

        # 打印回复和调试信息
        print(f"\n助手: {response}")
        print(f"\n[Debug] 消息分析:")
        print(
            f"用户消息 - 主题: {debug_info['user_message']['topic']}, 重要性: {debug_info['user_message']['importance']}"
        )
        print(
            f"助手消息 - 主题: {debug_info['assistant_message']['topic']}, 重要性: {debug_info['assistant_message']['importance']}"
        )
        if debug_info["relevant_concepts"]:
            print(f"相关概念: {', '.join(debug_info['relevant_concepts'])}")

        # 显示当前对话历史
        print("\n[Debug] 当前对话历史:")
        for msg in agent.memory.get_chat_history():
            print(
                f"{msg['timestamp']} [{msg['topic']}] ({msg['importance']}分) - {msg['role']}: {msg['content'][:50]}..."
            )
