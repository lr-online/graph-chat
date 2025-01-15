import json
import re
from pathlib import Path
from typing import AsyncGenerator, Any
from typing import Dict, List, Optional, Set

import aiofiles
from loguru import logger

from core import Agent
from core import MemoryManager
from infrastructure.neo4j_manager import Neo4jManager
from knowledge_graph import Neo4jKnowledgeGraph
from utils.formate_for_neo4j import sanitize_relation



class MemoryManagerWithNeo4j(MemoryManager):
    """扩展的记忆管理器，支持使用 Neo4j 知识图谱"""

    def __init__(self, neo4j_manager: Optional[Neo4jManager] = None):
        super().__init__()
        # 动态选择使用 KnowledgeGraph 或 Neo4jKnowledgeGraph
        if neo4j_manager:
            self.knowledge_graph = Neo4jKnowledgeGraph(neo4j_manager)
            logger.info("MemoryManager 使用 Neo4jKnowledgeGraph")
        else:
            logger.warning("Neo4j 未启用，使用本地 KnowledgeGraph")

    async def extract_knowledge(self) -> None:
        """从对话历史中提取知识并更新知识图谱"""
        try:
            # 获取最近的对话消息
            recent_messages = self.messages[-6:]
            conversation_history = self._format_messages_for_prompt(recent_messages)
            logger.info("开始知识提取")

            # 获取当前图谱中的概念
            if isinstance(self.knowledge_graph, Neo4jKnowledgeGraph):
                current_concepts = await self.knowledge_graph.get_graph_data()
                current_concepts = [node["name"] for node in current_concepts["nodes"]]
            else:
                current_concepts = list(self.knowledge_graph.graph.nodes)

            logger.info(f"当前图谱概念数: {len(current_concepts)}")

            # 构建提取提示
            extraction_prompt = f"""分析以下对话，提取关键概念和它们之间的关系。

    当前知识库中已有的概念：
    {', '.join(current_concepts)}

    对话历史：
    {conversation_history}

    请提取以下信息并以JSON格式返回。注意：
    1. 如果发现的概念与已有概念相关，应该更新或完善已有概念的描述
    2. 如果发现新的概念，应该与已有概念建立关联
    3. 如果发现现有概念之间的新关系，应该添加这些关系
    4. 分析要尽可能的详细，不要遗漏任何信息

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

            # 调用 OpenAI 接口
            response = await self.oai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": extraction_prompt}],
                response_format={"type": "json_object"},
            )

            # 解析返回的数据
            try:
                # logger.debug(f"完整的响应: {response}")
                content = response.choices[0].message.content
                # logger.debug(f"返回的内容: {content}")

                knowledge = json.loads(content)
                if not isinstance(knowledge, dict) or "concepts" not in knowledge or "relations" not in knowledge:
                    raise ValueError("知识提取的响应格式不正确")
            except Exception as parse_error:
                logger.error(f"解析知识提取响应时出错: {parse_error}")
                return

            logger.info("开始更新知识图谱")

            # 添加新概念
            # 添加概念和关系
            if isinstance(self.knowledge_graph, Neo4jKnowledgeGraph):
                new_nodes = await self.knowledge_graph.add_knowledge(
                    knowledge.get("concepts", []), knowledge.get("relations", [])
                )
            else:
                new_nodes = self.knowledge_graph.add_knowledge(
                    knowledge.get("concepts", []), knowledge.get("relations", [])
                )
            for log in new_nodes:
                logger.debug(log)

            # 添加新关系
            for relation in knowledge["relations"]:
                # 清理关系类型
                relation_type = sanitize_relation(relation["relation"])

                if isinstance(self.knowledge_graph, Neo4jKnowledgeGraph):
                    log = await self.knowledge_graph.add_relation(
                        relation["source"], relation["target"], relation_type
                    )
                else:
                    log = self.knowledge_graph.add_relation(
                        relation["source"], relation["target"], relation_type
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
                f"知识图谱更新完成 - "
                f"节点数: {len(current_concepts)}, "
                f"关系数: {len(knowledge['relations'])}"
            )

        except Exception as e:
            logger.error(f"知识提取错误: {str(e)}")


class AgentWithNeo4j(Agent):
    """扩展的 Agent 类，支持使用 Neo4j 知识图谱"""

    def __init__(
            self,
            system_prompt: str = "你是一个有帮助的AI助手",
            neo4j_manager: Optional[Neo4jManager] = None,
    ):
        # 调用父类初始化方法
        super().__init__(system_prompt=system_prompt)
        # 动态选择知识图谱
        if neo4j_manager:
            self.memory = MemoryManagerWithNeo4j(neo4j_manager)
            logger.info("Agent 使用 Neo4jKnowledgeGraph")
        else:
            logger.warning("Neo4j 未启用，使用本地 KnowledgeGraph")

    async def reply(
            self, user_input: str, message_type: str = "text", file_info: dict = None
    ) -> AsyncGenerator[Any, None]:
        """处理用户输入并生成回复"""
        try:
            logger.debug(f"处理用户输入: {user_input[:50]}...")

            # 如果是文件，处理文件内容并更新知识图谱
            file_content = None
            if message_type == "file" and file_info:
                try:
                    file_path = Path("uploads") / file_info.get("filename", "")
                    if file_path.exists() and file_path.suffix.lower() in [
                        ".txt",
                        ".md",
                        ".py",
                        ".json",
                        ".yaml",
                        ".yml",
                    ]:
                        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                            file_content = await f.read()

                        analysis_prompt = f"""分析以下文件内容，提取关键概念和知识点。
                        文件名: {file_info.get('original_name')}
                        文件内容:
                        {file_content[:128_000]}

                        请提取以下信息:
                        1. 文件的主要主题和目的
                        2. 关键概念和术语
                        3. 概念之间的关系
                        """
                        response = await self.memory.oai_client.chat.completions.create(
                            model="gpt-4o",
                            messages=[{"role": "user", "content": analysis_prompt}],
                            response_format={"type": "json_object"},
                        )
                        analysis = json.loads(response.choices[0].message.content)

                        # 更新知识图谱
                        if isinstance(self.memory.knowledge_graph, Neo4jKnowledgeGraph):
                            await self.memory.knowledge_graph.add_knowledge(
                                analysis["concepts"]
                            )
                            for relation in analysis["relations"]:
                                await self.memory.knowledge_graph.add_relation(
                                    relation["source"],
                                    relation["target"],
                                    relation["relation"],
                                )
                        else:
                            self.memory.knowledge_graph.add_knowledge(
                                analysis["concepts"]
                            )
                            for relation in analysis["relations"]:
                                self.memory.knowledge_graph.add_relation(
                                    relation["source"],
                                    relation["target"],
                                    relation["relation"],
                                )

                        user_input = f"[文件分析] {file_info.get('original_name')}\n\n文件内容：\n{file_content[:500]}..."
                except Exception as e:
                    logger.error(f"文件处理失败: {e}")

            # 添加用户消息
            await self.memory.add_message(user_input, "user", message_type, file_info or {})

            # 获取相关上下文和知识
            chat_history = "\n".join(
                [f"{msg.role}: {msg.content}" for msg in self.memory.messages[-5:]]
            )
            relevant_history = await self.memory.get_relevant_history(user_input, top_k=3)

            related_concepts = set()
            if isinstance(self.memory.knowledge_graph, Neo4jKnowledgeGraph):
                related_concepts = await self.memory.knowledge_graph.get_related_concepts(
                    user_input
                )
            else:
                for node in self.memory.knowledge_graph.graph.nodes():
                    if node.lower() in user_input.lower():
                        related_concepts.update(
                            self.memory.knowledge_graph.get_related_concepts(node)
                        )

            type_prompt = ""
            if message_type == "image":
                type_prompt = f"\n用户发送了一张图片，URL: {file_info.get('url', '')}"
            elif message_type == "file":
                type_prompt = f"\n用户发送了一个文件，文件名: {file_info.get('original_name', '')}"
                if file_content:
                    type_prompt += "\n该文件的内容已经被分析并添加到知识库中。"

            full_system_prompt = f"""{self.system_prompt}

                当前进行中的对话记录：
                {chat_history}

                相关的历史对话：
                {relevant_history}

                相关的知识概念：
                {', '.join(related_concepts) if related_concepts else '无'}
                {type_prompt}

                请基于这些上下文来回答用户的问题。如果文件内容或相关概念对回答有帮助，请充分利用这些信息。
                """

            messages = [
                {"role": "system", "content": full_system_prompt},
                {"role": "user", "content": user_input},
            ]

            response = await self.memory.oai_client.chat.completions.create(
                model="gpt-4o", messages=messages, max_tokens=1000, stream=True
            )

            logger.debug("开始生成回复")
            llm_response = ""
            async for chunk in response:
                chunk_content = chunk.choices[0].delta.content or ""
                yield chunk_content
                llm_response += chunk_content

            await self.memory.add_message(llm_response, "assistant", "text", {})
            logger.debug("回复生成完成")

        except Exception as e:
            logger.error(f"生成回复时出错: {str(e)}")
            yield "抱歉，处理您的消息时出现错误。"



