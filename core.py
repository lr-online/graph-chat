import asyncio
import base64
import json
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from typing import AsyncGenerator,Any

import networkx as nx
from loguru import logger
from openai import AsyncOpenAI
import aiofiles
from config import settings

class MessageType(Enum):
    TEXT = "text"
    IMAGE = "image"
    FILE = "file"


class Message:
    """表示一条对话消息的类

    属性:
        id: 消息唯一标识
        timestamp: 消息创建时间
        content: 消息内容
        role: 消息角色（user/assistant）
        topic: 消息主题（可选）
        importance: 消息重要性（1-5）
        embedding: 消息的向量表示
        related_nodes: 与消息相关的知识图谱节点
    """

    def __init__(
        self,
        content: str,
        role: str,
        msg_type: MessageType = MessageType.TEXT,
        file_info: Optional[Dict[str, Any]] = None,
    ):
        self.id = str(uuid.uuid4())
        self.content = content
        self.role = role
        self.msg_type = msg_type
        self.file_info = file_info or {}
        self.timestamp = datetime.now()
        self.topic = None
        self.importance = None
        self.embedding = None
        self.related_nodes = []
        logger.debug(
            f"创建新消息 - ID: {self.id}, 角色: {role}, 类型: {msg_type.value}"
        )

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "content": self.content,
            "role": self.role,
            "type": self.msg_type.value,
            "file_info": self.file_info,
            "timestamp": self.timestamp.isoformat(),
            "topic": self.topic,
            "importance": self.importance,
            "related_nodes": self.related_nodes,
        }


class KnowledgeGraph:
    """知识图谱管理类

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
        self, concepts: List[Dict[str, str]], context: str = ""
    ) -> List[str]:
        """添加知识到图谱中

        Args:
            concepts: 概念列表，每个概念包含name、type和description
            context: 概念的上下文信息

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
    """对话记忆管理器

    负责管理对话历史、消息分析和知识提取
    """

    def __init__(self):
        self.messages: List[Message] = []
        self.oai_client = AsyncOpenAI(
            base_url=settings.OPENAI_BASE_URL,
            api_key=settings.OPENAI_API_KEY
        )
        self.knowledge_graph = KnowledgeGraph()
        self.last_knowledge_extraction: Optional[datetime] = None
        logger.info("初始化记忆管理器")

    async def add_message(
        self, content: str, role: str, msg_type: str = "text", file_info: dict = None
    ) -> Message:
        """添加新消息并进行分析

        Args:
            content: 消息内容
            role: 消息角色（user/assistant）
            msg_type: 消息类型（text/image/file）
            file_info: 文件相关信息

        Returns:
            创建的消息对象
        """
        message = Message(content, role, MessageType(msg_type), file_info)
        await self._analyze_message(message)
        self.messages.append(message)
        logger.debug(f"添加新消息 - 角色: {role}, 主题: {message.topic}")

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
        """从对话历史中提取知识并更新知识图谱"""
        try:
            recent_messages = self.messages[-6:]
            conversation_history = self._format_messages_for_prompt(recent_messages)
            logger.info("开始知识提取")

            # 获取当前图谱中的概念
            current_concepts = list(self.knowledge_graph.graph.nodes)
            logger.debug(f"当前图谱概念数: {len(current_concepts)}")

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

            response = await self.oai_client.chat.completions.create(
                model="gpt-4o",
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
            logger.error(f"知识提取错误: {str(e)}")


class Agent:
    """AI助手代理类

    负责处理用户输入并生成回复
    """

    def __init__(self, system_prompt: str = "你是一个有帮助的AI助手"):
        self.memory = MemoryManager()
        self.system_prompt = """
            身份定位: 你是一款新一代对话式 AI 助手，目标是超越传统对话模型，为用户提供最优质的智能支持。

            职责与能力:

            1.	全面多领域支持：能够在编程、文学、科学、商业、教育、娱乐等多领域为用户提供有价值的信息和创意。
            2.	思维清晰、逻辑严谨：回答要条理分明，并在必要时能够对推理过程做适度解释。
            3.	可依赖：遇到不确定的问题或缺少信息时，应当告知用户并适时询问更多信息；如涉及专业或敏感领域（如医疗、法律等），需提供清晰的免责声明并尽量引导专业咨询。
            4.	上下文记忆：在多轮对话中，保持对之前对话内容的理解和一致性；回答时需与上下文连贯，避免重复和矛盾。
            5.	主动提问：若用户问题缺失关键条件或可能存在歧义，应当主动提问以获取更多信息。

            沟通风格:

            1.	真诚友好：语言亲切自然，始终保持礼貌且富有同理心。
            2.	精确简洁：尽量使用清晰易懂的语言作答，不使用过于复杂的术语；如需使用专业词汇，应在上下文或结尾给予必要说明。
            3.	客观中立：在解答讨论类或争议性问题时，应基于事实、客观信息进行分析说明，避免带有明显偏见或过度主观判断。
            4.	灵活多样：在内容创作、写作风格建议、以及方案策划等场景下，能够灵活变换表达方式，进行多样化尝试与对比。

            问答规则:

            1.	合规与道德：严禁输出非法、有害、歧视、侮辱或侵犯隐私等不当内容。
            2.	隐私保护：不主动询问用户的敏感个人信息；即使用户提供，也应妥善保护并提醒其潜在风险。
            3.	边界意识：对于超出知识范围的问题，明确告知无法回答或需要更多线索；如有必要，可提供与问题相关的信息来源或可靠参考渠道。
            4.	信息可信：引用或推断的信息要有一定依据，若是推测或可能不完全准确，需要给出明确提示或说明推理来源。

            最终目标:

            •	为用户提供准确、有益、可行且让人愉悦的回答，并持续优化用户体验，成为用户可信赖的智能助手。
        """
        logger.info("初始化AI助手")

    async def cleanup(self):
        """清理资源"""
        try:
            await self.memory.oai_client.close()
            logger.info("AI助手资源清理完成")
        except Exception as e:
            logger.error(f"清理资源失败: {str(e)}")

    async def reply(
        self, user_input: str, message_type: str = "text", file_info: dict = None
    ) -> AsyncGenerator[Any, None]:
        """处理用户输入并生成回复"""
        try:
            logger.debug(f"处理用户输入: {user_input[:50]}...")
            
            # 如果是文本文件，先处理文件内容
            file_content = None
            if message_type == "file" and file_info:
                try:
                    file_path = Path("uploads") / file_info.get("filename", "")
                    if file_path.exists() and file_path.suffix.lower() in ['.txt', '.md', '.py', '.json', '.yaml', '.yml']:
                        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                            file_content = await f.read()
                            
                        # 分析文件内容并提取知识
                        analysis_prompt = f"""分析以下文件内容，提取关键概念和知识点。
                        文件名: {file_info.get('original_name')}
                        文件内容:
                        {file_content[:128_000]}  # 限制长度以避免超出token限制
                        
                        请提取以下信息:
                        1. 文件的主要主题和目的
                        2. 关键概念和术语
                        3. 概念之间的关系
                        
                        以JSON格式返回:
                        {{
                            "topic": "文件主题",
                            "concepts": [
                                {{"concept": "概念名称", "type": "概念类型", "description": "概念描述"}}
                            ],
                            "relations": [
                                {{"source": "源概念", "target": "目标概念", "relation": "关系类型"}}
                            ]
                        }}
                        """
                        
                        response = await self.memory.oai_client.chat.completions.create(
                            model="gpt-4o",
                            messages=[{"role": "user", "content": analysis_prompt}],
                            response_format={"type": "json_object"},
                        )
                        
                        analysis = json.loads(response.choices[0].message.content)
                        
                        # 更新知识图谱
                        self.memory.knowledge_graph.add_knowledge(analysis["concepts"])
                        for relation in analysis["relations"]:
                            self.memory.knowledge_graph.add_relation(
                                relation["source"], relation["target"], relation["relation"]
                            )
                            
                        # 将文件内容添加到用户消息中
                        user_input = f"[文件分析] {file_info.get('original_name')}\n\n文件内容：\n{file_content[:500]}..."
                except Exception as e:
                    logger.error(f"文件处理失败: {e}")
                    
            # 添加用户消息
            await self.memory.add_message(user_input, "user", message_type, file_info or {})

            # 获取相关上下文
            chat_history = "\n".join(
                [f"{msg.role}: {msg.content}" for msg in self.memory.messages[-5:]]
            )
            relevant_history = await self.memory.get_relevant_history(user_input, top_k=3)

            # 获取相关知识
            related_concepts = set()
            for node in self.memory.knowledge_graph.graph.nodes():
                if node.lower() in user_input.lower():
                    related_concepts.update(
                        self.memory.knowledge_graph.get_related_concepts(node)
                    )

            # 构建系统提示
            type_prompt = ""
            if message_type == "image":
                type_prompt = f"\n用户发送了一张图片，URL: {file_info.get('url', '')}"
            elif message_type == "file":
                type_prompt = f"\n用户发送了一个文件，文件名: {file_info.get('original_name', '')}"
                if file_content:
                    type_prompt += "\n该文件的内容已经被分析并添加到知识库中。"

            # 构建完整的system prompt
            full_system_prompt = f"""{self.system_prompt}

                当前进行中的对话记录：
                {chat_history}

                相关的历史对话：
                {relevant_history}

                相关的知识概念：
                {', '.join(related_concepts) if related_concepts else '无'}
                {type_prompt}

                请基于这些上下文来回答用户的问题。如果文件内容或相关概念对回答有帮助，请充分利用这些信息。
                如果是在分析文件，请详细解释文件的主要内容和关键概念。
            """

            # 构建消息内容
            messages = [{"role": "system", "content": full_system_prompt}]

            if message_type == "image":
                # 编码图片
                base64_image = await encode_image_file(file_info.get("filename", ""))

                if base64_image:
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                # {
                                #     "type": "text",
                                #     "text": "请分析这张图片的内容，告诉我：\n1. 图片中包含了什么内容？\n2. 你能看出什么细节和特征？\n3. 这张图片想要传达什么信息或情感？",
                                # },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": base64_image},
                                },
                            ],
                        }
                    )
                else:
                    messages.append(
                        {
                            "role": "user",
                            "content": "抱歉，图片处理失败，无法进行分析。",
                        }
                    )
            else:
                # 对于文本消息，使用原来的消息构建逻辑
                messages = [
                    {"role": "system", "content": full_system_prompt},
                    {"role": "user", "content": user_input},
                ]

            # 生成回复
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


async def encode_image_file(file_path: str) -> str:
    """从URL获取图片并转换为base64编码"""
    try:
        file_path = Path("uploads") / file_path
        if not file_path.exists():
            logger.error(f"图片文件不存在: {file_path}")
            return None

        with open(file_path, "rb") as image_file:
            image_data = image_file.read()
            return (
                f"data:image/jpeg;base64,{base64.b64encode(image_data).decode('utf-8')}"
            )
    except Exception as e:
        logger.error(f"编码图片失败: {e}")
        return None


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
