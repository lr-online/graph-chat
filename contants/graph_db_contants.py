from enum import Enum


class KnowledgeGraphType(Enum):
    MEMORY = "memory"
    NEO4J = "neo4j"
    # 可扩展支持更多后端类型
    # MONGODB = "mongodb"
    # SQL = "sql"

    @classmethod
    def list(cls):
        """列出所有支持的图后端类型"""
        return [backend.value for backend in cls]

    @classmethod
    def validate(cls, backend: str):
        """验证给定的后端类型是否有效，不区分大小写"""
        if backend.lower() in (b.value for b in cls):
            return cls(backend.lower())
        raise ValueError(f"Invalid knowledge graph type: {backend}. Supported values are: {', '.join(cls.list())}")