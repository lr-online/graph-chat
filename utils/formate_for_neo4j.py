import re


def sanitize_relation(relation: str) -> str:
    """清理关系类型，确保符合 Neo4j 的语法规则"""
    if not relation:
        raise ValueError("关系类型不能为空")

    # 替换非法字符为下划线，保留中文、英文字母、数字和下划线
    sanitized = re.sub(r"[^\w\u4e00-\u9fff]", "_", relation.strip())

    # 如果首字符不是字母或中文，添加前缀
    if not re.match(r"^[A-Za-z\u4e00-\u9fff]", sanitized):
        sanitized = f"R_{sanitized}"

    return sanitized
