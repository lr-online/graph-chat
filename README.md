# Graph-Chat

Graph-Chat是一个基于知识图谱的智能对话系统，能够在对话过程中自动构建和更新知识图谱，实现知识的可视化展示和关联分析。

## 主要特性

### 1. 智能对话
- 基于OpenAI API的智能对话能力
- 支持Markdown格式的富文本显示
- 实时打字机效果的回复展示
- 代码高亮显示
- 支持Mermaid图表渲染

### 2. 知识图谱
- 实时提取对话中的知识点
- 自动构建概念之间的关系
- 交互式知识图谱可视化
- 支持节点拖拽和缩放
- 不同类型概念的差异化展示

### 3. 实时通信
- 基于WebSocket的实时通信
- 自动重连机制
- 连接状态实时显示
- 消息分块传输

## 技术架构

### 后端
- Python 3.12+
- FastAPI - Web框架
- NetworkX - 图数据结构处理
- OpenAI API - AI对话能力
- WebSocket - 实时通信
- Loguru - 日志管理

### 前端
- HTML5 + CSS3
- JavaScript (原生)
- vis-network - 知识图谱可视化
- marked.js - Markdown渲染
- highlight.js - 代码高亮
- mermaid - 图表渲染

## 快速开始

1. 克隆项目
2. 安装依赖
3. 运行项目
