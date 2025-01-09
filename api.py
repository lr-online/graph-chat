import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from loguru import logger

from core import Agent

# 配置logger
logger.remove()  # 移除默认的处理器
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """处理应用的生命周期事件"""
    logger.info("服务启动")
    yield
    logger.info("服务关闭")


app = FastAPI(lifespan=lifespan)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def get_home():
    """返回首页HTML"""
    return FileResponse("index.html")


async def on_message(message: str, agent: Agent, websocket: WebSocket):
    """处理消息"""
    async for chunk in agent.reply(message):
        await websocket.send_json({"type": "message_chunk", "data": chunk})
    await websocket.send_json({"type": "message_end", "data": {"debug_info": None}})
    graph_data = agent.memory.knowledge_graph.get_graph_data()
    await websocket.send_json({"type": "graph_update", "data": graph_data})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket连接处理"""
    agent = Agent()
    try:
        await websocket.accept()
        while True:
            message = await websocket.receive_text()
            await on_message(message, agent, websocket)

    except Exception as e:
        logger.error(f"WebSocket错误 error: {e}")
    finally:
        # 清理资源
        logger.info(f"WebSocket连接关闭")
