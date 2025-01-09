import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from loguru import logger
import secrets

from core import Agent

# 初始化Basic Auth
security = HTTPBasic()

# 设置用户名和密码（建议从环境变量获取）
USERNAME = "admin"
PASSWORD = "admin123"

def verify_auth(credentials: HTTPBasicCredentials = Depends(security)):
    """验证Basic Auth凭证"""
    is_username_ok = secrets.compare_digest(credentials.username, USERNAME)
    is_password_ok = secrets.compare_digest(credentials.password, PASSWORD)
    if not (is_username_ok and is_password_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

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
async def get_home(username: str = Depends(verify_auth)):
    """返回首页HTML"""
    logger.debug("访问首页")
    return FileResponse("index.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket连接处理"""
    agent = None
    try:
        await websocket.accept()
        agent = Agent()
        logger.info("新的WebSocket连接建立")

        while True:
            try:
                message = await websocket.receive_text()
                logger.info(f"收到消息: {message[:50]}...")  # 记录收到的消息
                async for chunk in agent.reply(message):
                    if not chunk:  # 跳过空消息
                        continue
                    await websocket.send_json({"type": "message_chunk", "data": chunk})

                # 消息处理完成后，发送消息结束标记
                await websocket.send_json(
                    {"type": "message_end", "data": {"debug_info": None}}
                )

                # 发送知识图谱更新
                graph_data = agent.memory.knowledge_graph.get_graph_data()
                await websocket.send_json({"type": "graph_update", "data": graph_data})
                logger.debug("知识图谱更新已发送")

            except WebSocketDisconnect:
                logger.info("WebSocket连接断开")
                break
            except Exception as e:
                logger.error(f"消息处理错误: {e}")
                await websocket.send_json({"type": "error", "data": "消息处理失败"})
    finally:
        if agent:
            await agent.cleanup()
            logger.info("资源清理完成")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"全局错误: {exc}")
    return JSONResponse(status_code=500, content={"message": "服务器内部错误"})
