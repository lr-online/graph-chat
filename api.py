import os
import secrets
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import aiofiles
from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from loguru import logger

from core import Agent

# 初始化Basic Auth
security = HTTPBasic()

# 设置用户名和密码（建议从环境变量获取）
USERNAME = os.getenv("AUTH_USERNAME", "admin")
PASSWORD = os.getenv("AUTH_PASSWORD", "admin123")


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

# 创建上传目录
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# 挂载静态文件目录
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


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
                data = await websocket.receive_json()

                # 处理不同类型的消息
                message_type = data.get("type", "text")
                content = data.get("content", "")
                file_info = data.get("file_info", {})

                logger.info(f"收到消息: {content[:50]}...")

                async for chunk in agent.reply(content, message_type, file_info):
                    if not chunk:
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
                logger.exception(f"消息处理错误: {e}")
                await websocket.send_json({"type": "error", "data": "消息处理失败"})
    finally:
        if agent:
            await agent.cleanup()
            logger.info("资源清理完成")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"全局错误: {exc}")
    return JSONResponse(status_code=500, content={"message": "服务器内部错误"})


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...), username: str = Depends(verify_auth)
):
    """处理文件上传"""
    try:
        # 生成唯一文件名
        ext = Path(file.filename).suffix
        filename = f"{uuid.uuid4()}{ext}"
        filepath = UPLOAD_DIR / filename

        # 保存文件
        async with aiofiles.open(filepath, "wb") as f:
            content = await file.read()
            await f.write(content)

        # 返回文件信息
        return {
            "filename": filename,
            "original_name": file.filename,
            "content_type": file.content_type,
            "size": len(content),
            "url": f"/uploads/{filename}",
        }
    except Exception as e:
        logger.error(f"文件上传失败: {e}")
        raise HTTPException(status_code=500, detail="文件上传失败")
