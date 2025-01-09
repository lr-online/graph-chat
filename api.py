from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import json
from main import Agent
from pathlib import Path
from loguru import logger
import sys
import time

# 配置logger
logger.remove()  # 移除默认的处理器
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/file_{time:YYYY-MM-DD}.log",
    rotation="00:00",  # 每天零点创建新文件
    retention="30 days",  # 保留30天的日志
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    encoding="utf-8"
)

app = FastAPI()

# 获取项目根目录
BASE_DIR = Path(__file__).resolve().parent
logger.info(f"项目根目录: {BASE_DIR}")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
logger.info("静态文件目录已挂载")

# 存储所有活跃的WebSocket连接
active_connections = {}

@app.get("/", response_class=HTMLResponse)
async def get_home():
    html_file = BASE_DIR / "static" / "index.html"
    logger.debug(f"访问首页: {html_file}")
    return html_file.read_text(encoding="utf-8")

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    logger.info(f"新的WebSocket连接: {client_id}")
    
    # 为每个连接创建一个新的Agent实例
    agent = Agent()
    active_connections[client_id] = websocket
    logger.debug(f"当前活跃连接数: {len(active_connections)}")
    
    # 发送初始知识图谱
    try:
        initial_graph_data = agent.memory.knowledge_graph.get_graph_data()
        await websocket.send_json({
            "knowledge_graph": initial_graph_data
        })
    except Exception as e:
        logger.error(f"发送初始知识图谱失败: {e}")
    
    try:
        while True:
            # 接收消息
            message = await websocket.receive_text()
            logger.info(f"收到消息 [client: {client_id}]: {message[:100]}...")
            
            start_time = time.time()
            
            try:
                # 调用Agent处理消息
                response, debug_info = agent.reply(message)
                
                # 获取更新后的知识图谱数据
                graph_data = agent.memory.knowledge_graph.get_graph_data()
                logger.debug(f"知识图谱数据: 节点数={len(graph_data['nodes'])}, 边数={len(graph_data['edges'])}")
                
                # 构建返回数据
                response_data = {
                    "response": response,
                    "debug_info": debug_info,
                    "history": agent.memory.get_chat_history(),
                    "knowledge_graph": graph_data
                }
                
                # 发送响应
                await websocket.send_json(response_data)
                
                process_time = time.time() - start_time
                logger.info(f"消息处理完成 [client: {client_id}] 耗时: {process_time:.2f}秒")
                logger.debug(f"调试信息: {debug_info}")
                
            except Exception as e:
                logger.error(f"消息处理错误 [client: {client_id}]: {str(e)}")
                # 发送错误信息给客户端
                error_response = {
                    "response": f"抱歉，处理消息时出现错误: {str(e)}",
                    "debug_info": {"error": str(e)},
                    "history": []
                }
                await websocket.send_json(error_response)
            
    except Exception as e:
        logger.error(f"WebSocket错误 [client: {client_id}]: {str(e)}")
    finally:
        # 清理连接
        if client_id in active_connections:
            del active_connections[client_id]
            logger.info(f"WebSocket连接关闭 [client: {client_id}]")
            logger.debug(f"当前活跃连接数: {len(active_connections)}")

@app.on_event("startup")
async def startup_event():
    logger.info("服务启动")
    # 确保日志目录存在
    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    logger.info(f"日志目录: {log_dir}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("服务关闭")

if __name__ == "__main__":
    import uvicorn
    logger.info("启动服务器...")
    uvicorn.run(app, host="0.0.0.0", port=9000) 