from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import json
from main import Agent
from pathlib import Path

app = FastAPI()

# 获取项目根目录
BASE_DIR = Path(__file__).resolve().parent

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

# 存储所有活跃的WebSocket连接
active_connections = {}

@app.get("/", response_class=HTMLResponse)
async def get_home():
    html_file = BASE_DIR / "static" / "index.html"
    return html_file.read_text(encoding="utf-8")

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    
    # 为每个连接创建一个新的Agent实例
    agent = Agent()
    active_connections[client_id] = websocket
    
    try:
        while True:
            # 接收消息
            message = await websocket.receive_text()
            
            # 调用Agent处理消息
            response, debug_info = agent.reply(message)
            
            # 构建返回数据
            response_data = {
                "response": response,
                "debug_info": debug_info,
                "history": agent.memory.get_chat_history()
            }
            
            # 发送响应
            await websocket.send_json(response_data)
            
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        # 清理连接
        if client_id in active_connections:
            del active_connections[client_id]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000) 