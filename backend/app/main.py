"""FastAPI 主入口。"""
import sys
from pathlib import Path

# 将 backend 目录加入 sys.path，确保 import config 和 app 模块能找到
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import chat, documents, feedback, knowledge_bases, debug

app = FastAPI(
    title="DocMind API",
    description="企业级多源技术文档智能问答系统",
    version="0.1.0",
)

# CORS（允许前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# 路由注册
app.include_router(chat.router)
# 文档管理接口
app.include_router(documents.router)
app.include_router(feedback.router)
app.include_router(knowledge_bases.router)
app.include_router(debug.router)

@app.get("/health")
async def health():
    return {"status": "ok", "service": "docmind"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)