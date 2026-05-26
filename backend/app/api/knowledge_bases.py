"""知识库 CRUD API。"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import uuid
from app.storage.vector_store import VectorStoreManager
from dotenv import load_dotenv
import os

load_dotenv()

router = APIRouter(prefix="/api/v1", tags=["knowledge-bases"])
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")


class KBCreateRequest(BaseModel):
    name: str
    description: str = ""
    embedding_model: str = "BAAI/bge-m3"


@router.post("/knowledge-bases")
async def create_kb(req: KBCreateRequest):
    """创建知识库。"""
    kb_id = str(uuid.uuid4())

    import json
    meta_dir = "./data/kb_meta"
    os.makedirs(meta_dir, exist_ok=True)
    with open(f"{meta_dir}/{kb_id}.json", "w") as f:
        json.dump({
            "id": kb_id,
            "name": req.name,
            "description": req.description,
            "embedding_model": req.embedding_model,
            "document_count": 0,
        }, f)

    return {"kb_id": kb_id, "name": req.name, "status": "created"}


@router.get("/knowledge-bases")
async def list_kbs():
    """列出所有知识库。"""
    import json, glob
    kbs = []
    for fpath in glob.glob("./data/kb_meta/*.json"):
        with open(fpath) as f:
            kbs.append(json.load(f))
    return {"knowledge_bases": kbs, "total": len(kbs)}


@router.delete("/knowledge-bases/{kb_id}")
async def delete_kb(kb_id: str):
    """删除知识库及其所有向量数据。"""
    vm = VectorStoreManager(persist_dir=CHROMA_DIR)
    try:
        vm.delete_collection(kb_id)
    except Exception:
        pass

    meta_path = f"./data/kb_meta/{kb_id}.json"
    if os.path.exists(meta_path):
        os.remove(meta_path)

    return {"status": "deleted", "kb_id": kb_id}
