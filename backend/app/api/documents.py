"""文档管理 API。"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Optional
import os
import uuid
from pathlib import Path
from app.ingestion.loader import load_file
from app.ingestion.splitter import split_documents, add_chunk_index
from app.storage.vector_store import VectorStoreManager
import config

#路由器配置
router = APIRouter(prefix="/api/v1", tags=["documents"])
CHROMA_DIR = config.CHROMA_PERSIST_DIR

os.makedirs(config.UPLOAD_DIR, exist_ok=True)

@router.post("/documents/upload")
async def update_documents(kb_id: str = "docmind",file: UploadFile = File(...),):
    """上传文档并立即索引。"""
    #生成文件 ID
    file_id = str(uuid.uuid4())
    #获取文件后缀
    ext = Path(file.filename).suffix
    #保存文件
    save_path = os.path.join(config.UPLOAD_DIR, f"{file_id}{ext}")

    #读取文件
    context = await file.read()
    with open(save_path, "wb") as f:
        f.write(context)

    # 加载 → 分割 → 索引
    docs = load_file(save_path)
    chunks = split_documents(docs)
    chunks = add_chunk_index(chunks)

    vm = VectorStoreManager(persist_dir=CHROMA_DIR)
    vm.index_documents(chunks, collection_name=kb_id)

    return {
        "doc_id": file_id,
        "filename": file.filename,
        "status": "indexed",
        "chunk_count": len(chunks),
    }

@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """删除文档（简化：仅返回确认）。"""
    # 完整的删除实现需要在 Chroma 中按 doc_id 过滤删除
    # 这里做简化返回
    return {"status": "deleted", "doc_id": doc_id}