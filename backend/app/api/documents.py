"""文档管理 API。"""
from fastapi import APIRouter, UploadFile, File, HTTPException
import os
from app.ingestion.loader import load_file
from app.ingestion.splitter import split_documents, add_chunk_index
from app.storage.vector_store import VectorStoreManager
import config

#路由器配置
router = APIRouter(prefix="/api/v1", tags=["documents"])
CHROMA_DIR = config.CHROMA_PERSIST_DIR

os.makedirs(config.KNOWLEDGE_BASE_DIR, exist_ok=True)

@router.post("/documents/upload")
async def upload_document(kb_id: str = "docmind", file: UploadFile = File(...)):
    """上传文档并立即索引。"""
    save_path = os.path.join(config.KNOWLEDGE_BASE_DIR, file.filename)

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    docs = load_file(save_path)
    chunks = split_documents(docs)
    chunks = add_chunk_index(chunks)

    vm = VectorStoreManager(persist_dir=CHROMA_DIR)
    vm.index_documents(chunks, collection_name=kb_id)

    return {"filename": file.filename, "status": "indexed", "chunk_count": len(chunks)}


@router.delete("/documents/{filename}")
async def delete_document(kb_id: str = "docmind", filename: str = ""):
    """删除文档：从磁盘和向量库中同时移除。"""
    import chromadb

    file_path = os.path.join(config.KNOWLEDGE_BASE_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    try:
        client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
        col = client.get_collection(kb_id)
        col.delete(where={"file_path": file_path})
    except Exception as e:
        print(f"  ⚠️  删除向量失败：{e}")

    return {"status": "deleted", "filename": filename}


@router.post("/documents/rescan")
async def rescan_documents(kb_id: str = "docmind"):
    """重新扫描 knowledge_base 目录，全量重建索引。"""
    import config as cfg
    from app.ingestion.loader import load_file
    from app.ingestion.splitter import split_documents, add_chunk_index

    kb_dir = cfg.KNOWLEDGE_BASE_DIR
    vm = VectorStoreManager(persist_dir=cfg.CHROMA_PERSIST_DIR)

    files = sorted(f for f in os.listdir(kb_dir)
                   if f.endswith((".md", ".txt", ".pdf", ".docx", ".html")))

    all_chunks = []
    for fname in files:
        try:
            docs = load_file(os.path.join(kb_dir, fname))
            chunks = split_documents(docs)
            chunks = add_chunk_index(chunks)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  ⚠️  跳过 {fname}：{e}")

    try:
        vm.delete_collection(kb_id)
    except Exception:
        pass

    vm.index_documents_batch(all_chunks, collection_name=kb_id)

    return {
        "status": "rescanned",
        "files": len(files),
        "chunks": len(all_chunks),
    }