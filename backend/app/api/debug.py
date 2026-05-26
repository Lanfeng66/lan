"""检索调试 API：查看每一步的中间结果。"""
from fastapi import APIRouter, Query
from typing import Optional
from app.llm.chat_model import get_llm
from app.storage.vector_store import VectorStoreManager
from dotenv import load_dotenv
import os
import time

load_dotenv()

router = APIRouter(prefix="/api/v1/debug", tags=["debug"])
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")


@router.get("/retrieve")
async def debug_retrieve(
    kb_id: str = Query("docmind"),
    query: str = Query(...),
    top_k: int = Query(30),
):
    """逐步展示检索过程。"""
    vm = VectorStoreManager(persist_dir=CHROMA_DIR)
    llm = get_llm()
    steps = []
    total_start = time.time()

    # Step 1: Query Rewrite
    t0 = time.time()
    rewrite_prompt = f"将以下问题改写为更适合检索的形式：{query}"
    rewritten = llm.invoke(rewrite_prompt).content.strip()
    steps.append({
        "step": "query_rewrite",
        "input": query,
        "output": rewritten,
        "elapsed_ms": int((time.time() - t0) * 1000),
    })

    # Step 2: 向量检索
    t1 = time.time()
    vector_results = vm.similarity_search(rewritten, collection_name=kb_id, top_k=top_k)
    vector_formatted = []
    for doc in vector_results:
        vector_formatted.append({
            "content_preview": doc.page_content[:200],
            "source": doc.metadata.get("file_path", "未知"),
            "chunk_index": doc.metadata.get("chunk_index", 0),
        })
    steps.append({
        "step": "vector_search",
        "count": len(vector_formatted),
        "results": vector_formatted[:10],
        "elapsed_ms": int((time.time() - t1) * 1000),
    })

    # Step 3: BM25 检索
    t2 = time.time()
    bm25_ranked = []
    if len(vector_results) > 0:
        from rank_bm25 import BM25Okapi
        try:
            import jieba
            corpus = [list(jieba.cut(doc.page_content)) for doc in vector_results]
            tokenized_q = list(jieba.cut(rewritten))
        except ImportError:
            corpus = [doc.page_content.split() for doc in vector_results]
            tokenized_q = rewritten.split()

        bm25 = BM25Okapi(corpus)
        bm25_scores = bm25.get_scores(tokenized_q)
        bm25_ranked = sorted(
            enumerate(bm25_scores), key=lambda x: x[1], reverse=True
        )
    bm25_formatted = []
    for idx, score in bm25_ranked[:10]:
        doc = vector_results[idx]
        bm25_formatted.append({
            "content_preview": doc.page_content[:200],
            "bm25_score": float(score),
            "source": doc.metadata.get("file_path", "未知"),
            "chunk_index": doc.metadata.get("chunk_index", idx),
        })
    steps.append({
        "step": "bm25_search",
        "count": len(bm25_ranked),
        "results": bm25_formatted,
        "elapsed_ms": int((time.time() - t2) * 1000),
    })

    # Step 4: RRF 融合
    t3 = time.time()
    k = 60
    scores = {}
    for rank, doc in enumerate(vector_results):
        did = doc.page_content[:100]
        scores[did] = scores.get(did, 0) + 1 / (k + rank + 1)
    for rank, (idx, _) in enumerate(bm25_ranked):
        did = vector_results[idx].page_content[:100]
        scores[did] = scores.get(did, 0) + 1 / (k + rank + 1)

    seen = set()
    fused = []
    for did, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        if did not in seen:
            seen.add(did)
            fused.append({"content_preview": did, "rrf_score": score})
    steps.append({
        "step": "rrf_fusion",
        "count": len(fused),
        "results": fused[:10],
        "elapsed_ms": int((time.time() - t3) * 1000),
    })

    total_elapsed = int((time.time() - total_start) * 1000)

    return {
        "original_query": query,
        "rewritten_query": rewritten,
        "steps": steps,
        "total_elapsed_ms": total_elapsed,
    }
