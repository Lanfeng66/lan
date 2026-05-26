import os
from typing import List, Dict
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

from app.graph.state import RAGState
from app.llm.chat_model import get_llm
from app.storage.vector_store import VectorStoreManager
import config
from rank_bm25 import BM25Okapi
import jieba

CHROMA_DIR = config.CHROMA_PERSIST_DIR
COLLECTION = "docmind"

llm = get_llm()
vm = VectorStoreManager(persist_dir=CHROMA_DIR)

# ==================== 节点函数 ====================
def rewrite_query(state: RAGState) -> dict:
    """节点1：口语化问题 → 检索友好格式。"""
    question = state["question"]
    prompt = f"""将以下用户问题改写为更好的检索关键词（去除口语，保留关键字）：
原始问题：{question}
改写后："""
    try:
        response = llm.invoke(prompt)
        rewritten = response.content.strip()
        print(f"  🔄 Query Rewrite: {question} → {rewritten}")
        return {"rewritten_query": rewritten}
    except Exception as e:
        print(f"  ⚠️ Query 改写失败: {e}，使用原始问题")
        return {"rewritten_query": question}

def vector_retrieve(state: RAGState) -> dict:
    """节点2：向量检索。"""
    query = state.get("rewritten_query") or state["question"]
    try:
        docs = vm.mmr_search(query, COLLECTION, top_k=30)
        results = [_doc_to_dict(d) for d in docs]
    except Exception as e:
        print(f"  ⚠️ 向量检索失败（可能未索引）: {e}")
        results = []
    print(f"  🔍 向量检索: {len(results)} 条结果")
    return {"vector_docs": results}

def bm25_retrieve(state: RAGState) -> dict:
    """节点3：BM25 关键词检索。"""
    # 获取向量检索的结果列表里的所有 chunk 文本作为 BM25 语料
    # 简化实现：直接复用向量数据库中所有 chunk
    # 生产环境应维护一个独立的 BM25 索引
    query = state["rewritten_query"] or state["question"]
    # 这里做一个简化版 BM25：从 vector_docs 拿 30 条
    all_docs = state.get("vector_docs", [])
    if not all_docs:
        return {"bm25_docs": []}

    #分词
    corpus = [_jieba_tokenize(d["content"]) for d in all_docs]
    tokenized_query = _jieba_tokenize(query)

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenized_query)

    #按分数排序
    indexed = sorted(
        enumerate(scores), key=lambda x: x[1], reverse=True
    )

    results = [all_docs[i] for i, _ in indexed[:30]]
    print(f"  🔑 BM25 检索: {len(results)} 条结果")
    return {"bm25_docs": results}

def rrf_fusion(state: RAGState) -> dict:
    """节点4：RRF 融合向量和 BM25 的结果。"""
    vector_docs = state.get("vector_docs", [])
    bm25_docs = state.get("bm25_docs", [])
    k = 60  # RRF 参数

    # 计算 RRF 分数
    scores = {}
    for rank, doc in enumerate(vector_docs):
        doc_id = doc.get("content", "")[:100]  # 用内容前100字当 key
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

    for rank, doc in enumerate(bm25_docs):
        doc_id = doc.get("content", "")[:100]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

    #合并去重，按 RRF 分数排序
    seen = {}
    all_docs = vector_docs + bm25_docs
    for doc in all_docs:
        doc_id = doc.get("content", "")[:100]
        if doc_id not in seen:
            seen[doc_id] = (scores.get(doc_id, 0), doc)

    sorted_docs = sorted(seen.values(), key=lambda x: x[0], reverse=True)
    fused = [d for _, d in sorted_docs[:30]]
    print(f"  🔗 RRF 融合: {len(fused)} 条结果")
    return {"retrieved_docs": fused}

def rerank(state: RAGState) -> dict:
    """节点5：Cross-encoder 重排序（并行批量 LLM 打分）。"""
    docs = state.get("retrieved_docs", [])
    if len(docs) <= 5:
        return {"final_context": docs}

    question = state["question"]
    candidates = docs[:8]  # RRF 已排好序，8 个候选足够选 Top-5

    # 一次性构建所有 prompt，批量并行调用
    prompts = []
    for doc in candidates:
        prompt = (
            "问题：" + question + "\n"
            "资料片段：" + doc['content'][:300] + "\n\n"
            "这条资料对回答问题的有用程度，请打 1-10 分。\n"
            "只回复数字。"
        )
        prompts.append(prompt)

    try:
        responses = llm.batch(prompts, config={"max_concurrency": 8})
    except Exception:
        print("  Batch rerank failed, fallback to top-N")
        return {"final_context": candidates[:5]}

    scored = []
    for doc, response in zip(candidates, responses):
        try:
            score = float(response.content.strip()) / 10.0
        except (ValueError, AttributeError):
            score = 0.5
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    top5 = [d for _, d in scored[:5]]
    print(f"  Rerank: {len(scored)} candidates -> {len(top5)} selected (batch)")
    return {"final_context": top5}

def generate(state: RAGState) -> dict:
    """节点6：生成答案。"""
    question = state.get("question", "")
    context_docs = state.get("final_context") or state.get("retrieved_docs") or []

    # 拼接上下文
    if context_docs:
        context_parts = []
        for i, doc in enumerate(context_docs):
            source = doc.get("source", "未知")
            chunk_idx = doc.get("chunk_index", i)
            context_parts.append(
                f"[文档{i+1}] 来源: {source} 第{chunk_idx}段\n{doc['content']}"
            )
        context = "\n\n".join(context_parts)
    else:
        context = "（暂无参考资料）"

    system_msg = SystemMessage(content=f"""你是技术文档助手。只用以下资料回答问题，不编造。
如果资料为空，请告知用户当前知识库中没有相关内容。

【参考资料】
{context}""")
    human_msg = HumanMessage(content=question)
    try:
        response = llm.invoke([system_msg, human_msg])
    except Exception as e:
        print(f"  ⚠️ LLM 生成失败: {e}")
        return {"answer": f"抱歉，生成答案时出错：{e}", "citations": []}

    #构建来源列表
    citations = []
    for i, doc in enumerate(context_docs):
        source = doc.get("source", "未知")
        chunk_idx = doc.get("chunk_index", i)
        citations.append(
            {
                "index": i + 1,
                "source": source,
                "chunk_index": chunk_idx,
                "preview": doc.get("content", "")[:150],
            }
        )
    print(f"  ✨ 生成答案: {len(response.content)} 字")
    return {
        "answer": response.content,
        "citations": citations,
    }

def check_relevance(state: RAGState) -> dict:
    """节点7：相关性检查（决定是否需要重试）。"""
    # 简化实现：检查是否有检索结果
    docs = state.get("retrieved_docs", [])
    if not docs:
        return {"confidence_score": 0.0}
    # 有结果就给 0.7（真实场景用 Cross-encoder 打分）
    return {"confidence_score": 0.7}


def _jieba_tokenize(text: str) -> List[str]:
    """中文分词。"""
    try:
        return list(jieba.cut(text))
    except ImportError:
        return text.split()

# ==================== 辅助函数 ====================

def _doc_to_dict(doc: Document) -> dict:
    return {
        "content": doc.page_content,
        "source": doc.metadata.get("file_path", "未知"),
        "chunk_index": doc.metadata.get("chunk_index", 0),
        "metadata": doc.metadata,
    }


def decompose_query(state: RAGState) -> dict:
    """节点 X：检测并拆分复杂问题。"""
    question = state.get("question", "")

    # 检测是否需要拆分
    detect_prompt = f"""判断以下问题是否包含多个子问题（比如要求对比、列出多个步骤、或包含"分别""各自"等词）。
    只回复 YES 或 NO。

    问题：{question}"""
    response = llm.invoke(detect_prompt)
    need_decompose = response.content.strip().upper().startswith("YES")

    if not need_decompose:
        print("  📌 无需拆分，直接检索")
        return {"sub_queries": []}

    # 拆分
    split_prompt = f"""将以下复杂问题拆分为 2-4 个独立的子问题，每个子问题一行，用序号标注。
确保每个子问题可以独立检索回答。

原问题：{question}

子问题："""
    response = llm.invoke(split_prompt)
    lines =  response.content.strip().split("\n")
    sub_queries = []
    for line in lines:
        line = line.strip()
        # 去掉前面的序号 "1. " "1) " "- " 等
        for prefix in ["1. ", "2. ", "3. ", "4. ", "1) ", "2) ", "3) ", "4) ", "- "]:
            if line.startswith(prefix):
                line = line[len(prefix):]
                break
        if line and len(line) > 3:
            sub_queries.append(line)

    print(f"  🔀 Query 分解: {len(sub_queries)} 个子问题")
    for sq in sub_queries:
        print(f"      → {sq}")
    return {"sub_queries": sub_queries}


def multi_query_retrieve(state: RAGState) -> dict:
    """节点 Y：为每个子问题分别检索，然后合并去重。"""
    sub_queries = state.get("sub_queries", [])
    if not sub_queries:
        return {"retrieved_docs": []}

    # 检索每个子问题
    all_docs = []
    for sq in sub_queries:
        docs = vm.similarity_search(sq, collection_name=COLLECTION, top_k=10)
        all_docs.extend([_doc_to_dict(d) for d in docs])

    # 去重
    seen = set()
    unique_docs = []
    for doc in all_docs:
        key = doc["content"][:100]
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    print(f"  📦 多查询合并: {len(all_docs)} → {len(unique_docs)} (去重后)")
    return {"vector_docs": unique_docs}


def hallucination_check(state: RAGState) -> dict:
    """节点：幻觉检测。"""
    from app.quality.hallucination_check import evaluate_answer

    answer = state.get("answer", "")
    contexts = state.get("final_context", state.get("retrieved_docs", []))

    if not answer or not contexts:
        return {"confidence_score": 0.5}

    result = evaluate_answer(
        question=state["question"],
        answer=answer,
        contexts=contexts,
    )

    hallucination_ratio = result["hallucination_ratio"]
    overall_pass = result["overall_pass"]

    print(f"  🔬 幻觉检测: 幻觉率={hallucination_ratio:.0%}, "
          f"相关性={result['relevance_score']}, "
          f"完整性={result['completeness_score']}, "
          f"准确性={result['accuracy_score']}, "
          f"通过={overall_pass}")

    return {
        "confidence_score": 1.0 - hallucination_ratio,
    }


def self_refine(state: RAGState) -> dict:
    """节点：自修正——给 LLM 一次改进机会。"""
    question = state["question"]
    previous_answer = state["answer"]
    contexts = state.get("final_context", state.get("retrieved_docs", []))

    context_parts = []
    for i, doc in enumerate(contexts):
        source = doc.get("source", "未知") if isinstance(doc, dict) else "未知"
        content = doc["content"] if isinstance(doc, dict) else doc.page_content
        context_parts.append(f"[资料{i+1}] {source}\n{content}")
    context = "\n\n---\n\n".join(context_parts)

    refine_prompt = f"""你之前给了以下回答，但可能存在问题（信息不准确、不完整或有编造）。

【你的前一次回答】
{previous_answer}

【参考资料（唯一事实依据）】
{context}

【用户问题】
{question}

请重新回答。你必须：
1. 修正前一次回答中不准确的任何信息
2. 删除没有依据的任何断言
3. 如果有依据不足的地方，明确说明
4. 用 [来源: 资料X] 标注出处

重新回答："""

    from langchain_core.messages import HumanMessage
    response = llm.invoke([HumanMessage(content=refine_prompt)])

    print(f"  🔄 自修正完成: {len(previous_answer)}字 → {len(response.content)}字")
    return {"answer": response.content}