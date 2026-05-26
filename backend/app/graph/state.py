"""LangGraph 状态定义。"""
from typing import TypedDict, List, Optional


class RAGState(TypedDict):
    """RAG 状态定义。"""

    # 输入
    question: str
    #改写后的输入
    rewritten_query: str
    #向量检索结果
    vector_docs: List[dict]
    #BM25检索结果
    bm25_docs: List[dict]
    #融合后的结果
    retrieved_docs: List[dict]
    #最终拼接的文档
    final_context: List[dict]
    #最终生成的答案
    answer: str
    #引用来源
    citations: List[dict]
    #是否需要重试
    retry_count: int
    #相关性检查得分
    confidence_score: float
    # 新增：拆分后的子问题列表
    sub_queries: List[str]
    # 新增：对话历史
    conversation_history: List[dict]