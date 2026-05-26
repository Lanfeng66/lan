"""LangGraph 工作流组装。"""
from langgraph.graph import StateGraph, END
from app.graph.state import RAGState
from app.graph.nodes import (
    rewrite_query,
    vector_retrieve,
    bm25_retrieve,
    rrf_fusion,
    rerank,
    generate,
    check_relevance,
    decompose_query,
    multi_query_retrieve,
    hallucination_check,
    self_refine
)
def build_rag_graph():
    """构建并返回编译后的 RAG 工作流。"""
    graph = StateGraph(RAGState)

    # 添加节点
    graph.add_node("rewrite", rewrite_query)
    graph.add_node("vector_search", vector_retrieve)
    graph.add_node("bm25_search", bm25_retrieve)
    graph.add_node("fusion", rrf_fusion)
    graph.add_node("rerank", rerank)
    graph.add_node("generate", generate)
    graph.add_node("relevance_check", check_relevance)
    graph.add_node("decompose", decompose_query)
    graph.add_node("multi_retrieve", multi_query_retrieve)
    graph.add_node("hallucination_check", hallucination_check)
    graph.add_node("self_refine", self_refine)

    # 设置入口
    graph.set_entry_point("rewrite")

    # 连线：rewrite → 并发(向量 + BM25)
    # 改连线：rewrite → decompose（条件）
    def route_after_rewrite(state: RAGState) -> str:
        """判断是否需要分解。先走 decompose，内部判断后跳过或执行。"""
        # 对于明确包含对比/并列的问题才进分解
        q = state["question"]
        complex_keywords = ["区别", "对比", "比较", "分别", "各自", "优缺点", "异同", "vs", "VS"]
        if any(kw in q for kw in complex_keywords):
            return "decompose"
        return "vector_search"

    graph.add_conditional_edges(
        "rewrite",
        route_after_rewrite,
        {
            "decompose": "decompose",
            "vector_search": "vector_search",
        }
    )

    # 向量检索结果 → 融合
    graph.add_edge("rewrite", "bm25_search")

    # decompose → multi_retrieve（如果拆了子问题）→ 直接跳到 fusion
    graph.add_conditional_edges(
        "decompose",
        lambda s: "multi_retrieve" if s.get("sub_queries") else "vector_search",
        {
            "multi_retrieve": "multi_retrieve",
            "vector_search": "vector_search",
        }
    )
    graph.add_edge("multi_retrieve", "fusion")

    # 向量检索结果 → 融合
    graph.add_edge("vector_search", "fusion")
    graph.add_edge("bm25_search", "fusion")

    # 融合 → 重排 → 生成
    graph.add_edge("fusion", "rerank")
    graph.add_edge("rerank", "generate")

    # 生成 → 质检
    graph.add_edge("generate", "hallucination_check")

    # 条件边：质检不通过 → 自修正 → 重新生成（最多 1 次）
    def should_refine(state: RAGState) -> str:
        score = state.get("confidence_score", 1.0)
        retries = state.get("retry_count", 0)
        if score < 0.7 and retries < 1:
            print(f"  ⚠️  质检不通过 (score={score:.2f})，触发自修正...")
            return "self_refine"
        return "end"

    graph.add_conditional_edges(
        "hallucination_check",
        should_refine,
        {"self_refine": "self_refine", "end": END}
    )

    # self_refine → 重新生成
    graph.add_edge("self_refine", "generate")

    return graph.compile()

# 编译好的应用实例
rag_app = build_rag_graph()

if __name__ == "__main__":
    result = rag_app.invoke({
        'question': 'Redis集群怎么扩容，请说具体命令',
        'retry_count': 0,
        'confidence_score': 0.0,
        'sub_queries': [],
        'conversation_history': [],
    })
    print('=== 答案 ===')
    print(result['answer'])
    print(f'\n=== 置信度: {result.get("confidence_score", "N/A")} ===')

