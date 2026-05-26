"""LangGraph 工作流组装——支持 fast/quality 双模式。"""
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
    self_refine,
)


def _route_after_rewrite(state: RAGState) -> str:
    q = state["question"]
    complex_keywords = ["区别", "对比", "比较", "分别", "各自", "优缺点", "异同", "vs", "VS"]
    if any(kw in q for kw in complex_keywords):
        return "decompose"
    return "vector_search"


def build_rag_graph(fast_mode: bool = True):
    """构建 RAG 工作流。

    Args:
        fast_mode: True=跳过质检回路，3 次 LLM 调用 (< 10s)；
                   False=全流程含幻觉检测和自修正 (> 60s)
    """
    graph = StateGraph(RAGState)

    # ── 核心节点 ──
    graph.add_node("rewrite", rewrite_query)
    graph.add_node("vector_search", vector_retrieve)
    graph.add_node("bm25_search", bm25_retrieve)
    graph.add_node("fusion", rrf_fusion)
    graph.add_node("rerank", rerank)
    graph.add_node("generate", generate)

    # ── 可选节点 ──
    graph.add_node("decompose", decompose_query)
    graph.add_node("multi_retrieve", multi_query_retrieve)
    if not fast_mode:
        graph.add_node("hallucination_check", hallucination_check)
        graph.add_node("self_refine", self_refine)
        graph.add_node("relevance_check", check_relevance)

    graph.set_entry_point("rewrite")

    # ── 连线 ──
    graph.add_conditional_edges(
        "rewrite", _route_after_rewrite,
        {"decompose": "decompose", "vector_search": "vector_search"},
    )
    graph.add_edge("rewrite", "bm25_search")
    graph.add_conditional_edges(
        "decompose",
        lambda s: "multi_retrieve" if s.get("sub_queries") else "vector_search",
        {"multi_retrieve": "multi_retrieve", "vector_search": "vector_search"},
    )
    graph.add_edge("multi_retrieve", "fusion")
    graph.add_edge("vector_search", "fusion")
    graph.add_edge("bm25_search", "fusion")
    graph.add_edge("fusion", "rerank")
    graph.add_edge("rerank", "generate")

    # ── fast_mode: generate → END（跳过质检）──
    if fast_mode:
        graph.add_edge("generate", END)
        print("⚡ RAG 工作流已编译 (fast 模式: 检索→重排→生成)")
    else:
        graph.add_edge("generate", "hallucination_check")

        def _should_refine(state: RAGState) -> str:
            score = state.get("confidence_score", 1.0)
            if score < 0.7 and state.get("retry_count", 0) < 1:
                return "self_refine"
            return "end"

        graph.add_conditional_edges(
            "hallucination_check", _should_refine,
            {"self_refine": "self_refine", "end": END},
        )
        graph.add_edge("self_refine", "generate")
        print("🔬 RAG 工作流已编译 (quality 模式: 含幻觉检测+自修正)")

    return graph.compile()


# ── 默认使用 fast 模式 ──
rag_app = build_rag_graph(fast_mode=True)

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

