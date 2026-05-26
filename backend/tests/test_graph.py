import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.graph.workflow import rag_app

def test_langgraph_rag():
    question = "Redis集群扩容的具体步骤是什么？"

    print(f"❓ 问题: {question}\n")
    start = time.time()

    result = rag_app.invoke({
        "question": question,
        "retry_count": 0,
        "confidence_score": 0.0,
    })

    elapsed = time.time() - start

    print(f"\n{'='*60}")
    print(f"🤖 回答:\n{result['answer']}")
    print(f"\n📚 引用 ({len(result.get('citations', []))} 条):")
    for c in result.get("citations", []):
        print(f"   [{c['index']}] {c['source']} · Chunk #{c['chunk_index']}")
    print(f"\n⏱️  总耗时: {elapsed:.2f}s")
    print(f"🔄 最终改写 Query: {result.get('rewritten_query', 'N/A')}")

if __name__ == "__main__":
    test_langgraph_rag()