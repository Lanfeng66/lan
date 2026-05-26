import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from app.rag.pipeline import BasicRAGPipeline

def test_basic_rag():
    pipeline = BasicRAGPipeline(collection_name="docmind")

    questions = [
        "Redis集群怎么扩容？",
        "Docker镜像构建有哪些优化方法？",
        "微服务中Saga模式是什么？",
        "Windows 10怎么安装？（故意测试拒答）",
    ]
    for q in questions:
        print(f"\n{'='*60}")
        print(f"❓ 问题: {q}")
        start = time.time()
        result = pipeline.ask(q, top_k=5)
        elapsed = time.time() - start
        print(f"🤖 回答: {result['answer']}")
        print(f"\n📚 引用来源 ({len(result['sources'])} 条):")
        for s in result["sources"]:
            print(f"   [{s['index']}] {s['file']} · Chunk #{s['chunk_index']}")
        print(f"⏱️  耗时: {elapsed:.2f}s")

if __name__ == "__main__":
    test_basic_rag()
    
