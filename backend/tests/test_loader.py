from app.ingestion.loader import load_file
from app.ingestion.splitter import split_documents, add_chunk_index
def test_load_markdown():
    docs = load_file("../../test_data/01-Redis集群运维手册.md")
    assert len(docs) > 0
    assert docs[0].metadata["source_type"] == "markdown"
    print(f"✅ 加载成功：{len(docs)} 个文档片段")
    print(f"   第一个片段前 200 字：{docs[0].page_content[:200]}")
    return docs

def test_split_pipeline():
    # 加载
    docs = load_file("../../test_data/01-Redis集群运维手册.md")
    print(f"📄 加载：{len(docs)} 个文档片段")

    # 分割
    chunks = split_documents(docs, chunk_size=800, chunk_overlap=150)
    chunks = add_chunk_index(chunks)
    print(f"✂️  分割：{len(chunks)} 个 Chunk")

    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- Chunk #{i} (长度 {len(chunk.page_content)} 字) ---")
        print(chunk.page_content[:200])
        print(f"   metadata: {chunk.metadata}")

    print(f"\n✅ 完整管线：1 文档 → {len(chunks)} Chunks")

if __name__ == "__main__":
    #test_load_markdown()
    test_split_pipeline()
