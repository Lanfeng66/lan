"""一键索引 test_data 目录下的所有文档。"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from app.ingestion.loader import load_file
from app.ingestion.splitter import split_documents, add_chunk_index
from app.storage.vector_store import VectorStoreManager
import config

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../test_data")
CHROMA_DIR = config.CHROMA_PERSIST_DIR

def index_all():
    vm = VectorStoreManager(persist_dir=CHROMA_DIR)

    all_chunks = []
    for filename in sorted(os.listdir(TEST_DATA_DIR)):
        if not filename.endswith((".md", ".txt", ".pdf")):
            continue

        filepath = os.path.join(TEST_DATA_DIR, filename)

        try:
            docs = load_file(filepath)
            chunks = split_documents(docs)
            chunks = add_chunk_index(chunks)
            all_chunks.extend(chunks)
            print(f"   → {len(chunks)} 个 Chunk")
        except Exception as e:
            print(f"   ⚠️  跳过（错误：{e}）")
    print(f"\n📊 总计 {len(all_chunks)} 个 Chunk，开始索引...")

    if not all_chunks:
        print("⚠️  没有可索引的文档，请检查 test_data 目录。")
        return

    # 先清空旧集合（方便重复跑脚本）
    try:
        vm.delete_collection("docmind")
    except Exception as e:
        print(f"   ⚠️  删除旧集合失败：{e}")

    vm.index_documents(all_chunks, collection_name="docmind")
    print("\n🎉 索引完成！")

if __name__ == "__main__":
    index_all()
    # vm = VectorStoreManager(persist_dir='./data/chroma_db')
    # results = vm.mmr_search('Redis集群怎么扩容？', collection_name='docmind', top_k=5, fetch_k=20)
    #
    # for i, doc in enumerate(results):
    #     print(f'\n--- 结果 {i + 1} ---')
    #     print(f'来源: {doc.metadata.get("file_path", "未知")}')
    #     print(f'内容: {doc.page_content[:200]}...')







