"""索引 knowledge_base 目录下的所有文档。"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from app.ingestion.loader import load_file
from app.ingestion.splitter import split_documents, add_chunk_index
from app.storage.vector_store import VectorStoreManager
import config

KB_DIR = config.KNOWLEDGE_BASE_DIR
CHROMA_DIR = config.CHROMA_PERSIST_DIR


def index_all(reset: bool = False):
    os.makedirs(KB_DIR, exist_ok=True)
    vm = VectorStoreManager(persist_dir=CHROMA_DIR)

    files = sorted(f for f in os.listdir(KB_DIR)
                   if f.endswith((".md", ".txt", ".pdf", ".docx", ".html")))

    if not files:
        print(f"⚠️  {KB_DIR} 中没有可索引的文件")
        return

    all_chunks = []
    for filename in files:
        filepath = os.path.join(KB_DIR, filename)
        print(f"\n📄 处理：{filename}")

        try:
            docs = load_file(filepath)
            chunks = split_documents(docs)
            chunks = add_chunk_index(chunks)
            all_chunks.extend(chunks)
            print(f"   → {len(chunks)} 个 Chunk")
        except Exception as e:
            print(f"   ⚠️  跳过（错误：{e}）")

    print(f"\n📊 总计 {len(all_chunks)} 个 Chunk，开始索引...")

    if reset:
        try:
            vm.delete_collection("docmind")
            print("   🗑️  已清空旧集合")
        except Exception as e:
            print(f"   ⚠️  删除旧集合失败：{e}")

    vm.index_documents_batch(all_chunks, collection_name="docmind")
    print(f"\n🎉 索引完成！{len(files)} 个文件 → {len(all_chunks)} 个 Chunk → 集合 'docmind'")


if __name__ == "__main__":
    # 默认全量重建（删除旧集合→重新索引），保证和 knowledge_base 目录一致
    # 加 --append 参数则增量追加，不删除旧数据
    append = "--append" in sys.argv
    if not append:
        print("📋 默认模式：全量重建（删除旧集合→重新索引）")
        print("   加 --append 参数可切换为增量追加模式")
    index_all(reset=not append)







