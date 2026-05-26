"""GitHub 增量同步脚本（配合 cron/定时触发）。"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv; load_dotenv()

from app.ingestion.github_source import clone_or_pull, load_repo_documents
from app.ingestion.splitter import split_documents, add_chunk_index
from app.storage.vector_store import VectorStoreManager

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
HASH_FILE = "./data/github_file_hashes.json"


def sync_github_repo(
    repo_url: str = "https://github.com/redis/redis-doc.git",
    branch: str = "main",
    collection_name: str = "docmind",
):
    print(f"🔄 同步 GitHub: {repo_url} (branch={branch})")

    # 1. 克隆/拉取
    repo_dir = clone_or_pull(repo_url, branch)

    # 2. 读取上次的 hash 记录
    old_hashes = {}
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE) as f:
            old_hashes = json.load(f)

    # 3. 加载变更文件
    result = load_repo_documents(repo_dir, old_hashes)
    print(f"  📊 统计: 总{result['stats']['total']}个文件, "
          f"新增{result['stats']['new']}, "
          f"修改{result['stats']['modified']}, "
          f"未变{result['stats']['unchanged']}")

    # 4. 如果有变更，分割并索引
    if result["documents"]:
        chunks = split_documents(result["documents"])
        chunks = add_chunk_index(chunks)
        print(f"  ✂️  新增/更新 {len(chunks)} 个 Chunk")

        vm = VectorStoreManager(persist_dir=CHROMA_DIR)
        vm.index_documents(chunks, collection_name=collection_name)
        print(f"  ✅ 索引更新完成")

    # 5. 保存新的 hash 记录
    with open(HASH_FILE, "w") as f:
        json.dump(result["new_hashes"], f, indent=2)

    print("✅ 同步完成")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="https://github.com/redis/redis-doc.git")
    parser.add_argument("--branch", default="main")
    args = parser.parse_args()

    sync_github_repo(args.repo, args.branch)
