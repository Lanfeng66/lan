"""GitHub 仓库数据源。"""
import os
import hashlib
from typing import List, Dict
from langchain_core.documents import Document
from app.ingestion.loader import load_file


def clone_or_pull(
    repo_url: str,
    branch: str = "main",
    target_dir: str = "./data/repos",
    token: str = None,
) -> str:
    """克隆或拉取 GitHub 仓库。Returns: 仓库在本地的工作目录路径"""
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    repo_dir = os.path.join(target_dir, repo_name)

    if os.path.exists(repo_dir):
        import subprocess
        result = subprocess.run(
            ["git", "-C", repo_dir, "pull", "origin", branch],
            capture_output=True, text=True
        )
        print(f"  📥 Git Pull: {result.stdout.strip()}")
        return repo_dir
    else:
        os.makedirs(target_dir, exist_ok=True)
        clone_url = repo_url
        if token:
            clone_url = repo_url.replace("https://", f"https://{token}@")
        import subprocess
        subprocess.run(
            ["git", "clone", "-b", branch, clone_url, repo_dir],
            check=True,
        )
        print(f"  📦 Git Clone: {repo_url} → {repo_dir}")
        return repo_dir


def list_doc_files(
    repo_dir: str,
    extensions: List[str] = None,
    exclude_dirs: List[str] = None,
) -> List[str]:
    """列出仓库中所有文档文件。"""
    if extensions is None:
        extensions = [".md", ".txt", ".rst"]
    if exclude_dirs is None:
        exclude_dirs = [".git", "node_modules", "__pycache__", ".venv", "vendor"]

    files = []
    for root, dirs, filenames in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for f in filenames:
            if any(f.endswith(ext) for ext in extensions):
                files.append(os.path.join(root, f))

    return files


def compute_file_hash(filepath: str) -> str:
    """计算文件 SHA256，用于增量检测。"""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()


def load_repo_documents(
    repo_dir: str,
    file_hashes: Dict[str, str] = None,
) -> Dict:
    """加载仓库中所有文档，支持增量更新。"""
    if file_hashes is None:
        file_hashes = {}

    doc_files = list_doc_files(repo_dir)
    all_docs = []
    new_hashes = {}
    stats = {"total": len(doc_files), "new": 0, "modified": 0, "unchanged": 0}

    for filepath in doc_files:
        current_hash = compute_file_hash(filepath)
        new_hashes[filepath] = current_hash

        old_hash = file_hashes.get(filepath, "")
        if old_hash == current_hash:
            stats["unchanged"] += 1
            continue

        if old_hash:
            stats["modified"] += 1
            print(f"  ✏️  文件变更: {filepath}")
        else:
            stats["new"] += 1
            print(f"  ➕ 新文件: {filepath}")

        try:
            docs = load_file(filepath)
            all_docs.extend(docs)
        except Exception as e:
            print(f"  ⚠️  解析失败 {filepath}: {e}")

    return {
        "documents": all_docs,
        "new_hashes": new_hashes,
        "stats": stats,
    }
