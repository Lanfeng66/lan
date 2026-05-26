"""DocMind 中心化配置模块。

导入时自动从项目根目录加载 .env，不依赖当前工作目录。
所有配置值均为模块级常量，通过 ``import config`` 使用。
"""
import os
import sys
from dotenv import load_dotenv

# ── 强制 UTF-8 输出（避免 Windows GBK 终端报 UnicodeEncodeError）──
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── 项目根目录（config.py 与 .env 同级）──
_project_root = os.path.dirname(os.path.abspath(__file__))
_env_path = os.path.join(_project_root, ".env")

load_dotenv(_env_path, override=True)


def _resolve_path(path: str) -> str:
    """将 ``./`` 开头的相对路径解析为基于项目根目录的绝对路径。"""
    if path.startswith("./"):
        return os.path.normpath(os.path.join(_project_root, path[2:]))
    return path


# ── 路径常量 ──
PROJECT_ROOT = _project_root
DATA_DIR = os.path.join(_project_root, "data")

# ── LLM / 模型供应商 ──
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ── 向量数据库 (Chroma) ──
CHROMA_PERSIST_DIR = _resolve_path(os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db"))


UPLOAD_DIR = _resolve_path(os.getenv("UPLOAD_DIR", "./data/uploads"))

# ── 关系型数据库 ──
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/docmind.db")

# ── 外部 API Keys ──
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")


def as_dict() -> dict:
    """返回所有公开配置的字典（API Key 自动掩码）。"""
    return {
        "PROJECT_ROOT": PROJECT_ROOT,
        "DATA_DIR": DATA_DIR,
        "LLM_MODEL": LLM_MODEL,
        "OPENAI_BASE_URL": OPENAI_BASE_URL,
        "OPENAI_API_KEY": "***" if OPENAI_API_KEY else "",
        "CHROMA_PERSIST_DIR": CHROMA_PERSIST_DIR,
        "DATABASE_URL": DATABASE_URL,
        "TAVILY_API_KEY": "***" if TAVILY_API_KEY else "",
    }
