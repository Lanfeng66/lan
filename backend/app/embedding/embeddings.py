"""Embedding 模型封装。"""
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

_embedding_instance = None  # 单例实例
def get_embedding_model(model_name: str = "BAAI/bge-m3") -> Embeddings:
    """获取 Embedding 模型单例（懒加载，避免重复加载模型）。"""
    global _embedding_instance
    if _embedding_instance is None:
        _embedding_instance = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},  # 没有 GPU 就用 cpu
            encode_kwargs={"normalize_embeddings": True},
        )
        print(f"✅ Embedding 模型已加载：{model_name}")
    return _embedding_instance