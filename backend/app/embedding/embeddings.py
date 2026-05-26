"""Embedding 模型封装——自动检测设备 + FP16 加速 + 批量编码。"""
import torch
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

_embedding_instance = None


def _detect_device() -> str:
    """自动检测最优设备：cuda > mps > cpu。"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_embedding_model(
    model_name: str = "BAAI/bge-m3",
    use_fp16: bool = True,
) -> Embeddings:
    """获取 Embedding 模型单例（懒加载，避免重复加载模型）。

    Args:
        model_name: HuggingFace 模型名或本地路径
        use_fp16: 是否启用 FP16 半精度（GPU 上可降低 50% 显存，提速 30-50%）
    """
    global _embedding_instance
    if _embedding_instance is None:
        device = _detect_device()

        model_kwargs = {"device": device}
        if use_fp16 and device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16

        # GPU 上增大 batch_size 充分利用并行能力
        batch_size = 64 if device == "cuda" else 16

        _embedding_instance = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": batch_size,
                "show_progress_bar": True,
            },
        )
        precision = "FP16" if model_kwargs.get("torch_dtype") == torch.float16 else "FP32"
        print(f"✅ Embedding 模型已加载：{model_name}，设备：{device}，精度：{precision}，batch={batch_size}")
    return _embedding_instance


def embed_batch(texts: list[str]) -> list[list[float]]:
    """批量编码文本列表——索引时一次性处理所有文本，避免碎片化调用。"""
    model = get_embedding_model()
    return model.embed_documents(texts)