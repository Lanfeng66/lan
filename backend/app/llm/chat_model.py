from langchain_openai import ChatOpenAI
import config

_llm_instance = None  # 单例实例
def get_llm() -> ChatOpenAI:
    """获取 LLM 单例。"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatOpenAI(
            model=config.LLM_MODEL,
            base_url=config.OPENAI_BASE_URL,
            api_key=config.OPENAI_API_KEY,
            temperature=0.1,   # RAG 场景用低温，减少编造
            streaming=False,    # 启用流式输出，llm.stream() 才会逐 token 返回
        )
    return _llm_instance
