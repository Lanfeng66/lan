"""基础 RAG 问答管线。"""
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from app.llm.chat_model import get_llm
from app.storage.vector_store import VectorStoreManager
import config

CHROMA_DIR = config.CHROMA_PERSIST_DIR

RAG_SYSTEM_PROMPT = """你是一个技术文档助手。请严格遵循以下规则：

1. 只使用下面【参考资料】中提供的信息回答问题
2. 如果资料中没有相关信息，请直接说：根据现有资料，我无法回答这个问题。
3. 不要在资料之外编造任何事实、数字、配置项或命令
4. 回答时，用 [来源: 文件名 第X段] 的格式标注信息出处

【参考资料】
{context}"""

class BasicRAGPipeline:
    """最简 RAG 管线：检索 → 拼接 → 生成。"""
    def __init__(self, collection_name: str = "docmind"):
        self.collection_name = collection_name
        self.vector_store = VectorStoreManager(CHROMA_DIR)
        self.llm = get_llm()

    def ask(self, question: str, top_k: int = 3)->dict :
        """
            单轮问答。
            Returns:{"question": ..., "answer": ..., "sources": [...]}
        """
        # Step 1: 检索
        retrieved_docs = self.vector_store.similarity_search(
            question,
            self.collection_name,
            top_k
        )
        # Step 2: 拼接
        context = self._format_context(retrieved_docs)

        # Step 3: 构建 Prompt
        system_msg = SystemMessage(content=RAG_SYSTEM_PROMPT.format(context=context))
        human_msg = HumanMessage(content=question)

        # Step 4: 调用 LLM
        answer = self.llm.invoke([system_msg, human_msg])

        # Step 5: 构建来源列表
        sources = []
        for i, doc in enumerate(retrieved_docs):
            sources.append({
                "index": i + 1,
                "file": doc.metadata.get("file_path", "未知"),
                "chunk_index": doc.metadata.get("chunk_index", "未知"),
                "preview": doc.page_content[:150],
            })

        return {
            "question": question,
            "answer": answer.content,
            "sources": sources,
        }

    def _format_context(self, docs: List[Document]) -> str:
        """将检索到的文档格式化为 LLM 可读的上下文字符串。"""
        parts = []
        for i, doc in enumerate(docs):
            file_name = doc.metadata.get("file_path", "未知").split("/")[-1]
            chunk_idx = doc.metadata.get("chunk_index", i)
            parts.append(
                f"[文档{i+1}] 来源: {file_name} 第{chunk_idx}段\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(parts)
