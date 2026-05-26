"""多轮对话管理器：持久化、压缩、指代消解。"""
import uuid
from typing import List, Dict, Optional
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

class ConversationManager:
    """管理对话状态。简化版用内存字典存储，生产环境换成数据库。"""
    def __init__(self):
        self._store: Dict[str, List[dict]] = {}  # conv_id → [{role, content, time}]、

    def create_conversation(self) -> str:
        """创建一个新对话。"""
        conv_id = str(uuid.uuid4())
        self._store[conv_id] = []
        return conv_id

    def get_history(self, conv_id: str, last_n: int = 5) -> List[dict]:
        # 获取指定对话的历史记录
        return self._store.get(conv_id, [])[-last_n:]

    def add_message(self, conv_id: str, role: str, content: str):
        """添加一条消息。"""
        if conv_id not in self._store:
            self._store[conv_id] = []
        self._store[conv_id].append({
            "role": role,
            "content": content,
            "time": datetime.now().isoformat(),
        })

    def rewrite_with_history(
            self, conv_id: str, question: str
    ) -> str:
        """
        结合对话历史，将依赖上下文的指代词替换为具体的实体。
        例如 "它怎么部署？" → "DocMind 服务怎么部署？"
        """
        history = self.get_history(conv_id, last_n=6)
        if not history:
            return question

        from app.llm.chat_model import get_llm
        llm = get_llm()

        history_text = "\n".join([
            f"{'用户' if m['role'] == 'user' else '助手'}: {m['content'][:200]}"
            for m in history
        ])

        prompt = f"""根据对话历史，将用户当前问题中的指代词替换为具体所指。

        对话历史：
        {history_text}

        当前问题：{question}

        规则：
        - 如果问题中有"它"、"这个"、"那个"等指代词，替换为历史中提到过的事物
        - 如果问题本身已经完整清晰，直接原样返回
        - 只返回改写后的问题，不包含任何解释

        改写后的问题："""

        response = llm.invoke(prompt)
        rewritten = response.content.strip()
        if rewritten != question:
            print(f"  💬 指代消解: {question} → {rewritten}")
        return rewritten

    def build_compressed_context(self, conv_id: str, max_tokens: int = 2000) -> str:
        """
        对话历史过长时，对早期消息做摘要压缩，保留最近几轮完整对话。
        策略：前 70% 的消息压缩成摘要，后 30% 保留原文。
        """
        history = self._store.get(conv_id, [])
        if not history:
            return ""

        # 粗略估算 token 数（中文约 1.5 字符/token）
        total_chars = sum(len(m["content"]) for m in history)
        estimated_tokens = total_chars / 1.5

        if estimated_tokens <= max_tokens:
            # 不需要压缩
            return "\n".join([
                f"{'👤' if m['role'] == 'user' else '🤖'}: {m['content']}"
                for m in history
            ])

        # 需要压缩：前 70% 做摘要，后 30% 保留原文
        split_idx = int(len(history) * 0.7)
        early = history[:split_idx]
        recent = history[split_idx:]

        from app.llm.chat_model import get_llm
        llm = get_llm()

        early_text = "\n".join([
            f"{'用户' if m['role'] == 'user' else '助手'}: {m['content'][:300]}"
            for m in early
        ])

        summary_prompt = f"""请用 2-3 句话摘要以下对话中讨论的主要技术话题和关键结论：

    {early_text}

    摘要："""

        response = llm.invoke(summary_prompt)
        summary = response.content.strip()

        recent_text = "\n".join([
            f"{'👤' if m['role'] == 'user' else '🤖'}: {m['content']}"
            for m in recent
        ])

        return f"【对话前期摘要】{summary}\n\n【最近对话】\n{recent_text}"

    def detect_topic_switch(
            self, conv_id: str, new_question: str, threshold: float = 0.3
    ) -> bool:
        """
                检测当前问题是否与上一轮话题完全不同。
                如果语义相似度低于阈值，则认为是话题切换。
                """
        history = self.get_history(conv_id, last_n=2)
        if not history:
            return False

        from app.embedding.embeddings import get_embedding_model
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        emb_model = get_embedding_model()
        last_user_msgs = [
            m["content"] for m in history if m["role"] == "user"
        ]
        if not last_user_msgs:
            return False

        # 计算当前问题和上一条用户消息的嵌入
        last_emb = emb_model.embed_query(last_user_msgs[-1])
        curr_emb = emb_model.embed_query(new_question)

        similarity = cosine_similarity(
            np.array(last_emb).reshape(1, -1),
            np.array(curr_emb).reshape(1, -1),
        )[0][0]

        is_switch = similarity < threshold
        if is_switch:
            print(f"  🎯 话题切换检测: similarity={similarity:.3f} < {threshold}")
        return is_switch




# 全局单例
conv_manager = ConversationManager()

