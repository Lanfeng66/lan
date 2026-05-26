"""幻觉检测：NLI（自然语言推理）检查答案是否被检索上下文支持。"""
from typing import List, Dict, Tuple
import re
from app.llm.chat_model import get_llm



def extract_claims(answer: str) -> List[str]:
    """
    从答案中提取事实性断言。
    简化策略：按句号/换行拆分，过滤太短的句子。
    """
    sentences = re.split(r"[。\n]+", answer)
    claims = []
    for s in sentences:
        s = s.strip()
        # 过滤：太短的不是断言，包含"我"、"建议"等不是事实
        if len(s) > 10 and not any(w in s for w in ["建议", "可以试试", "可能"]):
            claims.append(s)
    return claims

def check_claim_against_context(claim: str,contexts: List[str],) -> Tuple[bool, str]:
    """
    检查单个断言是否被上下文支持。
    用 LLM 做 NLI（自然语言推理）。

    Returns:
        (is_supported, explanation)
        is_supported: True=被支持, False=矛盾或无依据
        explanation: 简短解释
    """
    llm = get_llm()

    context_text = "\n---\n".join(contexts[:5])

    prompt = f"""请判断以下【断言】是否能在【参考资料】中找到直接依据。

    【断言】: {claim}

    【参考资料】:
    {context_text}

    请用以下三种标签之一回复（只回复标签名）：
    - 支持: 参考资料中明确包含该信息
    - 矛盾: 参考资料中有相反的信息
    - 无依据: 参考资料中未提及该信息

    标签："""

    response = llm.invoke(prompt)
    label = (response.content if hasattr(response, "content") else str(response)).strip()

    is_supported = "支持" in label
    return is_supported, label

def evaluate_answer(question: str,answer: str,contexts: List[str],) -> dict:
    """
    对答案做全面质量评估。

    Returns:
        {
            "hallucination_ratio": float,  # 幻觉断言比例
            "claim_details": List[dict],    # 每个断言的检测结果
            "relevance_score": int,         # 1-5
            "completeness_score": int,      # 1-5
            "accuracy_score": int,          # 1-5
            "overall_pass": bool,           # 是否通过质检
        }
    """
    contexts_text = [c["content"] if isinstance(c, dict) else c for c in contexts]
    # 1. 提取断言并逐个检查
    claims = extract_claims(answer)
    claim_details = []
    unsupported_count = 0
    # 限制检查数量为10个
    for claim in claims[:10]:
        is_supported, label = check_claim_against_context(claim, contexts_text)
        claim_details.append({
            "claim": claim,
            "is_supported": is_supported,
            "label": label,
        })
        if not is_supported:
            unsupported_count += 1

    # 计算幻觉断言比例 （0-1之间）
    hallucination_ratio = unsupported_count / max(len(claim_details), 1)

    # 2. LLM 综合评分
    from app.llm.chat_model import get_llm
    llm = get_llm()

    score_prompt = f"""请对以下答案做质量评估，从三个维度打分（1-5）：

    问题：{question}
    答案：{answer}

    请按以下格式回复（每行一个数字）：
    相关性：X
    完整性：X
    准确性：X

    说明：相关性=答案是否切题；完整性=是否覆盖了关键信息；准确性=答案中提供的具体信息（数字、命令、配置项）是否正确。"""
    response = llm.invoke(score_prompt)
    scores_text = response.content if hasattr(response, "content") else str(response)

    # 解析分数
    relevance = 3
    completeness = 3
    accuracy = 3
    for line in scores_text.split("\n"):
        m = re.search(r'(\d)', line)
        if m and "相关" in line:
            relevance = int(m.group(1))
        elif m and "完整" in line:
            completeness = int(m.group(1))
        elif m and "准确" in line:
            accuracy = int(m.group(1))

    overall_pass = (
            hallucination_ratio < 0.3  # 幻觉率 < 30%
            and relevance >= 3
            and accuracy >= 3
    )

    return {
        "hallucination_ratio": hallucination_ratio,
        "claim_details": claim_details,
        "relevance_score": relevance,
        "completeness_score": completeness,
        "accuracy_score": accuracy,
        "overall_pass": overall_pass,
    }
