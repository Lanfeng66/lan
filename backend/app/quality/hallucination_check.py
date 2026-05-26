"""幻觉检测：NLI（自然语言推理）检查答案是否被检索上下文支持。"""
from typing import List, Dict, Tuple
import re
from app.llm.chat_model import get_llm


def extract_claims(answer: str) -> List[str]:
    """从答案中提取事实性断言。按句号/换行拆分，过滤太短的句子。"""
    sentences = re.split(r"[。\n]+", answer)
    claims = []
    for s in sentences:
        s = s.strip()
        if len(s) > 10 and not any(w in s for w in ["建议", "可以试试", "可能"]):
            claims.append(s)
    return claims


def check_claim_against_context(claim: str, contexts: List[str]) -> Tuple[bool, str]:
    """检查单个断言是否被上下文支持（保留用于兼容和单条检查）。"""
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


def check_claims_batch(
    claims: List[str],
    contexts: List[str],
    llm=None,
) -> List[Tuple[bool, str]]:
    """批量检查多条断言：将所有 claims 合并为一个 prompt，一次 LLM 调用完成。"""
    if not claims:
        return []

    if llm is None:
        llm = get_llm()

    context_text = "\n---\n".join(contexts[:5])
    claims_text = "\n".join(f"断言{i+1}: {claim}" for i, claim in enumerate(claims))

    prompt = f"""请判断以下每条【断言】是否能在【参考资料】中找到直接依据。

【参考资料】:
{context_text}

【断言列表】:
{claims_text}

对每条断言，请严格按格式回复（共{len(claims)}行，每行一条）：
断言N|标签

标签必须是以下三种之一：
- 支持: 参考资料中明确包含该信息
- 矛盾: 参考资料中有相反的信息
- 无依据: 参考资料中未提及该信息

回复："""

    response = llm.invoke(prompt)
    response_text = response.content if hasattr(response, "content") else str(response)

    # 解析每行 "断言N|标签"
    results = []
    for i in range(len(claims)):
        is_supported = False
        label = "无依据"
        pattern = re.compile(rf"断言{i+1}\s*[|：:]\s*(.+)")
        match = pattern.search(response_text)
        if match:
            label = match.group(1).strip()
        is_supported = "支持" in label
        results.append((is_supported, label))

    return results


def evaluate_answer(question: str, answer: str, contexts: List[str]) -> dict:
    """
    对答案做全面质量评估。

    Returns:
        {
            "hallucination_ratio": float,
            "claim_details": List[dict],
            "relevance_score": int,
            "completeness_score": int,
            "accuracy_score": int,
            "overall_pass": bool,
        }
    """
    contexts_text = [c["content"] if isinstance(c, dict) else c for c in contexts]

    # 1. 批量检查所有断言（最多 5 条，一次 LLM 调用）
    claims = extract_claims(answer)
    claims_to_check = claims[:5]

    if claims_to_check:
        llm = get_llm()
        claim_results = check_claims_batch(claims_to_check, contexts_text, llm=llm)
    else:
        claim_results = []

    claim_details = []
    unsupported_count = 0
    for claim, (is_supported, label) in zip(claims_to_check, claim_results):
        claim_details.append({
            "claim": claim,
            "is_supported": is_supported,
            "label": label,
        })
        if not is_supported:
            unsupported_count += 1

    hallucination_ratio = unsupported_count / max(len(claim_details), 1)

    # 2. LLM 综合评分
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
        hallucination_ratio < 0.3
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
