"""RAGAS 评估脚本。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.run_config import RunConfig

from openai import OpenAI
from ragas.llms import llm_factory
from app.embedding.embeddings import get_embedding_model
from app.rag.pipeline import BasicRAGPipeline
import config


def get_ragas_llm():
    """获取 RAGAS 评估用的 LLM。"""
    client = OpenAI(
        base_url=config.OPENAI_BASE_URL,
        api_key=config.OPENAI_API_KEY,
    )
    return llm_factory(
        model=config.LLM_MODEL,
        client=client,
        max_tokens=4096,
    )


def run_evaluation():
    print("📊 运行 RAG 管线，生成答案...")

    pipeline = BasicRAGPipeline()
    test_cases = [
        {
            "question": "Redis集群怎么水平扩容？",
            "ground_truth": "1. 使用redis-cli add-node添加新节点 2. 使用redis-cli reshard重新分配slot 3. 为新节点添加从节点"
        },
        {
            "question": "Docker多阶段构建的好处是什么？",
            "ground_truth": "减小最终镜像体积，将编译环境和运行环境分离，提高安全性"
        },
        {
            "question": "什么是Saga模式？",
            "ground_truth": "Saga是微服务中的分布式事务解决方案，分为编排式和协同式两种，通过一系列本地事务加补偿操作实现最终一致性"
        },
        {
            "question": "Python中asyncio和threading的区别是什么？",
            "ground_truth": "asyncio是单线程协程，适合I/O密集型；threading是多线程，受GIL限制，CPU密集型应使用multiprocessing"
        },
        {
            "question": "MySQL索引失效的常见场景有哪些？",
            "ground_truth": "函数包裹列、前导模糊查询LIKE '%xxx'、隐式类型转换、否定条件!=、JOIN字段字符集不同"
        },
    ]

    # 先跑管线，收集结果
    samples = []
    for case in test_cases:
        r = pipeline.ask(case["question"], top_k=5)
        print(f"  ❓ {case['question']}")
        print(f"  💬 {r['answer'][:80]}...")
        samples.append(SingleTurnSample(
            user_input=case["question"],
            response=r["answer"],
            retrieved_contexts=[s["preview"] for s in r["sources"]],
            reference=case["ground_truth"],
        ))

    # 初始化 LLM 和 Embeddings
    print("\n🔧 初始化评估模型...")
    ragas_llm = get_ragas_llm()
    embeddings = get_embedding_model()
    run_config = RunConfig()

    metrics = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }

    for name, m in metrics.items():
        m.llm = ragas_llm
        if name == "answer_relevancy":
            m.embeddings = embeddings
        m.init(run_config)

    # 逐个指标、逐个样本计算
    print("🔬 运行 RAGAS 评估...")
    all_scores = {name: [] for name in metrics}

    for name, m in metrics.items():
        print(f"\n  📏 计算 {name}...")
        for i, sample in enumerate(samples):
            try:
                score = m.single_turn_score(sample)
            except Exception as e:
                print(f"    ⚠️ 样本 {i+1} 出错: {e}")
                score = np.nan
            all_scores[name].append(score)
            print(f"    样本 {i+1}: {score:.4f}")

    # 汇总打印
    print("\n" + "=" * 50)
    print("📊 RAGAS 评估结果")
    print("=" * 50)
    for name, scores in all_scores.items():
        valid = [s for s in scores if not np.isnan(s)]
        if name == "faithfulness":
            print(f"  {name}: {[round(s, 4) for s in scores]}")
            if valid:
                print(f"  {name}(mean): {np.mean(valid):.4f}")
        else:
            mean = np.mean(valid) if valid else np.nan
            print(f"  {name}: {mean:.4f}")

    print("\n💡 指标解读：")
    print("  Faithfulness > 0.8      → 答案基本忠实于检索内容")
    print("  Answer Relevancy > 0.7  → 答案与问题相关度可接受")
    print("  Context Precision > 0.7 → 检索结果噪音较少")
    print("  Context Recall > 0.8    → 检索覆盖了标准答案所需信息")

    return all_scores


if __name__ == "__main__":
    run_evaluation()
