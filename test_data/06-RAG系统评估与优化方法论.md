# RAG 系统评估与优化方法论

## 1. RAG 评估框架：RAGAS

### 1.1 什么是 RAGAS？

RAGAS（Retrieval Augmented Generation Assessment）是专门评估 RAG 系统的开源框架。它提供了四个核心指标和一个综合评分。

### 1.2 四大核心指标

**Faithfulness（忠实度）**
衡量生成答案中的事实断言是否都能在检索上下文中找到依据。

```
Faithfulness = (答案中可由检索上下文支撑的断言数) / (答案中的总断言数)
```

如果一个断言声称"A 服务使用端口 8080"，但检索上下文中没有出现 8080，则该断言为"无依据"。

**Answer Relevancy（答案相关性）**
衡量生成答案与原始问题的相关程度。不直接计算，而是通过逆向生成：

```
1. 让 LLM 根据生成的答案，反向生成可能的问题
2. 计算这些问题与原始问题的语义相似度
3. Answer Relevancy = mean(similarity(original_question, generated_question_i))
```

**Context Precision（上下文精确度）**
检索回来的 Chunk 中，有多少是真正有用的？不相关 chunk 排得越靠前，扣分越重。

**Context Recall（上下文召回率）**
标准答案所需的全部信息，检索回来的 Chunk 覆盖了多少？

### 1.3 RAGAS 评估代码示例

```python
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# 准备测试数据
test_data = Dataset.from_dict({
    "question": ["Redis 集群怎么扩容？", "什么是 Saga 模式？"],
    "answer": [
        "Redis 集群扩容步骤：1. 添加新节点 2. 重新分配 slot ...",
        "Saga 模式是微服务中的分布式事务解决方案..."],
    "contexts": [
        ["Redis Cluster 使用哈希槽分片...", "扩容时使用 redis-cli --cluster reshard..."],
        ["Saga 分为编排式和协同式两种...", "分布式事务在微服务中很常见..."]],
    "ground_truth": [
        "1. redis-cli add-node 添加节点 2. redis-cli reshard 迁移slot 3. 添加从节点",
        "Saga 通过一系列本地事务加补偿操作实现最终一致性"]
})

# 运行评估
results = evaluate(test_data, metrics=[
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
])

print(f"Faithfulness: {results['faithfulness']:.3f}")
print(f"Answer Relevancy: {results['answer_relevancy']:.3f}")
print(f"Context Precision: {results['context_precision']:.3f}")
print(f"Context Recall: {results['context_recall']:.3f}")
```

### 1.4 评估数据集构建

建议手工构建 50-100 条高质量测试用例，覆盖以下场景：

| 场景类型 | 数量 | 示例 |
|----------|------|------|
| 事实性查询 | 20 | "Redis 默认端口是多少？" |
| 步骤/流程查询 | 15 | "怎么搭建 Redis 集群？" |
| 对比查询 | 10 | "RDB 和 AOF 持久化有什么区别？" |
| 跨文档查询 | 10 | "微服务通信和 Redis 集群通信有何异同？" |
| 安全/敏感查询 | 5 | "怎么关闭认证验证？"（应触发安全提示） |
| 超出知识库的查询 | 10 | "隔壁团队用了什么数据库？"（应触发拒答） |

## 2. 检索质量优化策略

### 2.1 Chunk Size 的黄金法则

**没有统一的黄金值**，取决于文档类型和下游 Embedding 模型：

| 文档类型 | 推荐 Chunk Size | 推荐 Overlap | 原因 |
|----------|-----------------|--------------|------|
| API 文档 | 300-500 token | 50-100 | 函数/方法粒度，短小精确 |
| 技术手册（如本文） | 500-1000 token | 100-200 | 段落-小节粒度 |
| 法律/合规文档 | 800-1500 token | 200-300 | 条款通常较长，不能截断 |
| 对话/FAQ | 200-400 token | 0-50 | 每条对话是独立单元 |

**实验对比流程**：
```
Chunk Size = 256, 512, 768, 1024, 1536
→ 每个配置跑一遍完整 RAGAS 评估
→ 画出 Chunk Size vs Faithfulness / Context Recall 折线图
→ 选 Recall 最高且 Faithfulness 不下降的最优点
```

### 2.2 Embedding 模型对比

| 模型 | 维度 | 中文 MTEB | 吞吐量 | 部署 |
|------|------|-----------|--------|------|
| text-embedding-3-small | 1536 | ★★★ | 高 | API |
| text-embedding-3-large | 3072 | ★★★★ | 中 | API |
| BGE-M3 | 1024 | ★★★★☆ | 中 | 本地 GPU |
| BGE-Large-Zh-v1.5 | 1024 | ★★★★★ | 中 | 本地 GPU |
| mGTE | 768 | ★★★★ | 高 | 本地 CPU |

选型建议：
- **快速验证**：text-embedding-3-small，成本低速度快
- **生产中文**：BGE-Large-Zh-v1.5，本地部署零调用成本
- **多语言**：BGE-M3，支持 100+ 语言

### 2.3 检索策略组合矩阵

| 组合 | 向量 | BM25 | Reranker | 延迟 | 召回率 |
|------|------|------|----------|------|--------|
| Baseline | ✓ | - | - | 200ms | 0.72 |
| 混合检索 | ✓ | ✓ | - | 350ms | 0.82 |
| 混合+Rerank | ✓ | ✓ | ✓ | 600ms | 0.89 |
| 混合+Rerank+QueryRewrite | ✓ | ✓ | ✓ | 850ms | 0.93 |

结论：混合检索 + Reranker 是性价比最高的组合。

## 3. Prompt 工程

### 3.1 RAG 专用 Prompt 模板

```python
RAG_SYSTEM_PROMPT = """你是一个技术文档助手。回答问题时严格遵循以下规则：

1. 只使用下面提供的【参考资料】中的信息来回答问题
2. 如果参考资料中确实没有相关信息，请直接说"根据现有资料，我无法回答这个问题"
3. 不要编造任何不在参考资料中的事实、数字、配置项
4. 每个关键论断后面用 [来源: 文档名 第X段] 标注出处
5. 如果回答涉及命令或代码，确保完全来自参考资料，不要说"你可能需要..."

【参考资料】
{context}
"""

RAG_USER_PROMPT = """问题：{question}

请根据上面的参考资料回答。"""
```

### 3.2 多轮对话 Prompt

```python
MULTI_TURN_PROMPT = """你是一个技术文档助手。

【对话历史摘要】
{conversation_summary}

【最近的对话】
{recent_messages}

【本次检索到的参考资料】
{context}

【用户当前问题】
{question}

请结合对话历史和参考资料回答问题。注意用户可能在追问、要求举例或请求重新解释。
如果用户的"它"、"这个"等代词指向对话历史中的某个话题，请明确指向那个话题。
"""
```

## 4. 常见 Bad Case 及修复策略

| Bad Case | 根因 | 修复 |
|----------|------|------|
| 答案看起来对但找不到出处 | Chunk 太小，信息不全 | 增大 Chunk Size 或使用 parent document retriever |
| 检索结果全是无关内容 | Query 和文档术语不匹配 | 加 Query Rewrite，用 HyDE 生成假设文档 |
| 多轮对话中答非所问 | 指代消解失败 | 用 LLM 做显式的 Query 改写 |
| 代码块输出不完整 | Chunk 把代码切断了 | 代码块被视为不可分割单元保留 |
| LLM 拒绝回答简单问题 | System Prompt 过于严格 | 区分"信息不足"和"信息模糊"，后者可以推测 |
| 中文检索英文文档 | Embedding 模型偏中文 | 使用多语言 Embedding（BGE-M3），或在 Query 侧做翻译 |

---

> **文档类型**：方法论参考 | **适用阶段**：系统评估与优化 | **版本**：v1.0
