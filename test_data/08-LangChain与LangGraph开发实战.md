# LangChain 与 LangGraph 开发实战

## 第 1 章：LangChain 核心组件

### 1.1 Document Loaders（文档加载器）

LangChain 提供了 100+ 文档加载器。对于 RAG 项目，常用以下：

```python
from langchain_community.document_loaders import (
    PyPDFLoader,          # PDF 文件
    UnstructuredMarkdownLoader,  # Markdown 文件
    TextLoader,           # 纯文本
    WebBaseLoader,        # 网页内容
    Docx2txtLoader,       # Word 文档
    CSVLoader,            # CSV 数据
    JSONLoader,           # JSON 数据
)

# PDF 加载示例
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("/data/Redis运维手册.pdf")
documents = loader.load()  # 返回 List[Document]，每页一个 Document

# 网页加载
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://redis.io/docs/latest/operate/")
docs = loader.load()
```

### 1.2 Text Splitters（文本分割器）

```python
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    TokenTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

# 方案 A：递归字符分割——最通用，适合未知文档类型
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=100,
    separators=["\n\n", "\n", "。", ".", " ", ""]  # 按优先级尝试分割
)
chunks = splitter.split_documents(documents)

# 方案 B：按 Markdown 标题层级分割——适合技术文档
headers_to_split_on = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
chunks = markdown_splitter.split_text(markdown_text)

# 方案 C：按 token 数分割——精确控制 token 消费
from langchain.text_splitter import TokenTextSplitter
splitter = TokenTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    model_name="gpt-4o"
)
chunks = splitter.split_documents(documents)
```

### 1.3 Embeddings（向量嵌入）

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# OpenAI Embedding（API 调用）
openai_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=1024   # 可降维以节省存储
)

# BGE-M3 本地部署（推荐生产环境）
bge_embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True}
)
```

### 1.4 Vector Stores（向量数据库）

```python
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import Milvus

# Chroma（本地开发）
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=bge_embeddings,
    persist_directory="./chroma_db",
    collection_metadata={"hnsw:space": "cosine"}
)

# Milvus（生产环境）
from pymilvus import connections, Collection

vectorstore = Milvus.from_documents(
    documents=chunks,
    embedding=bge_embeddings,
    connection_args={"host": "localhost", "port": "19530"},
    collection_name="docmind_kb_001",
    drop_old=False,
)
```

### 1.5 Retrievers（检索器）

```python
# 基础检索
retriever = vectorstore.as_retriever(
    search_type="similarity",  # 余弦相似度
    search_kwargs={"k": 5}
)

# MMR 检索（最大边际相关性）—— 避免返回重复内容
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5}
)

# 带元数据过滤的检索
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": {"source_type": "pdf", "section_title": "部署"}
    }
)

# Self-query retriever —— LLM 自动推断过滤条件
from langchain.retrievers import SelfQueryRetriever

retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_content_description="技术文档",
    metadata_field_info=[
        {"name": "source_type", "description": "文档类型", "type": "string"},
        {"name": "kb_id", "description": "所属知识库ID", "type": "string"},
    ]
)
```

## 第 2 章：LangGraph 高级工作流

### 2.1 为什么用 LangGraph？

LangChain 的 `Chain`（如 `RunnableSequence`）是**线性的**，只能 A → B → C。真实 RAG 管线需要：

- **条件分支**：检索结果不相关时回到 Query Rewrite 重试
- **并发执行**：向量检索和 BM25 检索同时进行
- **循环**：自修正回路 —— 幻觉检查失败 → 重新生成 → 再检查
- **状态管理**：多轮对话需要维护会话状态

### 2.2 StateGraph 基础

```python
from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END
import operator

class RAGState(TypedDict):
    question: str
    rewritten_query: str
    sub_queries: List[str]
    retrieved_docs: List[dict]
    final_context: List[dict]
    answer: str
    confidence_score: float
    retry_count: int

# 创建图
graph = StateGraph(RAGState)

# 添加节点
def rewrite_query(state: RAGState) -> RAGState:
    # 将口语化问题改写为检索友好的查询
    rewritten = llm.invoke(
        f"将以下问题改写为更适合检索的形式：{state['question']}"
    )
    return {"rewritten_query": rewritten}

def retrieve(state: RAGState) -> RAGState:
    # 执行检索
    query = state.get("rewritten_query", state["question"])
    docs = retriever.invoke(query)
    return {"retrieved_docs": docs}

def generate(state: RAGState) -> RAGState:
    # 生成答案
    context = format_context(state["retrieved_docs"])
    answer = llm.invoke(f"根据以下资料回答问题：\n{context}\n\n问题：{state['question']}")
    return {"answer": answer}

graph.add_node("rewrite", rewrite_query)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)

# 添加边
graph.add_edge("rewrite", "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

# 设置入口
graph.set_entry_point("rewrite")

# 编译并运行
app = graph.compile()
result = app.invoke({"question": "Redis集群怎么扩容？", "retry_count": 0})
```

### 2.3 条件边（Conditional Edges）

```python
from langgraph.graph import StateGraph, END

def should_retry(state: RAGState) -> str:
    """判断是否需要重新检索"""
    if state["retry_count"] >= 2:
        return "end"  # 最多重试 2 次
    if state["confidence_score"] < 0.5:
        return "rewrite"  # 重试
    return "end"

graph.add_conditional_edges(
    "generate",          # 源节点
    should_retry,        # 判断函数
    {
        "rewrite": "rewrite",  # 回到改写节点
        "end": END             # 结束
    }
)
```

### 2.4 并行节点

```python
from langgraph.graph import StateGraph

# 向量检索节点
def vector_search(state: RAGState) -> RAGState:
    docs = vector_retriever.invoke(state["rewritten_query"])
    return {"vector_docs": docs}

# BM25 检索节点
def bm25_search(state: RAGState) -> RAGState:
    docs = bm25_retriever.search(state["rewritten_query"])
    return {"bm25_docs": docs}

# 融合节点（等两个检索都完成后自动调用）
def rrf_fusion(state: RAGState) -> RAGState:
    fused = reciprocal_rank_fusion(
        state["vector_docs"],
        state["bm25_docs"]
    )
    return {"retrieved_docs": fused}

graph.add_node("vector_search", vector_search)
graph.add_node("bm25_search", bm25_search)
graph.add_node("rrf_fusion", rrf_fusion)

# 并发：从 rewrite 同时出发到 vector_search 和 bm25_search
graph.add_edge("rewrite", "vector_search")
graph.add_edge("rewrite", "bm25_search")

# 两个搜索都完成后进入融合
graph.add_edge("vector_search", "rrf_fusion")
graph.add_edge("bm25_search", "rrf_fusion")
```

### 2.5 完整的 RAG 工作流

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

class FullRAGState(TypedDict):
    question: str
    conversation_history: list
    rewritten_query: str
    sub_queries: list
    retrieved_docs: list
    final_context: list
    answer: str
    citations: list
    hallucination_score: float
    retry_count: int

def build_rag_graph():
    graph = StateGraph(FullRAGState)

    # 添加所有节点
    graph.add_node("rewrite", rewrite_with_history)      # Query改写
    graph.add_node("decompose", decompose_complex_query)  # 复杂问题分解
    graph.add_node("vector_search", vector_retrieve)      # 向量检索
    graph.add_node("bm25_search", bm25_retrieve)          # BM25检索
    graph.add_node("fusion", rrf_fusion)                  # 结果融合
    graph.add_node("rerank", cross_encoder_rerank)        # 重排序
    graph.add_node("compress", context_compress)          # 上下文压缩
    graph.add_node("generate", generate_with_citations)   # 生成答案
    graph.add_node("hallucination_check", check_hallucination)  # 幻觉检测

    # 入口
    graph.set_entry_point("rewrite")

    # 条件边：是否复杂问题？
    graph.add_conditional_edges(
        "rewrite",
        lambda s: "decompose" if is_complex(s["question"]) else "vector_search",
        {"decompose": "decompose", "vector_search": "vector_search"}
    )

    # 并行检索
    graph.add_edge("decompose", "vector_search")
    graph.add_edge("decompose", "bm25_search")
    graph.add_edge("rewrite", "bm25_search")

    # 汇聚到融合
    graph.add_edge("vector_search", "fusion")
    graph.add_edge("bm25_search", "fusion")

    # 融合 → 重排序 → 压缩 → 生成
    graph.add_edge("fusion", "rerank")
    graph.add_edge("rerank", "compress")
    graph.add_edge("compress", "generate")

    # 条件边：幻觉检测
    graph.add_conditional_edges(
        "generate",
        lambda s: "generate" if s["hallucination_score"] < 0.7 and s["retry_count"] < 1 else "end",
        {"generate": "generate", "end": END}
    )

    # 编译（带内存检查点，支持多轮对话）
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)

rag_app = build_rag_graph()

# 运行
config = {"configurable": {"thread_id": "conversation-001"}}
result = rag_app.invoke(
    {"question": "Redis集群怎么扩容？", "retry_count": 0},
    config=config
)
```

## 第 3 章：生产化注意事项

### 3.1 回调与监控

```python
from langchain.callbacks import StdOutCallbackHandler
from langsmith import Client

# LangSmith 追踪（调试利器）
client = Client()
run = client.create_run(...)

# 自定义回调
class MetricsCallback(BaseCallbackHandler):
    def on_llm_start(self, *args, **kwargs):
        self.start_time = time.time()

    def on_llm_end(self, response, *args, **kwargs):
        elapsed = time.time() - self.start_time
        metrics.record("llm_call_duration", elapsed)
```

### 3.2 流式输出

```python
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# LangGraph 支持流式
for event in rag_app.stream(
    {"question": "Redis集群怎么扩容？"},
    config=config
):
    # event 包含每个节点的输出
    node_name = list(event.keys())[0]
    if node_name == "generate":
        print(event[node_name]["answer"], end="", flush=True)
```

---

> **文档版本**：v2.0 | **适用版本**：LangChain ≥ 0.2, LangGraph ≥ 0.1 | **最后更新**：2025-10-20
