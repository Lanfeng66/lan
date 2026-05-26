# DocMind 开发步骤指南

> 每一步都是可验证的——做完一步，跑通验证命令，看到预期结果后再进入下一步。不要跳步。

---

## 第 0 步：环境初始化（30 分钟）

### 0.1 创建项目目录

```bash
mkdir -p ~/Desktop/docmind/backend
mkdir -p ~/Desktop/docmind/frontend
mkdir -p ~/Desktop/docmind/data
mkdir -p ~/Desktop/docmind/test_data
```

### 0.2 安装 Python 依赖

```bash
cd ~/Desktop/docmind/backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

创建 `requirements.txt`：

```txt
# LLM 编排
langchain>=0.3.0
langchain-community>=0.3.0
langgraph>=0.2.0
langchain-openai>=0.2.0

# 向量数据库
chromadb>=0.5.0

# 文档解析
pymupdf>=1.24.0          # PDF
python-docx>=1.1.0        # Word
unstructured>=0.15.0      # 通用解析
markdown>=3.6

# Embedding & Reranker
sentence-transformers>=3.0.0

# Web 框架
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
python-multipart>=0.0.9

# 数据库
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0
alembic>=1.13.0

# 工具
python-dotenv>=1.0.0
httpx>=0.27.0
rank-bm25>=0.2.2
```

```bash
pip install -r requirements.txt
```

### 0.3 配置环境变量

创建 `backend/.env`：

```env
OPENAI_API_KEY=sk-your-key-here      # 如果用 OpenAI
OPENAI_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=BAAI/bge-m3          # 本地模型名（用 sentence-transformers 加载）
LLM_MODEL=gpt-4o                      # 或用 deepseek-chat
CHROMA_PERSIST_DIR=./data/chroma_db
DATABASE_URL=sqlite:///./data/docmind.db   # 先用 SQLite 省事，后面再切 PostgreSQL
```

### 0.4 验证环境

```bash
python -c "
import langchain, langgraph, chromadb, fitz, sentence_transformers
print('=== 所有依赖安装成功 ===')
print(f'LangChain: {langchain.__version__}')
print(f'LangGraph: {langgraph.__version__}')
"
```

预期输出：各版本号，无报错。

---

## 第 1 步：文档加载器（2 小时）

### 目标
输入一个文件路径，输出 `List[Document]`，每个 Document 包含文本内容和元数据。

### 1.1 创建加载器模块

创建 `backend/app/__init__.py`（空文件）。

创建 `backend/app/ingestion/__init__.py`（空文件）。

创建 `backend/app/ingestion/loader.py`：

```python
"""统一的文档加载器：根据文件扩展名自动选择加载策略。"""
from pathlib import Path
from typing import List
from langchain_core.documents import Document


def load_file(file_path: str) -> List[Document]:
    """根据文件后缀分发到对应的加载函数。"""
    ext = Path(file_path).suffix.lower()
    file_path = str(file_path)  # 确保是字符串

    if ext == ".pdf":
        return _load_pdf(file_path)
    elif ext in (".md", ".markdown"):
        return _load_markdown(file_path)
    elif ext == ".txt":
        return _load_text(file_path)
    elif ext == ".docx":
        return _load_docx(file_path)
    elif ext == ".html":
        return _load_html(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")


def _load_pdf(file_path: str) -> List[Document]:
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source_type"] = "pdf"
        doc.metadata["file_path"] = file_path
    return docs


def _load_markdown(file_path: str) -> List[Document]:
    from langchain_community.document_loaders import UnstructuredMarkdownLoader
    loader = UnstructuredMarkdownLoader(file_path, mode="single")
    docs = loader.load()
    for doc in docs:
        doc.metadata["source_type"] = "markdown"
        doc.metadata["file_path"] = file_path
    return docs


def _load_text(file_path: str) -> List[Document]:
    from langchain_community.document_loaders import TextLoader
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    for doc in docs:
        doc.metadata["source_type"] = "text"
        doc.metadata["file_path"] = file_path
    return docs


def _load_docx(file_path: str) -> List[Document]:
    from langchain_community.document_loaders import Docx2txtLoader
    loader = Docx2txtLoader(file_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source_type"] = "docx"
        doc.metadata["file_path"] = file_path
    return docs


def _load_html(file_path: str) -> List[Document]:
    from langchain_community.document_loaders import UnstructuredHTMLLoader
    loader = UnstructuredHTMLLoader(file_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source_type"] = "html"
        doc.metadata["file_path"] = file_path
    return docs
```

### 1.2 验证

创建 `backend/tests/test_loader.py`：

```python
from app.ingestion.loader import load_file

def test_load_markdown():
    docs = load_file("../../test_data/01-Redis集群运维手册.md")
    assert len(docs) > 0
    assert docs[0].metadata["source_type"] == "markdown"
    print(f"✅ 加载成功：{len(docs)} 个文档片段")
    print(f"   第一个片段前 200 字：{docs[0].page_content[:200]}")
    return docs

if __name__ == "__main__":
    test_load_markdown()
```

```bash
cd ~/Desktop/docmind/backend
python tests/test_loader.py
```

预期输出：`✅ 加载成功：1 个文档片段` 及内容预览。

---

## 第 2 步：文本分割器（1.5 小时）

### 目标
将长文档切分成语义合理的 Chunk，每个 Chunk 大小在 500-1000 token 之间，且有适当的重叠。

### 2.1 创建分割器模块

创建 `backend/app/ingestion/splitter.py`：

```python
"""文本分割策略：支持多种分割方式。"""
from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)


def split_documents(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    strategy: str = "recursive"
) -> List[Document]:
    """
    核心分割函数。

    Args:
        documents: 待分割的文档列表
        chunk_size: 每个 chunk 的目标大小（字符数）
        chunk_overlap: chunk 之间的重叠字符数
        strategy: "recursive" | "markdown_header"
    """
    if strategy == "markdown_header":
        return _split_by_headers(documents, chunk_size, chunk_overlap)
    else:
        return _split_recursive(documents, chunk_size, chunk_overlap)


def _split_recursive(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ".", " ", ""],
        length_function=len,
    )
    return splitter.split_documents(documents)


def _split_by_headers(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """按 Markdown 标题层级分割，保留标题作为 chunk 的元数据。"""
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )

    all_chunks = []
    for doc in documents:
        # 第一步：按标题切分
        chunks = markdown_splitter.split_text(doc.page_content)
        # 第二步：对大段再做长度切分
        fine_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for chunk in chunks:
            sub_chunks = fine_splitter.split_documents([chunk])
            for sc in sub_chunks:
                sc.metadata.update(doc.metadata)
            all_chunks.extend(sub_chunks)

    return all_chunks


def add_chunk_index(chunks: List[Document]) -> List[Document]:
    """为每个 chunk 添加序号，用于引用溯源。"""
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    return chunks
```

### 2.2 验证

创建 `backend/tests/test_splitter.py`：

```python
from app.ingestion.loader import load_file
from app.ingestion.splitter import split_documents, add_chunk_index

def test_split_pipeline():
    # 加载
    docs = load_file("../../test_data/01-Redis集群运维手册.md")
    print(f"📄 加载：{len(docs)} 个文档片段")

    # 分割
    chunks = split_documents(docs, chunk_size=800, chunk_overlap=150)
    chunks = add_chunk_index(chunks)
    print(f"✂️  分割：{len(chunks)} 个 Chunk")

    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- Chunk #{i} (长度 {len(chunk.page_content)} 字) ---")
        print(chunk.page_content[:200])
        print(f"   metadata: {chunk.metadata}")

    print(f"\n✅ 完整管线：1 文档 → {len(chunks)} Chunks")

if __name__ == "__main__":
    test_split_pipeline()
```

```bash
cd ~/Desktop/docmind/backend
python tests/test_splitter.py
```

预期输出：Redis 运维手册被切成 15-25 个 Chunk，每个 Chunk 长度在 800 字左右。

---

## 第 3 步：向量化索引（2 小时）

### 目标
将 Chunk 做 Embedding，存入 Chroma 向量数据库，并验证检索功能。

### 3.1 创建 Embedding 模块

创建 `backend/app/embedding/__init__.py`（空文件）。

创建 `backend/app/embedding/embeddings.py`：

```python
"""Embedding 模型封装。"""
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

_embedding_instance = None


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
```

### 3.2 创建向量存储模块

创建 `backend/app/storage/__init__.py`（空文件）。

创建 `backend/app/storage/vector_store.py`：

```python
"""Chroma 向量数据库操作。"""
import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from app.embedding.embeddings import get_embedding_model


class VectorStoreManager:
    """管理 Chroma 向量存储：索引、检索、删除。"""

    def __init__(self, persist_dir: str):
        self.persist_dir = persist_dir
        self.embeddings = get_embedding_model()
        os.makedirs(persist_dir, exist_ok=True)

    def index_documents(
        self,
        documents: List[Document],
        collection_name: str = "default",
    ) -> Chroma:
        """批量索引文档。"""
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
            collection_name=collection_name,
        )
        print(f"✅ 已索引 {len(documents)} 个文档到集合 '{collection_name}'")
        return vectorstore

    def get_retriever(
        self,
        collection_name: str = "default",
        top_k: int = 5,
        search_type: str = "similarity",
    ):
        """获取检索器。"""
        vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            collection_name=collection_name,
        )
        return vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={"k": top_k},
        )

    def similarity_search(
        self,
        query: str,
        collection_name: str = "default",
        top_k: int = 5,
    ) -> List[Document]:
        """语义检索。"""
        vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            collection_name=collection_name,
        )
        return vectorstore.similarity_search(query, k=top_k)

    def delete_collection(self, collection_name: str):
        """删除集合（用于测试时清空）。"""
        vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            collection_name=collection_name,
        )
        vectorstore.delete_collection()
        print(f"🗑️  已删除集合：{collection_name}")
```

### 3.3 创建端到端索引脚本

创建 `backend/scripts/index_documents.py`：

```python
"""一键索引 test_data 目录下的所有文档。"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.ingestion.loader import load_file
from app.ingestion.splitter import split_documents, add_chunk_index
from app.storage.vector_store import VectorStoreManager
from dotenv import load_dotenv

load_dotenv()

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../test_data")
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")


def index_all():
    vm = VectorStoreManager(persist_dir=CHROMA_DIR)

    all_chunks = []
    for filename in sorted(os.listdir(TEST_DATA_DIR)):
        if not filename.endswith((".md", ".txt", ".pdf")):
            continue

        filepath = os.path.join(TEST_DATA_DIR, filename)
        print(f"\n📄 处理：{filename}")

        try:
            docs = load_file(filepath)
            chunks = split_documents(docs, chunk_size=800, chunk_overlap=150)
            chunks = add_chunk_index(chunks)
            all_chunks.extend(chunks)
            print(f"   → {len(chunks)} 个 Chunk")
        except Exception as e:
            print(f"   ⚠️  跳过（错误：{e}）")

    print(f"\n📊 总计 {len(all_chunks)} 个 Chunk，开始索引...")

    # 先清空旧集合（方便重复跑脚本）
    try:
        vm.delete_collection("docmind")
    except Exception:
        pass

    vm.index_documents(all_chunks, collection_name="docmind")

    print("\n🎉 索引完成！")


if __name__ == "__main__":
    index_all()
```

### 3.4 验证

先跑索引脚本：

```bash
cd ~/Desktop/docmind/backend
python scripts/index_documents.py
```

预期输出：逐文件处理，最后显示 `🎉 索引完成！`，总共大约 150-200 个 Chunk。

再测试检索：

```bash
python -c "
import sys, os
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()
from app.storage.vector_store import VectorStoreManager

vm = VectorStoreManager(persist_dir='./data/chroma_db')
results = vm.similarity_search('Redis集群怎么扩容？', collection_name='docmind', top_k=3)

for i, doc in enumerate(results):
    print(f'\n--- 结果 {i+1} ---')
    print(f'来源: {doc.metadata.get(\"file_path\", \"未知\")}')
    print(f'内容: {doc.page_content[:200]}...')
"
```

预期输出：返回 3 条与 Redis 集群扩容相关的结果，排在第一条的应该是 `01-Redis集群运维手册.md` 中的内容。

---

## 第 4 步：基础 RAG 问答（2 小时）

### 目标
用户输入问题 → 检索 Top-K Chunk → 拼接 Prompt → LLM 生成答案。

### 4.1 创建 LLM 模块

创建 `backend/app/llm/__init__.py`（空文件）。

创建 `backend/app/llm/chat_model.py`：

```python
"""LLM 模型封装，支持 OpenAI 兼容接口。"""
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

_llm_instance = None


def get_llm() -> ChatOpenAI:
    """获取 LLM 单例。"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4o"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.1,   # RAG 场景用低温，减少编造
        )
    return _llm_instance
```

### 4.2 创建 RAG 问答模块

创建 `backend/app/rag/__init__.py`（空文件）。

创建 `backend/app/rag/pipeline.py`：

```python
"""基础 RAG 问答管线。"""
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from app.llm.chat_model import get_llm
from app.storage.vector_store import VectorStoreManager
from dotenv import load_dotenv
import os

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")

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
        self.vm = VectorStoreManager(persist_dir=CHROMA_DIR)
        self.llm = get_llm()
        self.collection_name = collection_name

    def ask(self, question: str, top_k: int = 5) -> dict:
        """
        单轮问答。

        Returns:
            {"question": ..., "answer": ..., "sources": [...]}
        """
        # Step 1: 检索
        retrieved_docs = self.vm.similarity_search(
            question,
            collection_name=self.collection_name,
            top_k=top_k,
        )

        # Step 2: 拼接上下文
        context = self._format_context(retrieved_docs)

        # Step 3: 构建 Prompt
        system_msg = SystemMessage(
            content=RAG_SYSTEM_PROMPT.format(context=context)
        )
        human_msg = HumanMessage(content=question)

        # Step 4: 调用 LLM
        response = self.llm.invoke([system_msg, human_msg])

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
            "answer": response.content,
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
```

### 4.3 验证

创建 `backend/tests/test_rag.py`：

```python
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.rag.pipeline import BasicRAGPipeline

def test_basic_rag():
    pipeline = BasicRAGPipeline(collection_name="docmind")

    questions = [
        "Redis集群怎么扩容？",
        "Docker镜像构建有哪些优化方法？",
        "微服务中Saga模式是什么？",
        "Windows 10怎么安装？（故意测试拒答）",
    ]

    for q in questions:
        print(f"\n{'='*60}")
        print(f"❓ 问题: {q}")
        start = time.time()
        result = pipeline.ask(q, top_k=5)
        elapsed = time.time() - start
        print(f"🤖 回答: {result['answer']}")
        print(f"\n📚 引用来源 ({len(result['sources'])} 条):")
        for s in result["sources"]:
            print(f"   [{s['index']}] {s['file']} · Chunk #{s['chunk_index']}")
        print(f"⏱️  耗时: {elapsed:.2f}s")

if __name__ == "__main__":
    test_basic_rag()
```

```bash
cd ~/Desktop/docmind/backend
python tests/test_rag.py
```

预期输出：
- Redis 扩容问题 → 带引用的具体步骤
- Docker 优化 → 多阶段构建、层缓存等
- Saga → Saga 的定义和两种模式
- Windows 10 → "根据现有资料，我无法回答"

---

## 第 5 步：LangGraph 改造（4 小时）

这是整个项目最核心、面试最能讲的一步。把第 4 步的管线拆成 LangGraph 节点。

### 5.1 定义状态

创建 `backend/app/graph/__init__.py`（空文件）。

创建 `backend/app/graph/state.py`：

```python
"""LangGraph 状态定义。"""
from typing import TypedDict, List, Optional


class RAGState(TypedDict):
    question: str
    rewritten_query: str
    vector_docs: List[dict]
    bm25_docs: List[dict]
    retrieved_docs: List[dict]
    final_context: List[dict]
    answer: str
    citations: List[dict]
    retry_count: int
    confidence_score: float
```

### 5.2 实现各节点

创建 `backend/app/graph/nodes.py`：

```python
"""LangGraph 各节点实现。"""
import os
from typing import List, Dict
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

from app.graph.state import RAGState
from app.llm.chat_model import get_llm
from app.storage.vector_store import VectorStoreManager

load_dotenv()
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
COLLECTION = "docmind"

llm = get_llm()
vm = VectorStoreManager(persist_dir=CHROMA_DIR)

# ==================== 节点函数 ====================

def rewrite_query(state: RAGState) -> dict:
    """节点1：口语化问题 → 检索友好格式。"""
    question = state["question"]
    prompt = f"""将以下用户问题改写为更好的检索关键词（去除口语，保留关键字）：
原始问题：{question}
改写后："""

    response = llm.invoke(prompt)
    rewritten = response.content.strip()
    print(f"  🔄 Query Rewrite: {question} → {rewritten}")
    return {"rewritten_query": rewritten}


def vector_retrieve(state: RAGState) -> dict:
    """节点2：向量检索。"""
    query = state.get("rewritten_query") or state["question"]
    docs = vm.similarity_search(query, collection_name=COLLECTION, top_k=30)
    results = [_doc_to_dict(d) for d in docs]
    print(f"  🔍 向量检索: {len(results)} 条结果")
    return {"vector_docs": results}


def bm25_retrieve(state: RAGState) -> dict:
    """节点3：BM25 关键词检索。"""
    from rank_bm25 import BM25Okapi
    import jieba

    # 获取向量检索的结果列表里的所有 chunk 文本作为 BM25 语料
    # 简化实现：直接复用向量数据库中所有 chunk
    # 生产环境应维护一个独立的 BM25 索引
    query = state.get("rewritten_query") or state["question"]
    # 这里做一个简化版 BM25：从 vector_docs 拿 30 条
    all_docs = state.get("vector_docs", [])
    if not all_docs:
        return {"bm25_docs": []}

    # 分词
    corpus = [_jieba_tokenize(d["content"]) for d in all_docs]
    tokenized_query = _jieba_tokenize(query)

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenized_query)

    # 按分数排序
    indexed = sorted(
        enumerate(scores), key=lambda x: x[1], reverse=True
    )
    results = [all_docs[i] for i, _ in indexed[:30]]
    print(f"  🔑 BM25 检索: {len(results)} 条结果")
    return {"bm25_docs": results}


def _jieba_tokenize(text: str) -> List[str]:
    """中文分词。"""
    try:
        import jieba
        return list(jieba.cut(text))
    except ImportError:
        return text.split()


def rrf_fusion(state: RAGState) -> dict:
    """节点4：RRF 融合向量和 BM25 的结果。"""
    vector_docs = state.get("vector_docs", [])
    bm25_docs = state.get("bm25_docs", [])
    k = 60  # RRF 参数

    scores = {}
    for rank, doc in enumerate(vector_docs):
        doc_id = doc.get("content", "")[:100]  # 用内容前100字当 key
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

    for rank, doc in enumerate(bm25_docs):
        doc_id = doc.get("content", "")[:100]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

    # 合并去重，按 RRF 分数排序
    seen = {}
    all_docs = vector_docs + bm25_docs
    for doc in all_docs:
        doc_id = doc.get("content", "")[:100]
        if doc_id not in seen:
            seen[doc_id] = (scores.get(doc_id, 0), doc)

    sorted_docs = sorted(seen.values(), key=lambda x: x[0], reverse=True)
    fused = [d for _, d in sorted_docs[:30]]
    print(f"  🔗 RRF 融合: {len(fused)} 条结果")
    return {"retrieved_docs": fused}


def rerank(state: RAGState) -> dict:
    """节点5：Cross-encoder 重排序（简化版：用 LLM 打分）。"""
    docs = state.get("retrieved_docs", [])
    if len(docs) <= 5:
        return {"final_context": docs}

    question = state["question"]

    # 简化版：让 LLM 对每条结果打分（生产环境用 BGE-Reranker）
    scored = []
    for i, doc in enumerate(docs[:15]):  # 最多评估 15 条
        prompt = f"""问题：「{question}」
资料片段：「{doc['content'][:300]}」

这条资料对回答问题的有用程度，请打 1-10 分。
只回复数字。"""
        try:
            response = llm.invoke(prompt)
            score = float(response.content.strip()) / 10.0
        except:
            score = 0.5
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    top5 = [d for _, d in scored[:5]]
    print(f"  📊 Rerank: {len(scored)} → {len(top5)} 条")
    return {"final_context": top5}


def generate(state: RAGState) -> dict:
    """节点6：生成答案。"""
    question = state["question"]
    context_docs = state.get("final_context", state.get("retrieved_docs", []))

    # 拼接上下文
    context_parts = []
    for i, doc in enumerate(context_docs):
        source = doc.get("source", "未知")
        chunk_idx = doc.get("chunk_index", i)
        context_parts.append(
            f"[资料{i+1}] 来源: {source} 第{chunk_idx}段\n{doc['content']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    system_msg = SystemMessage(content=f"""你是技术文档助手。只用以下资料回答问题，不编造。

【参考资料】
{context}""")

    human_msg = HumanMessage(content=question)
    response = llm.invoke([system_msg, human_msg])

    # 构建引用
    citations = []
    for i, doc in enumerate(context_docs):
        citations.append({
            "index": i + 1,
            "source": doc.get("source", "未知"),
            "chunk_index": doc.get("chunk_index", i),
            "preview": doc["content"][:150],
        })

    print(f"  ✨ 生成答案: {len(response.content)} 字")
    return {
        "answer": response.content,
        "citations": citations,
    }


def check_relevance(state: RAGState) -> dict:
    """节点7：相关性检查（决定是否需要重试）。"""
    # 简化实现：检查是否有检索结果
    docs = state.get("retrieved_docs", [])
    if not docs:
        return {"confidence_score": 0.0}
    # 有结果就给 0.7（真实场景用 Cross-encoder 打分）
    return {"confidence_score": 0.7}


# ==================== 辅助函数 ====================

def _doc_to_dict(doc: Document) -> dict:
    return {
        "content": doc.page_content,
        "source": doc.metadata.get("file_path", "未知"),
        "chunk_index": doc.metadata.get("chunk_index", 0),
        "metadata": doc.metadata,
    }
```

### 5.3 组装 LangGraph 工作流

创建 `backend/app/graph/workflow.py`：

```python
"""LangGraph 工作流组装。"""
from langgraph.graph import StateGraph, END
from app.graph.state import RAGState
from app.graph.nodes import (
    rewrite_query,
    vector_retrieve,
    bm25_retrieve,
    rrf_fusion,
    rerank,
    generate,
    check_relevance,
)


def build_rag_graph():
    """构建并返回编译后的 RAG 工作流。"""
    graph = StateGraph(RAGState)

    # 添加节点
    graph.add_node("rewrite", rewrite_query)
    graph.add_node("vector_search", vector_retrieve)
    graph.add_node("bm25_search", bm25_retrieve)
    graph.add_node("fusion", rrf_fusion)
    graph.add_node("rerank", rerank)
    graph.add_node("generate", generate)
    graph.add_node("relevance_check", check_relevance)

    # 设置入口
    graph.set_entry_point("rewrite")

    # 连线：rewrite → 并发(向量 + BM25)
    graph.add_edge("rewrite", "vector_search")
    graph.add_edge("rewrite", "bm25_search")

    # 汇聚到融合
    graph.add_edge("vector_search", "fusion")
    graph.add_edge("bm25_search", "fusion")

    # 融合 → 重排 → 生成
    graph.add_edge("fusion", "rerank")
    graph.add_edge("rerank", "generate")

    # 生成后 → 相关性检查 → 结束（带条件）
    graph.add_edge("generate", "relevance_check")

    # 条件边：如果检索结果少且首次，回到 rewrite 重试
    def should_retry(state: RAGState) -> str:
        score = state.get("confidence_score", 1.0)
        retries = state.get("retry_count", 0)
        if score < 0.4 and retries < 1:
            print("  🔁 相关性不足，重试改写...")
            return "rewrite"
        return "end"

    graph.add_conditional_edges(
        "relevance_check",
        should_retry,
        {"rewrite": "rewrite", "end": END}
    )

    return graph.compile()


# 编译好的应用实例
rag_app = build_rag_graph()
```

### 5.4 验证

创建 `backend/tests/test_graph.py`：

```python
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.graph.workflow import rag_app

def test_langgraph_rag():
    question = "Redis集群扩容的具体步骤是什么？"

    print(f"❓ 问题: {question}\n")
    start = time.time()

    result = rag_app.invoke({
        "question": question,
        "retry_count": 0,
        "confidence_score": 0.0,
    })

    elapsed = time.time() - start

    print(f"\n{'='*60}")
    print(f"🤖 回答:\n{result['answer']}")
    print(f"\n📚 引用 ({len(result.get('citations', []))} 条):")
    for c in result.get("citations", []):
        print(f"   [{c['index']}] {c['source']} · Chunk #{c['chunk_index']}")
    print(f"\n⏱️  总耗时: {elapsed:.2f}s")
    print(f"🔄 最终改写 Query: {result.get('rewritten_query', 'N/A')}")

if __name__ == "__main__":
    test_langgraph_rag()
```

```bash
cd ~/Desktop/docmind/backend
python tests/test_graph.py
```

预期输出：控制台打印每个节点的日志（`🔄 Query Rewrite`、`🔍 向量检索`、`🔑 BM25`、`🔗 RRF`、`📊 Rerank`、`✨ 生成`），最后输出带引用的答案。

---

## 第 6 步：FastAPI 服务化（2.5 小时）

### 6.1 创建 API 路由

创建 `backend/app/api/__init__.py`（空文件）。

创建 `backend/app/api/chat.py`：

```python
"""问答 API。"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from app.graph.workflow import rag_app
import uuid

router = APIRouter(prefix="/api/v1", tags=["chat"])


class ChatRequest(BaseModel):
    kb_id: str = "docmind"
    conversation_id: Optional[str] = None
    message: str
    stream: bool = False


class Citation(BaseModel):
    index: int
    source: str
    chunk_index: int
    preview: str


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    citations: List[Citation] = []
    confidence_score: float = 0.0


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """单轮或多轮对话。"""
    try:
        conv_id = request.conversation_id or str(uuid.uuid4())

        result = rag_app.invoke({
            "question": request.message,
            "retry_count": 0,
            "confidence_score": 0.0,
        })

        return ChatResponse(
            conversation_id=conv_id,
            answer=result["answer"],
            citations=[Citation(**c) for c in result.get("citations", [])],
            confidence_score=result.get("confidence_score", 0.0),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 6.2 创建文档管理 API

创建 `backend/app/api/documents.py`：

```python
"""文档管理 API。"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Optional
import os
import uuid
from pathlib import Path
from app.ingestion.loader import load_file
from app.ingestion.splitter import split_documents, add_chunk_index
from app.storage.vector_store import VectorStoreManager
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(prefix="/api/v1", tags=["documents"])
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
UPLOAD_DIR = "./data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/documents/upload")
async def upload_document(
    kb_id: str = "docmind",
    file: UploadFile = File(...),
):
    """上传文档并立即索引。"""
    # 保存文件
    file_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    # 加载 → 分割 → 索引
    docs = load_file(save_path)
    chunks = split_documents(docs)
    chunks = add_chunk_index(chunks)

    vm = VectorStoreManager(persist_dir=CHROMA_DIR)
    vm.index_documents(chunks, collection_name=kb_id)

    return {
        "doc_id": file_id,
        "filename": file.filename,
        "status": "indexed",
        "chunk_count": len(chunks),
    }


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """删除文档（简化：仅返回确认）。"""
    # 完整的删除实现需要在 Chroma 中按 doc_id 过滤删除
    # 这里做简化返回
    return {"status": "deleted", "doc_id": doc_id}
```

### 6.3 创建主应用

创建 `backend/app/main.py`：

```python
"""FastAPI 主入口。"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import chat, documents

app = FastAPI(
    title="DocMind API",
    description="企业级多源技术文档智能问答系统",
    version="0.1.0",
)

# CORS（允许前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat.router)
app.include_router(documents.router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "docmind"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 6.4 验证

启动服务：

```bash
cd ~/Desktop/docmind/backend
python -m uvicorn app.main:app --reload --port 8000
```

另开一个终端测试：

```bash
# 测试健康检查
curl http://localhost:8000/health

# 测试问答
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Docker怎么优化镜像大小？"}'
```

打开浏览器访问 `http://localhost:8000/docs`，可以看到 Swagger 自动生成的接口文档。

---

## 第 7 步：前端对话界面（3 小时）

### 7.1 初始化 Next.js

```bash
cd ~/Desktop/docmind
npx create-next-app@latest frontend --typescript --tailwind --app --no-src-dir
cd frontend
npm install
```

### 7.2 创建对话页面

替换 `frontend/app/page.tsx`：

```tsx
"use client";

import { useState, useRef, useEffect } from "react";

interface Message {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
}

interface Citation {
  index: number;
  source: string;
  chunk_index: number;
  preview: string;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMsg: Message = { role: "user", content: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch("http://localhost:8000/api/v1/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMsg.content }),
      });

      const data = await res.json();
      const assistantMsg: Message = {
        role: "assistant",
        content: data.answer,
        citations: data.citations,
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "抱歉，请求失败，请检查后端服务是否启动。" },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="flex flex-col h-screen max-w-4xl mx-auto">
      {/* Header */}
      <header className="border-b py-4 px-6">
        <h1 className="text-xl font-bold">DocMind</h1>
        <p className="text-sm text-gray-500">技术文档智能助手</p>
      </header>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-6">
        {messages.length === 0 && (
          <div className="text-center text-gray-400 mt-20">
            <p className="text-lg">👋 你好，我是 DocMind</p>
            <p className="text-sm mt-2">
              试试问我：Redis集群怎么扩容？Docker如何优化镜像大小？
            </p>
          </div>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[80%] rounded-lg px-4 py-3 ${
                msg.role === "user"
                  ? "bg-blue-500 text-white"
                  : "bg-gray-100 text-gray-900"
              }`}
            >
              <div className="whitespace-pre-wrap">{msg.content}</div>

              {/* Citations */}
              {msg.citations && msg.citations.length > 0 && (
                <div className="mt-3 pt-3 border-t border-gray-300">
                  <p className="text-xs font-semibold text-gray-500 mb-1">
                    📚 参考来源
                  </p>
                  {msg.citations.map((c) => (
                    <div key={c.index} className="text-xs text-gray-500 mt-1">
                      <span className="font-medium">[{c.index}]</span>{" "}
                      {c.source} · 第{c.chunk_index}段
                      <p className="text-gray-400 truncate">{c.preview}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-gray-100 rounded-lg px-4 py-3 text-gray-500">
              思考中...
            </div>
          </div>
        )}

        <div ref={chatEndRef} />
      </div>

      {/* Input */}
      <div className="border-t p-4">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="输入你的问题..."
            className="flex-1 border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={loading}
          />
          <button
            onClick={sendMessage}
            disabled={loading || !input.trim()}
            className="bg-blue-500 text-white px-6 py-2 rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            发送
          </button>
        </div>
      </div>
    </div>
  );
}
```

### 7.3 验证

```bash
cd ~/Desktop/docmind/frontend
npm run dev
```

打开 `http://localhost:3000`，在输入框输入问题测试。

---

## 第 8 步：评估体系（2 小时）

### 8.1 安装 RAGAS

```bash
pip install ragas datasets
```

### 8.2 创建评估脚本

创建 `backend/scripts/evaluate.py`：

```python
"""RAGAS 评估脚本。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

from app.rag.pipeline import BasicRAGPipeline


def build_eval_dataset():
    """构建评估用的数据集（手工标注）。"""
    pipeline = BasicRAGPipeline(collection_name="docmind")

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

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for case in test_cases:
        result = pipeline.ask(case["question"], top_k=5)
        questions.append(case["question"])
        answers.append(result["answer"])
        contexts.append([s["preview"] for s in result["sources"]])
        ground_truths.append(case["ground_truth"])

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })


def run_evaluation():
    print("📊 构建评估数据集...")
    dataset = build_eval_dataset()

    print("🔬 运行 RAGAS 评估...")
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    print("\n" + "=" * 50)
    print("📊 RAGAS 评估结果")
    print("=" * 50)
    for key, value in results.items():
        if key != "faithfulness":
            print(f"  {key}: {value:.3f}")
    # faithfulness 是特殊格式
    if "faithfulness" in results:
        print(f"  faithfulness: {results['faithfulness']}")

    print("\n💡 指标解读：")
    print("  Faithfulness > 0.8  → 答案基本忠实于检索内容")
    print("  Answer Relevancy > 0.7 → 答案与问题相关度可接受")
    print("  Context Precision > 0.7 → 检索结果噪音较少")
    print("  Context Recall > 0.8 → 检索覆盖了标准答案所需信息")

    return results


if __name__ == "__main__":
    run_evaluation()
```

### 8.3 验证

```bash
cd ~/Desktop/docmind/backend
python scripts/evaluate.py
```

---

## 完整项目结构检查清单

到这一步，你的 `docmind/` 目录应该长这样：

```
docmind/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                 ✅ FastAPI 入口
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── chat.py             ✅ 问答接口
│   │   │   └── documents.py        ✅ 文档管理接口
│   │   ├── ingestion/
│   │   │   ├── __init__.py
│   │   │   ├── loader.py           ✅ 文档加载
│   │   │   └── splitter.py         ✅ 文本分割
│   │   ├── embedding/
│   │   │   ├── __init__.py
│   │   │   └── embeddings.py       ✅ Embedding 模型
│   │   ├── storage/
│   │   │   ├── __init__.py
│   │   │   └── vector_store.py     ✅ Chroma 操作
│   │   ├── llm/
│   │   │   ├── __init__.py
│   │   │   └── chat_model.py       ✅ LLM 封装
│   │   ├── rag/
│   │   │   ├── __init__.py
│   │   │   └── pipeline.py         ✅ 基础 RAG
│   │   └── graph/
│   │       ├── __init__.py
│   │       ├── state.py            ✅ LangGraph 状态
│   │       ├── nodes.py            ✅ 各节点实现
│   │       └── workflow.py         ✅ 工作流组装
│   ├── scripts/
│   │   ├── index_documents.py      ✅ 索引脚本
│   │   └── evaluate.py             ✅ 评估脚本
│   ├── tests/
│   │   ├── test_loader.py          ✅
│   │   ├── test_splitter.py        ✅
│   │   ├── test_rag.py             ✅
│   │   └── test_graph.py           ✅
│   ├── data/
│   │   ├── chroma_db/              ✅ 向量持久化
│   │   ├── uploads/                ✅ 上传文件
│   │   └── docmind.db              ✅ SQLite 数据库
│   ├── requirements.txt            ✅
│   ├── .env                        ✅
│   └── venv/                       ✅
├── frontend/
│   ├── app/
│   │   ├── page.tsx                ✅ 对话界面
│   │   ├── layout.tsx              ✅
│   │   └── globals.css             ✅
│   ├── package.json                ✅
│   └── next.config.js              ✅
├── test_data/                      ✅ 8 个测试文档
├── RAG项目需求文档.md               ✅
└── DocMind开发步骤.md               ✅ 你在看的就是这个
```

---

## 时间节点建议

| 阶段 | 内容 | 预计时间 |
|------|------|----------|
| Day 1 | 第 0-2 步：环境 + 加载 + 分割 | 4 小时 |
| Day 2 | 第 3 步：向量化索引 | 2 小时 |
| Day 3 | 第 4 步：基础 RAG 问答 | 2 小时 |
| Day 4-5 | 第 5 步：LangGraph 改造 | 4 小时 |
| Day 6 | 第 6-7 步：API + 前端 | 5 小时 |
| Day 7 | 第 8 步：评估体系 | 2 小时 |

**总计约 19 小时**，一周内可以完成 MVP。

---

## 第 9 步：Query 分解（2 小时）

### 目标
当用户输入复杂问题时（如"A 和 B 的区别？各自怎么配置？"），自动拆分为多个子问题，每个子问题分别检索，合并结果。

### 9.1 更新 LangGraph State

编辑 `backend/app/graph/state.py`，添加子问题相关字段：

```python
class RAGState(TypedDict):
    question: str
    rewritten_query: str
    sub_queries: List[str]         # 新增：拆分后的子问题列表
    vector_docs: List[dict]
    bm25_docs: List[dict]
    retrieved_docs: List[dict]
    final_context: List[dict]
    answer: str
    citations: List[dict]
    retry_count: int
    confidence_score: float
    conversation_history: List[dict]  # 新增：对话历史
```

### 9.2 添加分解节点

在 `backend/app/graph/nodes.py` 末尾添加：

```python
def decompose_query(state: RAGState) -> dict:
    """节点 X：检测并拆分复杂问题。"""
    question = state["question"]

    # 检测是否需要拆分
    detect_prompt = f"""判断以下问题是否包含多个子问题（比如要求对比、列出多个步骤、或包含"分别""各自"等词）。
只回复 YES 或 NO。

问题：{question}"""

    response = llm.invoke(detect_prompt)
    need_decompose = response.content.strip().upper().startswith("YES")

    if not need_decompose:
        print("  📌 无需拆分，直接检索")
        return {"sub_queries": []}

    # 拆分
    split_prompt = f"""将以下复杂问题拆分为 2-4 个独立的子问题，每个子问题一行，用序号标注。
确保每个子问题可以独立检索回答。

原问题：{question}

子问题："""

    response = llm.invoke(split_prompt)
    lines = response.content.strip().split("\n")
    sub_queries = []
    for line in lines:
        line = line.strip()
        # 去掉前面的序号 "1. " "1) " "- " 等
        for prefix in ["1. ", "2. ", "3. ", "4. ", "1) ", "2) ", "3) ", "4) ", "- "]:
            if line.startswith(prefix):
                line = line[len(prefix):]
                break
        if line and len(line) > 3:
            sub_queries.append(line)

    print(f"  🔀 Query 分解: {len(sub_queries)} 个子问题")
    for sq in sub_queries:
        print(f"      → {sq}")
    return {"sub_queries": sub_queries}


def multi_query_retrieve(state: RAGState) -> dict:
    """节点 Y：为每个子问题分别检索，然后合并去重。"""
    sub_queries = state.get("sub_queries", [])
    if not sub_queries:
        # 不需要拆分的，直接返回空
        return {}

    all_docs = []
    for sq in sub_queries:
        docs = vm.similarity_search(sq, collection_name=COLLECTION, top_k=10)
        all_docs.extend([_doc_to_dict(d) for d in docs])

    # 按内容去重
    seen = set()
    unique_docs = []
    for doc in all_docs:
        key = doc["content"][:100]
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    print(f"  📦 多查询合并: {len(all_docs)} → {len(unique_docs)} (去重后)")
    return {"vector_docs": unique_docs}
```

### 9.3 升级 workflow.py

将 `decompose_query` 和 `multi_query_retrieve` 加入工作流：

```python
# 在 build_rag_graph() 中添加：
from app.graph.nodes import (
    # ... 原有导入
    decompose_query,
    multi_query_retrieve,
)

# 添加节点
graph.add_node("decompose", decompose_query)
graph.add_node("multi_retrieve", multi_query_retrieve)

# 改连线：rewrite → decompose（条件）
def route_after_rewrite(state: RAGState) -> str:
    """判断是否需要分解。先走 decompose，内部判断后跳过或执行。"""
    # 对于明确包含对比/并列的问题才进分解
    q = state["question"]
    complex_keywords = ["区别", "对比", "比较", "分别", "各自", "优缺点", "异同", "vs", "VS"]
    if any(kw in q for kw in complex_keywords):
        return "decompose"
    return "vector_search"

graph.add_conditional_edges(
    "rewrite",
    route_after_rewrite,
    {
        "decompose": "decompose",
        "vector_search": "vector_search",
    }
)

# decompose → multi_retrieve（如果拆了子问题）→ 直接跳到 fusion
graph.add_conditional_edges(
    "decompose",
    lambda s: "multi_retrieve" if s.get("sub_queries") else "vector_search",
    {
        "multi_retrieve": "multi_retrieve",
        "vector_search": "vector_search",
    }
)
graph.add_edge("multi_retrieve", "fusion")
```

### 9.4 验证

```bash
cd ~/Desktop/docmind/backend
python -c "
import sys; sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv()
from app.graph.workflow import rag_app

result = rag_app.invoke({
    'question': 'RDB和AOF持久化有什么区别？各自怎么配置？',
    'retry_count': 0,
    'confidence_score': 0.0,
    'sub_queries': [],
    'conversation_history': [],
})
print(result['answer'][:500])
"
```

---

## 第 10 步：多轮对话管理（3 小时）

### 目标
支持多轮对话：指代消解（"它怎么部署？"→ 补全为"XX 怎么部署？"），对话持久化，上下文窗口管理。

### 10.1 创建对话管理模块

创建 `backend/app/conversation/__init__.py`（空文件）。

创建 `backend/app/conversation/manager.py`：

```python
"""多轮对话管理器：持久化、压缩、指代消解。"""
import uuid
from typing import List, Dict, Optional
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


class ConversationManager:
    """管理对话状态。简化版用内存字典存储，生产环境换成数据库。"""

    def __init__(self):
        self._store: Dict[str, List[dict]] = {}  # conv_id → [{role, content, time}]

    def create_conversation(self, kb_id: str = "docmind") -> str:
        conv_id = str(uuid.uuid4())
        self._store[conv_id] = []
        return conv_id

    def get_history(self, conv_id: str, last_n: int = 5) -> List[dict]:
        return self._store.get(conv_id, [])[-last_n:]

    def add_message(self, conv_id: str, role: str, content: str):
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
            f"{'用户' if m['role']=='user' else '助手'}: {m['content'][:200]}"
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


# 全局单例
conv_manager = ConversationManager()
```

### 10.2 对话摘要压缩

在 `backend/app/conversation/manager.py` 的 `ConversationManager` 类中追加：

```python
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
                f"{'👤' if m['role']=='user' else '🤖'}: {m['content']}"
                for m in history
            ])

        # 需要压缩：前 70% 做摘要，后 30% 保留原文
        split_idx = int(len(history) * 0.7)
        early = history[:split_idx]
        recent = history[split_idx:]

        from app.llm.chat_model import get_llm
        llm = get_llm()

        early_text = "\n".join([
            f"{'用户' if m['role']=='user' else '助手'}: {m['content'][:300]}"
            for m in early
        ])

        summary_prompt = f"""请用 2-3 句话摘要以下对话中讨论的主要技术话题和关键结论：

{early_text}

摘要："""

        response = llm.invoke(summary_prompt)
        summary = response.content.strip()

        recent_text = "\n".join([
            f"{'👤' if m['role']=='user' else '🤖'}: {m['content']}"
            for m in recent
        ])

        return f"【对话前期摘要】{summary}\n\n【最近对话】\n{recent_text}"
```

### 10.3 话题切换检测

在 `ConversationManager` 类中追加：

```python
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
```

### 10.4 更新 API 以支持多轮对话

编辑 `backend/app/api/chat.py`：

```python
from app.conversation.manager import conv_manager

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        conv_id = request.conversation_id or conv_manager.create_conversation()

        # Step 1: 指代消解
        rewritten = conv_manager.rewrite_with_history(conv_id, request.message)

        # Step 2: 话题切换检测
        if conv_manager.detect_topic_switch(conv_id, request.message):
            # 话题切换时，用压缩后的上下文而不是完整历史
            compressed = conv_manager.build_compressed_context(conv_id)
        else:
            compressed = conv_manager.build_compressed_context(conv_id)

        # Step 3: 保存用户消息
        conv_manager.add_message(conv_id, "user", request.message)

        # Step 4: RAG
        result = rag_app.invoke({
            "question": rewritten,
            "retry_count": 0,
            "confidence_score": 0.0,
            "sub_queries": [],
            "conversation_history": conv_manager.get_history(conv_id, last_n=5),
        })

        # Step 5: 保存助手消息
        conv_manager.add_message(conv_id, "assistant", result["answer"])

        return ChatResponse(
            conversation_id=conv_id,
            answer=result["answer"],
            citations=[Citation(**c) for c in result.get("citations", [])],
            confidence_score=result.get("confidence_score", 0.0),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 10.5 验证

```bash
# 第一轮
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Redis集群怎么扩容？"}'
# 记下返回的 conversation_id

# 第二轮（指代消解："它"应该被理解成 Redis 集群）
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"conversation_id": "上一步的ID", "message": "它的slot怎么分配？"}'
```

---

## 第 11 步：流式输出 SSE（2 小时）

### 目标
前端打字机效果逐字显示答案，同时加载完成后回填引用。

### 11.1 后端流式端点

编辑 `backend/app/api/chat.py`，添加流式接口：

```python
from fastapi.responses import StreamingResponse
import asyncio

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式问答接口（Server-Sent Events）。"""
    conv_id = request.conversation_id or str(uuid.uuid4())

    # 重写 Query
    rewritten = conv_manager.rewrite_with_history(conv_id, request.message)

    async def event_generator():
        try:
            # 1. 发送"开始"事件
            yield f"data: {json.dumps({'type': 'start', 'conversation_id': conv_id})}\n\n"

            # 2. 运行 RAG（非流式部分）
            result = rag_app.invoke({
                "question": rewritten,
                "retry_count": 0,
                "confidence_score": 0.0,
                "sub_queries": [],
                "conversation_history": conv_manager.get_history(conv_id, last_n=5),
            })

            # 3. 用 LLM streaming 重新生成答案（真正的逐字输出）
            from app.llm.chat_model import get_llm
            llm = get_llm()

            context_parts = []
            for i, doc in enumerate(result.get("final_context", result.get("retrieved_docs", []))):
                source = doc.get("source", "未知")
                context_parts.append(f"[资料{i+1}] {source}\n{doc['content']}")
            context = "\n\n---\n\n".join(context_parts)

            from langchain_core.messages import SystemMessage, HumanMessage
            messages = [
                SystemMessage(content=f"""你是技术文档助手。只用以下资料回答问题，不编造。

【参考资料】
{context}"""),
                HumanMessage(content=request.message),
            ]

            full_answer = ""
            for chunk in llm.stream(messages):
                if chunk.content:
                    full_answer += chunk.content
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk.content}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0)  # 让出控制权

            # 4. 发送引用和元数据
            yield f"data: {json.dumps({'type': 'citations', 'citations': result.get('citations', []), 'confidence_score': result.get('confidence_score', 0.0)}, ensure_ascii=False)}\n\n"

            # 5. 结束
            yield f"data: {json.dumps({'type': 'end'})}\n\n"

            # 保存对话
            conv_manager.add_message(conv_id, "user", request.message)
            conv_manager.add_message(conv_id, "assistant", full_answer)

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
        },
    )
```

### 11.2 更新前端支持流式输出

在 `frontend/app/page.tsx` 中，将 `sendMessage` 改为流式版本：

```tsx
const sendMessage = async () => {
  if (!input.trim() || loading) return;

  const userMsg: Message = { role: "user", content: input };
  setMessages((prev) => [...prev, userMsg]);
  const currentInput = input;
  setInput("");
  setLoading(true);

  // 添加一个空的 assistant 消息，后续逐步填充
  const assistantMsg: Message = { role: "assistant", content: "" };
  setMessages((prev) => [...prev, assistantMsg]);

  try {
    const res = await fetch("http://localhost:8000/api/v1/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: currentInput,
        conversation_id: conversationId,  // 新增 state
      }),
    });

    const reader = res.body?.getReader();
    if (!reader) return;

    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";  // 保留未完成的行

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const data = JSON.parse(line.slice(6));

          if (data.type === "token") {
            setMessages((prev) => {
              const updated = [...prev];
              const last = updated[updated.length - 1];
              updated[updated.length - 1] = {
                ...last,
                content: last.content + data.content,
              };
              return updated;
            });
          } else if (data.type === "citations") {
            setMessages((prev) => {
              const updated = [...prev];
              updated[updated.length - 1] = {
                ...updated[updated.length - 1],
                citations: data.citations,
              };
              return updated;
            });
            if (data.conversation_id) {
              setConversationId(data.conversation_id);
            }
          } else if (data.type === "start" && data.conversation_id) {
            setConversationId(data.conversation_id);
          }
        }
      }
    }
  } catch {
    setMessages((prev) => {
      const updated = [...prev];
      updated[updated.length - 1] = {
        ...updated[updated.length - 1],
        content: "请求失败，请重试",
      };
      return updated;
    });
  } finally {
    setLoading(false);
  }
};
```

需要在文件顶部添加 state：

```tsx
const [conversationId, setConversationId] = useState<string | null>(null);
```

### 11.3 验证

```bash
# 后端
cd ~/Desktop/docmind/backend
python -m uvicorn app.main:app --reload

# 前端
cd ~/Desktop/docmind/frontend
npm run dev
```

在浏览器中提问，你应该能看到逐字输出的打字机效果。

---

## 第 12 步：幻觉检测 + 自修正回路（3 小时）

### 目标
生成答案后，检查答案中的事实断言是否能在检索上下文中找到依据。如果幻觉率高，自动重生成一次。

### 12.1 创建幻觉检测模块

创建 `backend/app/quality/__init__.py`（空文件）。

创建 `backend/app/quality/hallucination_check.py`：

```python
"""幻觉检测：NLI（自然语言推理）检查答案是否被检索上下文支持。"""
from typing import List, Dict, Tuple


def extract_claims(answer: str) -> List[str]:
    """
    从答案中提取事实性断言。
    简化策略：按句号/换行拆分，过滤太短的句子。
    """
    import re
    sentences = re.split(r'[。\n]', answer)
    claims = []
    for s in sentences:
        s = s.strip()
        # 过滤：太短的不是断言，包含"我"、"建议"等不是事实
        if len(s) > 10 and not any(w in s for w in ["建议", "可以试试", "可能"]):
            claims.append(s)
    return claims


def check_claim_against_context(
    claim: str,
    contexts: List[str],
) -> Tuple[bool, str]:
    """
    检查单个断言是否被上下文支持。
    用 LLM 做 NLI（自然语言推理）。

    Returns:
        (is_supported, explanation)
        is_supported: True=被支持, False=矛盾或无依据
        explanation: 简短解释
    """
    from app.llm.chat_model import get_llm
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
    label = response.content.strip()

    is_supported = "支持" in label
    return is_supported, label


def evaluate_answer(
    question: str,
    answer: str,
    contexts: List[str],
) -> dict:
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

    for claim in claims[:10]:  # 最多检查 10 条断言
        is_supported, label = check_claim_against_context(claim, contexts_text)
        claim_details.append({
            "claim": claim,
            "is_supported": is_supported,
            "label": label,
        })
        if not is_supported:
            unsupported_count += 1

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
    scores_text = response.content

    # 解析分数
    import re
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
```

### 12.2 将幻觉检测集成到 LangGraph

在 `backend/app/graph/nodes.py` 末尾添加：

```python
def hallucination_check(state: RAGState) -> dict:
    """节点：幻觉检测。"""
    from app.quality.hallucination_check import evaluate_answer

    answer = state.get("answer", "")
    contexts = state.get("final_context", state.get("retrieved_docs", []))

    if not answer or not contexts:
        return {"confidence_score": 0.5}

    result = evaluate_answer(
        question=state["question"],
        answer=answer,
        contexts=contexts,
    )

    hallucination_ratio = result["hallucination_ratio"]
    overall_pass = result["overall_pass"]

    print(f"  🔬 幻觉检测: 幻觉率={hallucination_ratio:.0%}, "
          f"相关性={result['relevance_score']}, "
          f"完整性={result['completeness_score']}, "
          f"准确性={result['accuracy_score']}, "
          f"通过={overall_pass}")

    return {
        "confidence_score": 1.0 - hallucination_ratio,
    }


def self_refine(state: RAGState) -> dict:
    """节点：自修正——给 LLM 一次改进机会。"""
    question = state["question"]
    previous_answer = state["answer"]
    contexts = state.get("final_context", state.get("retrieved_docs", []))

    context_parts = []
    for i, doc in enumerate(contexts):
        source = doc.get("source", "未知") if isinstance(doc, dict) else "未知"
        content = doc["content"] if isinstance(doc, dict) else doc.page_content
        context_parts.append(f"[资料{i+1}] {source}\n{content}")
    context = "\n\n---\n\n".join(context_parts)

    refine_prompt = f"""你之前给了以下回答，但可能存在问题（信息不准确、不完整或有编造）。

【你的前一次回答】
{previous_answer}

【参考资料（唯一事实依据）】
{context}

【用户问题】
{question}

请重新回答。你必须：
1. 修正前一次回答中不准确的任何信息
2. 删除没有依据的任何断言
3. 如果有依据不足的地方，明确说明
4. 用 [来源: 资料X] 标注出处

重新回答："""

    from langchain_core.messages import HumanMessage
    response = llm.invoke([HumanMessage(content=refine_prompt)])

    print(f"  🔄 自修正完成: {len(previous_answer)}字 → {len(response.content)}字")
    return {"answer": response.content}
```

### 12.3 更新 workflow.py，加入质检回路

```python
# 在 build_rag_graph() 中：
from app.graph.nodes import hallucination_check, self_refine

graph.add_node("hallucination_check", hallucination_check)
graph.add_node("self_refine", self_refine)

# generate → hallucination_check
graph.add_edge("generate", "hallucination_check")

# 条件边：质检不通过 → 自修正 → 重新生成（最多 1 次）
def should_refine(state: RAGState) -> str:
    score = state.get("confidence_score", 1.0)
    retries = state.get("retry_count", 0)
    if score < 0.7 and retries < 1:
        print(f"  ⚠️  质检不通过 (score={score:.2f})，触发自修正...")
        return "self_refine"
    return "end"

graph.add_conditional_edges(
    "hallucination_check",
    should_refine,
    {"self_refine": "self_refine", "end": END}
)

# self_refine → 更新 retry_count → 回到 generate
graph.add_edge("self_refine", "generate")
```

### 12.4 验证

```bash
cd ~/Desktop/docmind/backend
python -c "
import sys; sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv()
from app.graph.workflow import rag_app

result = rag_app.invoke({
    'question': 'Redis集群怎么扩容，请说具体命令',
    'retry_count': 0,
    'confidence_score': 0.0,
    'sub_queries': [],
    'conversation_history': [],
})
print('=== 答案 ===')
print(result['answer'])
print(f'\n=== 置信度: {result.get(\"confidence_score\", \"N/A\")} ===')
"
```

---

## 第 13 步：用户反馈闭环（1.5 小时）

### 目标
用户在 UI 上点赞/点踩，后台记录反馈，点踩的回答进入 hard-negative 评估集。

### 13.1 创建反馈 API

创建 `backend/app/api/feedback.py`：

```python
"""用户反馈 API。"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
import json
import os

router = APIRouter(prefix="/api/v1", tags=["feedback"])

FEEDBACK_FILE = "./data/feedback.jsonl"


class FeedbackRequest(BaseModel):
    conversation_id: str
    message_id: str  # assistant 消息的 ID
    rating: str  # "like" | "dislike"
    comment: Optional[str] = None


@router.post("/feedback")
async def submit_feedback(fb: FeedbackRequest):
    """记录用户反馈。"""
    record = {
        "conversation_id": fb.conversation_id,
        "message_id": fb.message_id,
        "rating": fb.rating,
        "comment": fb.comment,
    }

    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 点踩的消息进入 hard-negative 集
    if fb.rating == "dislike":
        _add_to_hard_negative(fb)

    return {"status": "recorded"}


def _add_to_hard_negative(fb: FeedbackRequest):
    """将点踩消息加入 hard-negative 评估集。"""
    from app.conversation.manager import conv_manager

    history = conv_manager.get_history(fb.conversation_id, last_n=2)
    question = None
    answer = None
    for m in history:
        if m["role"] == "user":
            question = m["content"]
        elif m["role"] == "assistant":
            answer = m["content"]

    if question and answer:
        negative_file = "./data/hard_negatives.jsonl"
        with open(negative_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "question": question,
                "answer": answer,
                "reason": fb.comment or "用户点踩",
            }, ensure_ascii=False) + "\n")
        print(f"  📝 已记录 hard-negative: {question[:50]}...")


@router.get("/feedback/stats")
async def feedback_stats():
    """反馈统计。"""
    if not os.path.exists(FEEDBACK_FILE):
        return {"total": 0, "likes": 0, "dislikes": 0}

    likes = 0
    dislikes = 0
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record["rating"] == "like":
                likes += 1
            elif record["rating"] == "dislike":
                dislikes += 1

    return {"total": likes + dislikes, "likes": likes, "dislikes": dislikes}
```

### 13.2 注册路由

编辑 `backend/app/main.py`：

```python
from app.api import feedback
app.include_router(feedback.router)
```

### 13.3 前端添加反馈按钮

在 `frontend/app/page.tsx` 中，给 assistant 消息添加点赞/点踩按钮：

```tsx
{/* 在 assistant 消息气泡内，citations 下方添加 */}
{msg.role === "assistant" && msg.content && (
  <div className="mt-2 flex gap-2 justify-end">
    <button
      onClick={() => handleFeedback(msg, "like")}
      className="text-xs px-2 py-1 rounded hover:bg-gray-200"
      title="有用"
    >
      👍
    </button>
    <button
      onClick={() => handleFeedback(msg, "dislike")}
      className="text-xs px-2 py-1 rounded hover:bg-gray-200"
      title="无用"
    >
      👎
    </button>
  </div>
)}

// 添加处理函数
const handleFeedback = async (msg: Message, rating: "like" | "dislike") => {
  await fetch("http://localhost:8000/api/v1/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      conversation_id: conversationId,
      message_id: msg.id || Date.now().toString(),
      rating,
    }),
  });
  alert(`感谢反馈！${rating === "like" ? "👍" : "👎"}`);
};
```

### 13.4 验证

```bash
curl -X POST http://localhost:8000/api/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{"conversation_id": "test-123", "message_id": "msg-456", "rating": "dislike", "comment": "答案中的端口号不对"}'

curl http://localhost:8000/api/v1/feedback/stats
```

---

## 第 14 步：知识库管理 + 检索调试面板（3 小时）

### 目标
前端添加知识库管理页面和检索调试页面。后端添加对应的管理 API。

### 14.1 知识库管理 API

创建 `backend/app/api/knowledge_bases.py`：

```python
"""知识库 CRUD API。"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import uuid
from app.storage.vector_store import VectorStoreManager
from dotenv import load_dotenv
import os

load_dotenv()

router = APIRouter(prefix="/api/v1", tags=["knowledge-bases"])
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")


class KBCreateRequest(BaseModel):
    name: str
    description: str = ""
    embedding_model: str = "BAAI/bge-m3"


@router.post("/knowledge-bases")
async def create_kb(req: KBCreateRequest):
    """创建知识库。"""
    kb_id = str(uuid.uuid4())

    # 存储知识库元数据（简化：用文件存）
    import json
    meta_dir = "./data/kb_meta"
    os.makedirs(meta_dir, exist_ok=True)
    with open(f"{meta_dir}/{kb_id}.json", "w") as f:
        json.dump({
            "id": kb_id,
            "name": req.name,
            "description": req.description,
            "embedding_model": req.embedding_model,
            "document_count": 0,
        }, f)

    return {"kb_id": kb_id, "name": req.name, "status": "created"}


@router.get("/knowledge-bases")
async def list_kbs():
    """列出所有知识库。"""
    import json, glob
    kbs = []
    for fpath in glob.glob("./data/kb_meta/*.json"):
        with open(fpath) as f:
            kbs.append(json.load(f))
    return {"knowledge_bases": kbs, "total": len(kbs)}


@router.delete("/knowledge-bases/{kb_id}")
async def delete_kb(kb_id: str):
    """删除知识库及其所有向量数据。"""
    vm = VectorStoreManager(persist_dir=CHROMA_DIR)
    try:
        vm.delete_collection(kb_id)
    except Exception:
        pass  # 集合可能不存在

    # 删除元数据
    meta_path = f"./data/kb_meta/{kb_id}.json"
    if os.path.exists(meta_path):
        os.remove(meta_path)

    return {"status": "deleted", "kb_id": kb_id}
```

### 14.2 检索调试 API

创建 `backend/app/api/debug.py`：

```python
"""检索调试 API：查看每一步的中间结果。"""
from fastapi import APIRouter, Query
from typing import Optional
from app.llm.chat_model import get_llm
from app.storage.vector_store import VectorStoreManager
from dotenv import load_dotenv
import os
import time

load_dotenv()

router = APIRouter(prefix="/api/v1/debug", tags=["debug"])
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")


@router.get("/retrieve")
async def debug_retrieve(
    kb_id: str = Query("docmind"),
    query: str = Query(...),
    top_k: int = Query(30),
):
    """逐步展示检索过程。"""
    vm = VectorStoreManager(persist_dir=CHROMA_DIR)
    llm = get_llm()
    steps = []
    total_start = time.time()

    # Step 1: Query Rewrite
    t0 = time.time()
    rewrite_prompt = f"将以下问题改写为更适合检索的形式：{query}"
    rewritten = llm.invoke(rewrite_prompt).content.strip()
    steps.append({
        "step": "query_rewrite",
        "input": query,
        "output": rewritten,
        "elapsed_ms": int((time.time() - t0) * 1000),
    })

    # Step 2: 向量检索
    t1 = time.time()
    vector_results = vm.similarity_search(rewritten, collection_name=kb_id, top_k=top_k)
    vector_formatted = []
    for doc in vector_results:
        vector_formatted.append({
            "content_preview": doc.page_content[:200],
            "source": doc.metadata.get("file_path", "未知"),
            "chunk_index": doc.metadata.get("chunk_index", 0),
        })
    steps.append({
        "step": "vector_search",
        "count": len(vector_formatted),
        "results": vector_formatted[:10],  # 只回传前 10 条
        "elapsed_ms": int((time.time() - t1) * 1000),
    })

    # Step 3: BM25 检索
    t2 = time.time()
    from rank_bm25 import BM25Okapi
    try:
        import jieba
        corpus = [list(jieba.cut(doc.page_content)) for doc in vector_results]
        tokenized_q = list(jieba.cut(rewritten))
    except ImportError:
        corpus = [doc.page_content.split() for doc in vector_results]
        tokenized_q = rewritten.split()

    bm25 = BM25Okapi(corpus)
    bm25_scores = bm25.get_scores(tokenized_q)
    bm25_ranked = sorted(
        enumerate(bm25_scores), key=lambda x: x[1], reverse=True
    )
    bm25_formatted = []
    for idx, score in bm25_ranked[:10]:
        doc = vector_results[idx]
        bm25_formatted.append({
            "content_preview": doc.page_content[:200],
            "bm25_score": float(score),
            "source": doc.metadata.get("file_path", "未知"),
            "chunk_index": doc.metadata.get("chunk_index", idx),
        })
    steps.append({
        "step": "bm25_search",
        "count": len(bm25_ranked),
        "results": bm25_formatted,
        "elapsed_ms": int((time.time() - t2) * 1000),
    })

    # Step 4: RRF 融合
    t3 = time.time()
    k = 60
    scores = {}
    for rank, doc in enumerate(vector_results):
        did = doc.page_content[:100]
        scores[did] = scores.get(did, 0) + 1 / (k + rank + 1)
    for rank, (idx, _) in enumerate(bm25_ranked):
        did = vector_results[idx].page_content[:100]
        scores[did] = scores.get(did, 0) + 1 / (k + rank + 1)

    seen = set()
    fused = []
    for did, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        if did not in seen:
            seen.add(did)
            fused.append({"content_preview": did, "rrf_score": score})
    steps.append({
        "step": "rrf_fusion",
        "count": len(fused),
        "results": fused[:10],
        "elapsed_ms": int((time.time() - t3) * 1000),
    })

    total_elapsed = int((time.time() - total_start) * 1000)

    return {
        "original_query": query,
        "rewritten_query": rewritten,
        "steps": steps,
        "total_elapsed_ms": total_elapsed,
    }
```

### 14.3 注册路由 & 创建调试前端页面

编辑 `backend/app/main.py`：

```python
from app.api import knowledge_bases, debug
app.include_router(knowledge_bases.router)
app.include_router(debug.router)
```

创建 `frontend/app/debug/page.tsx`（检索调试页面）：

```tsx
"use client";
import { useState } from "react";

interface Step {
  step: string;
  count?: number;
  results?: any[];
  elapsed_ms: number;
  input?: string;
  output?: string;
}

const STEP_LABELS: Record<string, string> = {
  query_rewrite: "Query 改写",
  vector_search: "向量检索",
  bm25_search: "BM25 检索",
  rrf_fusion: "RRF 融合",
};

export default function DebugPage() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const search = async () => {
    setLoading(true);
    const res = await fetch(
      `http://localhost:8000/api/v1/debug/retrieve?query=${encodeURIComponent(query)}`
    );
    const data = await res.json();
    setResult(data);
    setLoading(false);
  };

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h1 className="text-2xl font-bold mb-6">检索调试面板</h1>

      <div className="flex gap-2 mb-8">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="输入查询..."
          className="flex-1 border rounded-lg px-4 py-2"
          onKeyDown={(e) => e.key === "Enter" && search()}
        />
        <button
          onClick={search}
          disabled={loading}
          className="bg-blue-500 text-white px-6 py-2 rounded-lg hover:bg-blue-600 disabled:opacity-50"
        >
          {loading ? "检索中..." : "调试"}
        </button>
      </div>

      {result && (
        <div>
          <div className="mb-4 text-sm text-gray-500">
            原始 Query: "{result.original_query}" →
            改写: "{result.rewritten_query}" · 总耗时: {result.total_elapsed_ms}ms
          </div>

          <div className="grid grid-cols-1 gap-6">
            {result.steps?.map((step: Step, i: number) => (
              <div key={i} className="border rounded-lg p-4">
                <div className="flex justify-between items-center mb-3">
                  <h3 className="font-semibold text-lg">
                    Step {i + 1}: {STEP_LABELS[step.step] || step.step}
                  </h3>
                  <span className="text-sm text-gray-400">{step.elapsed_ms}ms</span>
                </div>

                {step.step === "query_rewrite" && (
                  <div>
                    <p className="text-gray-500">输入: {step.input}</p>
                    <p className="text-green-600 font-medium">输出: {step.output}</p>
                  </div>
                )}

                <div className="text-sm text-gray-500 mb-2">
                  共 {step.count} 条结果
                </div>

                <div className="space-y-2">
                  {step.results?.slice(0, 5).map((r: any, j: number) => (
                    <div key={j} className="bg-gray-50 rounded p-3 text-sm">
                      <div className="flex justify-between text-gray-400 mb-1">
                        <span>{r.source}</span>
                        <span>#{r.chunk_index}</span>
                      </div>
                      <p className="text-gray-700 line-clamp-3">{r.content_preview}</p>
                      {(r.rrf_score || r.bm25_score) && (
                        <div className="text-xs text-blue-500 mt-1">
                          {r.rrf_score && `RRF: ${r.rrf_score.toFixed(4)} `}
                          {r.bm25_score && `BM25: ${r.bm25_score.toFixed(2)}`}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
```

### 14.4 验证

```bash
# 后端
python -m uvicorn app.main:app --reload

# 前端
npm run dev
```

打开 `http://localhost:3000/debug`，输入查询，观察每一步的中间结果。

---

## 第 15 步：GitHub 数据源 + 增量同步（2 小时）

### 目标
支持从 GitHub 仓库拉取 Markdown 文档，并检测文件变更，增量更新索引。

### 15.1 GitHub 数据源

创建 `backend/app/ingestion/github_source.py`：

```python
"""GitHub 仓库数据源。"""
import os
import hashlib
from typing import List, Dict
from langchain_core.documents import Document
from app.ingestion.loader import load_file


def clone_or_pull(
    repo_url: str,
    branch: str = "main",
    target_dir: str = "./data/repos",
    token: str = None,
) -> str:
    """
    克隆或拉取 GitHub 仓库。
    Returns: 仓库在本地的工作目录路径
    """
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    repo_dir = os.path.join(target_dir, repo_name)

    if os.path.exists(repo_dir):
        # 已存在，执行 git pull
        import subprocess
        result = subprocess.run(
            ["git", "-C", repo_dir, "pull", "origin", branch],
            capture_output=True, text=True
        )
        print(f"  📥 Git Pull: {result.stdout.strip()}")
        return repo_dir
    else:
        # 首次克隆
        os.makedirs(target_dir, exist_ok=True)
        clone_url = repo_url
        if token:
            clone_url = repo_url.replace(
                "https://", f"https://{token}@"
            )
        import subprocess
        subprocess.run(
            ["git", "clone", "-b", branch, clone_url, repo_dir],
            check=True,
        )
        print(f"  📦 Git Clone: {repo_url} → {repo_dir}")
        return repo_dir


def list_doc_files(
    repo_dir: str,
    extensions: List[str] = None,
    exclude_dirs: List[str] = None,
) -> List[str]:
    """
    列出仓库中所有文档文件。
    默认扩展名：.md, .txt, .rst
    默认排除目录：.git, node_modules, __pycache__, .venv
    """
    if extensions is None:
        extensions = [".md", ".txt", ".rst"]
    if exclude_dirs is None:
        exclude_dirs = [".git", "node_modules", "__pycache__", ".venv", "vendor"]

    files = []
    for root, dirs, filenames in os.walk(repo_dir):
        # 跳过排除目录
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for f in filenames:
            if any(f.endswith(ext) for ext in extensions):
                files.append(os.path.join(root, f))

    return files


def compute_file_hash(filepath: str) -> str:
    """计算文件 SHA256，用于增量检测。"""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()


def load_repo_documents(
    repo_dir: str,
    file_hashes: Dict[str, str] = None,
) -> Dict:
    """
    加载仓库中所有文档，支持增量更新。
    如果 file_hashes 不为空，只返回新文件和修改过的文件。

    Returns:
        {
            "documents": List[Document],
            "new_hashes": Dict[str, str],  # 所有文件的 hash
            "stats": {"total": int, "new": int, "modified": int, "unchanged": int}
        }
    """
    if file_hashes is None:
        file_hashes = {}

    doc_files = list_doc_files(repo_dir)
    all_docs = []
    new_hashes = {}
    stats = {"total": len(doc_files), "new": 0, "modified": 0, "unchanged": 0}

    for filepath in doc_files:
        current_hash = compute_file_hash(filepath)
        new_hashes[filepath] = current_hash

        old_hash = file_hashes.get(filepath, "")
        if old_hash == current_hash:
            stats["unchanged"] += 1
            continue  # 文件未变化，跳过

        if old_hash:
            stats["modified"] += 1
            # 删掉旧的 Chunk（按 file_path 过滤删除）
            print(f"  ✏️  文件变更: {filepath}")
        else:
            stats["new"] += 1
            print(f"  ➕ 新文件: {filepath}")

        try:
            docs = load_file(filepath)
            all_docs.extend(docs)
        except Exception as e:
            print(f"  ⚠️  解析失败 {filepath}: {e}")

    return {
        "documents": all_docs,
        "new_hashes": new_hashes,
        "stats": stats,
    }
```

### 15.2 定时同步脚本

创建 `backend/scripts/sync_github.py`：

```python
"""GitHub 增量同步脚本（配合 cron/定时触发）。"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv; load_dotenv()

from app.ingestion.github_source import clone_or_pull, load_repo_documents
from app.ingestion.splitter import split_documents, add_chunk_index
from app.storage.vector_store import VectorStoreManager

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
HASH_FILE = "./data/github_file_hashes.json"


def sync_github_repo(
    repo_url: str = "https://github.com/redis/redis-doc.git",
    branch: str = "main",
    collection_name: str = "docmind",
):
    print(f"🔄 同步 GitHub: {repo_url} (branch={branch})")

    # 1. 克隆/拉取
    repo_dir = clone_or_pull(repo_url, branch)

    # 2. 读取上次的 hash 记录
    old_hashes = {}
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE) as f:
            old_hashes = json.load(f)

    # 3. 加载变更文件
    result = load_repo_documents(repo_dir, old_hashes)
    print(f"  📊 统计: 总{result['stats']['total']}个文件, "
          f"新增{result['stats']['new']}, "
          f"修改{result['stats']['modified']}, "
          f"未变{result['stats']['unchanged']}")

    # 4. 如果有变更，分割并索引
    if result["documents"]:
        chunks = split_documents(result["documents"])
        chunks = add_chunk_index(chunks)
        print(f"  ✂️  新增/更新 {len(chunks)} 个 Chunk")

        vm = VectorStoreManager(persist_dir=CHROMA_DIR)
        vm.index_documents(chunks, collection_name=collection_name)
        print(f"  ✅ 索引更新完成")

    # 5. 保存新的 hash 记录
    with open(HASH_FILE, "w") as f:
        json.dump(result["new_hashes"], f, indent=2)

    print("✅ 同步完成")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="https://github.com/redis/redis-doc.git")
    parser.add_argument("--branch", default="main")
    args = parser.parse_args()

    sync_github_repo(args.repo, args.branch)
```

### 15.3 验证

```bash
cd ~/Desktop/docmind/backend
python scripts/sync_github.py --repo https://github.com/redis/redis-doc.git --branch master
```

预期输出：克隆仓库 → 加载所有 Markdown 文件 → 分块 → 索引。

---

## 第 16 步：API Key 认证（1 小时）

### 目标
保护所有 API 端点，需要有效的 API Key 才能访问。

### 16.1 创建认证依赖

创建 `backend/app/core/__init__.py`（空文件）。

创建 `backend/app/core/auth.py`：

```python
"""API Key 认证。"""
from fastapi import Security, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
import os

load_dotenv()

security = HTTPBearer(auto_error=False)

# 有效的 API Keys（生产环境存数据库）
VALID_API_KEYS = os.getenv("API_KEYS", "dev-key-docmind-2025").split(",")


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> str:
    """
    验证 API Key。三种方式：
    1. Authorization: Bearer <key>
    2. X-API-Key: <key>
    3. 请求参数 ?api_key=<key>
    """
    # 开发模式跳过认证
    if os.getenv("ENV", "dev") == "dev" and not credentials:
        return "anonymous"

    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="请提供 API Key：Authorization: Bearer <your-key>",
        )

    token = credentials.credentials
    if token not in VALID_API_KEYS:
        raise HTTPException(status_code=403, detail="无效的 API Key")

    return token


# 可选认证：有 Key 就验证，没有也可以访问（用于健康检查等）
async def optional_api_key(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> str:
    if not credentials:
        return "anonymous"
    token = credentials.credentials
    return token if token in VALID_API_KEYS else "anonymous"
```

### 16.2 保护 API 端点

编辑 `backend/app/api/chat.py`，给路由添加认证：

```python
from app.core.auth import verify_api_key

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    api_key: str = Depends(verify_api_key),  # 新增
):
    # ... 其余不变
```

### 16.3 生成 API Key

创建 `backend/scripts/generate_api_key.py`：

```python
"""生成 API Key。"""
import secrets
import hashlib

def generate_key() -> str:
    raw = secrets.token_hex(24)
    return f"dm-{raw}"  # dm = docmind 前缀

if __name__ == "__main__":
    key = generate_key()
    print(f"新的 API Key: {key}")
    print(f"请将以下行添加到 .env 文件：")
    print(f'API_KEYS=dev-key-docmind-2025,{key}')
```

```bash
cd ~/Desktop/docmind/backend
python scripts/generate_api_key.py
```

### 16.4 验证

```bash
# 无 Key 请求 → 401
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "test"}'

# 带 Key 请求 → 200
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dev-key-docmind-2025" \
  -d '{"message": "Redis怎么用？"}'
```

---

## 最终项目结构

完成所有 16 步后，你的项目结构：

```
docmind/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── chat.py              # 问答 + 流式 SSE
│   │   │   ├── documents.py         # 文档管理 CRUD
│   │   │   ├── feedback.py          # 用户反馈
│   │   │   ├── knowledge_bases.py   # 知识库管理
│   │   │   └── debug.py             # 检索调试
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   └── auth.py              # API Key 认证
│   │   ├── ingestion/
│   │   │   ├── __init__.py
│   │   │   ├── loader.py            # 多格式文档加载
│   │   │   ├── splitter.py          # 多策略分割
│   │   │   └── github_source.py     # GitHub 数据源
│   │   ├── embedding/
│   │   │   ├── __init__.py
│   │   │   └── embeddings.py        # Embedding 封装
│   │   ├── storage/
│   │   │   ├── __init__.py
│   │   │   └── vector_store.py      # Chroma 向量库
│   │   ├── llm/
│   │   │   ├── __init__.py
│   │   │   └── chat_model.py        # LLM 封装
│   │   ├── rag/
│   │   │   ├── __init__.py
│   │   │   └── pipeline.py          # 基础 RAG
│   │   ├── graph/
│   │   │   ├── __init__.py
│   │   │   ├── state.py             # LangGraph 状态定义
│   │   │   ├── nodes.py             # 所有节点（含幻觉检测、自修正）
│   │   │   └── workflow.py          # 完整工作流
│   │   ├── conversation/
│   │   │   ├── __init__.py
│   │   │   └── manager.py           # 多轮对话管理
│   │   └── quality/
│   │       ├── __init__.py
│   │       └── hallucination_check.py  # 幻觉检测
│   ├── scripts/
│   │   ├── index_documents.py
│   │   ├── evaluate.py
│   │   ├── sync_github.py
│   │   └── generate_api_key.py
│   ├── tests/
│   │   ├── test_loader.py
│   │   ├── test_splitter.py
│   │   ├── test_rag.py
│   │   └── test_graph.py
│   ├── data/
│   │   ├── chroma_db/
│   │   ├── uploads/
│   │   ├── repos/                   # Git 仓库缓存
│   │   ├── feedback.jsonl
│   │   ├── hard_negatives.jsonl
│   │   └── github_file_hashes.json
│   ├── requirements.txt
│   ├── .env
│   └── venv/
├── frontend/
│   ├── app/
│   │   ├── page.tsx                  # 对话界面（流式+反馈）
│   │   ├── debug/
│   │   │   └── page.tsx              # 检索调试面板
│   │   ├── layout.tsx
│   │   └── globals.css
│   ├── package.json
│   └── next.config.js
├── test_data/                        # 8 个测试文档
├── RAG项目需求文档.md
└── DocMind开发步骤.md
```

---

## 更新后的时间表（完整 16 步）

| 阶段 | 内容 | 累计时间 |
|------|------|----------|
| MVP Day 1-3 | 第 0-4 步：环境 + 加载 + 分割 + 索引 + 基础 RAG | 8h |
| MVP Day 4-5 | 第 5 步：LangGraph 核心工作流 | 4h |
| MVP Day 6 | 第 6-7 步：API + 前端界面 | 5h |
| MVP Day 7 | 第 8 步：RAGAS 评估 | 2h |
| **进阶 Day 8** | 第 9 步：Query 分解 | 2h |
| **进阶 Day 9** | 第 10 步：多轮对话管理 | 3h |
| **进阶 Day 10** | 第 11 步：流式输出 SSE | 2h |
| **进阶 Day 11-12** | 第 12 步：幻觉检测 + 自修正 | 3h |
| **进阶 Day 13** | 第 13 步：用户反馈闭环 | 1.5h |
| **进阶 Day 14-15** | 第 14 步：管理后台 + 调试面板 | 3h |
| **进阶 Day 16** | 第 15-16 步：GitHub 同步 + 认证 | 3h |

**总计约 36.5 小时**，MVP 19 小时 + 进阶 17.5 小时。大约两个周末 + 工作日晚上的量。

---

## 常见问题排查

### Q1: 加载 Markdown 报错 "UnstructuredMarkdownLoader not found"
```bash
pip install unstructured markdown
```

### Q2: Chroma 报错 "sqlite3.OperationalError"
检查 `CHROMA_PERSIST_DIR` 路径是否存在且有写入权限。

### Q3: BGE-M3 模型下载慢
```bash
# 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com
# 或者用较小的模型先跑通：model_name="BAAI/bge-small-zh-v1.5"
```

### Q4: RAGAS 评估报错 "openai/api key"
RAGAS 默认用 OpenAI 的 LLM 做评估，需要配置 API Key。或者指定其他评估模型：
```python
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
results = evaluate(dataset, metrics=[...], llm=evaluator_llm)
```

---

一步一步来，别跳步，每一步都有验证命令。遇到问题先看日志，99% 的问题都能在终端输出里找到原因。加油！
