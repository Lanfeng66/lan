# Python 代码规范与性能优化

## 1. 类型注解（Type Hints）

### 1.1 函数签名

```python
from typing import Optional, List, Dict, Any
from datetime import datetime

def search_documents(
    query: str,
    kb_id: str,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    min_score: float = 0.5
) -> List[Dict[str, Any]]:
    """
    在指定知识库中检索文档。

    Args:
        query: 用户输入的查询字符串
        kb_id: 知识库 UUID
        top_k: 返回的最大结果数，默认 5
        filters: 元数据过滤条件，如 {"source_type": "pdf"}
        min_score: 最低相关性分数阈值

    Returns:
        检索结果列表，每个元素包含 doc_id, text, score 字段
    """
    ...
```

### 1.2 使用 dataclass 减少样板代码

```python
from dataclasses import dataclass, field
from uuid import UUID, uuid4

@dataclass
class Document:
    id: UUID = field(default_factory=uuid4)
    kb_id: UUID = None
    title: str = ""
    source_type: str = "markdown"
    chunk_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_ready(self) -> bool:
        return self.chunk_count > 0
```

## 2. 异步编程（asyncio）

### 2.1 为什么 RAG 项目需要异步？

一个 RAG 查询链路包含多次 I/O：查询 Embedding API → 向量数据库检索 → BM25 检索 → 调用 Reranker → 调用 LLM API。同步执行至少耗时 3-5 秒，异步并发可将延迟降低 40-60%。

### 2.2 正确的异步用法

```python
import asyncio
from typing import List

async def hybrid_search(query: str, top_k: int = 30) -> List[dict]:
    """混合检索：向量检索和 BM25 检索并发执行"""
    vector_task = asyncio.create_task(vector_search(query, top_k))
    bm25_task = asyncio.create_task(bm25_search(query, top_k))

    vector_results, bm25_results = await asyncio.gather(
        vector_task, bm25_task
    )

    # RRF 融合
    return rrf_fusion(vector_results, bm25_results)
```

### 2.3 常见 asyncio 陷阱

```python
# 错误：在 async 函数中使用 time.sleep()
async def bad_sleep():
    import time
    time.sleep(1)  # 阻塞整个事件循环！

# 正确：使用 asyncio.sleep()
async def good_sleep():
    await asyncio.sleep(1)  # 让出控制权给其他协程

# 错误：在 async 函数中直接用 requests 库
async def bad_http():
    import requests
    resp = requests.get("https://api.example.com")  # 阻塞！

# 正确：使用 httpx 异步客户端
async def good_http():
    import httpx
    async with httpx.AsyncClient() as client:
        resp = await client.get("https://api.example.com")
```

## 3. 内存优化

### 3.1 生成器替代列表

```python
# 内存低效：一次性加载所有 chunk 的 embedding
def get_all_embeddings_bad(chunks):
    return [compute_embedding(c) for c in chunks]  # 1万个chunk→内存爆炸

# 内存高效：惰性生成
def get_all_embeddings_good(chunks):
    for chunk in chunks:
        yield compute_embedding(chunk)  # 一次只加载一个
```

### 3.2 使用 __slots__ 减少实例内存

```python
class ChunkWithSlots:
    __slots__ = ('id', 'text', 'embedding', 'metadata')
    def __init__(self, id, text, embedding=None, metadata=None):
        self.id = id
        self.text = text
        self.embedding = embedding
        self.metadata = metadata or {}
```

使用 `__slots__` 后，每个实例内存占用减少约 40-50%（取决于属性数量）。

### 3.3 大文件处理

```python
def parse_large_pdf(filepath: str, chunk_size: int = 1000):
    """流式解析大 PDF，避免一次性加载全部内容"""
    import fitz  # PyMuPDF

    doc = fitz.open(filepath)
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text = page.get_text()
        for i in range(0, len(text), chunk_size):
            yield {
                "text": text[i:i+chunk_size],
                "page_number": page_num + 1,
                "start_pos": i
            }
    doc.close()
```

## 4. 并发与并行

### 4.1 ThreadPoolExecutor（适合 I/O 密集型）

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def batch_embed(texts: List[str], batch_size: int = 20) -> List[List[float]]:
    embeddings = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(embed_single, text, idx): idx
            for idx, text in enumerate(texts)
        }
        # 按提交顺序收集结果
        results = [None] * len(texts)
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
    return results
```

### 4.2 ProcessPoolExecutor（适合 CPU 密集型）

```python
from concurrent.futures import ProcessPoolExecutor

# PDF 解析是 CPU 密集型，用进程池
def batch_parse_pdfs(filepaths: List[str]) -> List[dict]:
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(parse_pdf_file, filepaths))
    return results
```

## 5. 常见反模式

| 反模式 | 问题 | 改进 |
|--------|------|------|
| `except:` 裸捕获 | 吞掉所有异常，包括 KeyboardInterrupt | `except Exception as e:` |
| 可变默认参数 | `def f(lst=[])` 会在多次调用间共享 | `def f(lst=None): lst = lst or []` |
| 字符串拼接造 SQL | SQL 注入风险 | 使用参数化查询 |
| `for i in range(len(x))` | 不 Pythonic | `for item in x:` 或 `for i, item in enumerate(x):` |
| `if x == True` | 多余，且不能区分 True 和 truthy | `if x:` |
| 全局变量 | 测试困难，状态不可控 | 依赖注入 / Config 对象 |

---

> **文档版本**：v1.5 | **适用范围**：Python 3.11+ | **最后审核**：2025-09-01
