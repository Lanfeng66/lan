"""文本分割策略：支持多种分割方式。"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter,MarkdownHeaderTextSplitter
# 文本分割
def split_documents(text: str, chunk_size: int = 512, chunk_overlap: int = 0,strategy: str = "recursive") -> List[Document]:
    """
    核心分割函数。

    Args:
        documents: 待分割的文档列表
        chunk_size: 每个 chunk 的目标大小（字符数）
        chunk_overlap: chunk 之间的重叠字符数
        strategy: "recursive" | "markdown_header"
    """
    if strategy == "markdown_header":
        return _split_markdown_header(text, chunk_size, chunk_overlap)
    else:
        return _split_recursive(text, chunk_size, chunk_overlap)

def _split_recursive(documents: List[Document], chunk_size: int = 512, chunk_overlap: int = 0) -> List[Document]:
    """
    递归分割文本。

    Args:
        text: 待分割的文本
        chunk_size: 每个 chunk 的目标大小（字符数）
        chunk_overlap: chunk 之间的重叠字符数
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ".", " ", ""],
        length_function=len,
    )
    docs = text_splitter.split_documents(documents)
    return docs

def _split_markdown_header(documents: List[Document], chunk_size: int = 512, chunk_overlap: int = 0) -> List[Document]:
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
