"""统一的文档加载器：根据文件扩展名自动选择加载策略。"""
from typing import List
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader,TextLoader,Docx2txtLoader,UnstructuredHTMLLoader,UnstructuredMarkdownLoader


def load_file(file_path:str)->List[Document]:
    """根据文件后缀分发到对应的加载函数。"""
    #suffix获取文件后缀，转换为小写
    ext = Path(file_path).suffix.lower()

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

# 加载pdf文件
def _load_pdf(file_path:str)->List[Document]:
    """加载PDF文件。"""
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    # metadata添加元数据：source_type, file_path
    for doc in docs:
        doc.metadata["source_type"] = "pdf"
        doc.metadata["file_path"] = file_path
    return docs

# 加载markdown文件
def _load_markdown(file_path:str)->List[Document]:
    """加载Markdown文件。"""
    loader = UnstructuredMarkdownLoader(file_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source_type"] = "markdown"
        doc.metadata["file_path"] = file_path
    return docs

# 加载txt文件
def _load_text(file_path:str)->List[Document]:
    """加载txt文件。"""
    loader = TextLoader(file_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source_type"] = "text"
        doc.metadata["file_path"] = file_path
    return docs

# 加载docx文件
def _load_docx(file_path:str)->List[Document]:
    """加载docx文件。"""
    loader = Docx2txtLoader(file_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source_type"] = "docx"
        doc.metadata["file_path"] = file_path
    return docs

# 加载html文件
def _load_html(file_path:str)->List[Document]:
    """加载html文件。"""
    loader = UnstructuredHTMLLoader(file_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source_type"] = "html"
        doc.metadata["file_path"] = file_path
        return docs
