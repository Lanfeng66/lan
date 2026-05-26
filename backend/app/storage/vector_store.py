"""Chroma 向量数据库操作。"""
import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from app.embedding.embeddings import get_embedding_model

class VectorStoreManager:
    """管理 Chroma 向量存储：索引、检索、删除。"""
    def __init__(self, persist_dir: str):
        #存储向量数据库文件的目录路径
        self.persist_dir = persist_dir
        #获取单例 embedding 模型，用于文本向量化
        self.embeddings = get_embedding_model()
        #如果目录不存在则创建，exist_ok=True 避免目录已存在时报错
        os.makedirs(persist_dir, exist_ok=True)

    def index_documents(self, docs: List[Document], collection_name: str = "default") -> Chroma:
        """索引文档。"""
        #创建 Chroma 向量数据库实例
        vectorstore = Chroma.from_documents(
            #待索引的文档对象列表
            documents=docs,
            #向量数据持久化存储路径
            persist_directory=self.persist_dir,
            #向量化模型
            embedding=self.embeddings,
            #集合名称
            collection_name=collection_name
        )
        print(f"✅ 已索引 {len(docs)} 个文档到集合 '{collection_name}'")
        return vectorstore

    def get_retriever(self, collection_name: str = "default", top_k: int = 3, search_type: str = "similarity"):
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

    def mmr_search(
            self,
            query: str,
            collection_name: str = "default",
            top_k: int = 5,
            fetch_k: int = 20,
            lambda_mult: float = 0.5,
    ) -> List[Document]:
        """
        MMR 检索：在相关性和多样性之间取平衡。

        当代码示例里恰好出现了你的 query 字符串时，
        MMR 会惩罚和已选中结果高度相似的 chunk，
        从而把真正的技术文档推上来。

        fetch_k: 粗筛数量（越大越不容易漏）
        lambda_mult: 0=最大多样性, 1=最大相关性, 0.5=平衡
        """
        vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            collection_name=collection_name,
        )
        return vectorstore.max_marginal_relevance_search(
            query,
            k=top_k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
        )

    def similarity_search(self, query: str, collection_name: str = "default", top_k: int = 3) -> List[Document]:
        """语义检索。"""
        #创建 Chroma 向量数据库实例
        vectorstore = Chroma(
            #向量数据持久化存储路径
            persist_directory=self.persist_dir,
            #向量化模型
            embedding_function=self.embeddings,
            #集合名称
            collection_name=collection_name
        )
        #搜索文档
        docs = vectorstore.similarity_search(query, k=top_k)
        return docs

    def delete_collection(self, collection_name: str) -> None:
        """删除集合。"""
        #创建 Chroma 向量数据库实例
        vectorstore = Chroma(
            #向量数据持久化存储路径
            persist_directory=self.persist_dir,
            #向量化模型
            embedding_function=self.embeddings,
            #集合名称
            collection_name=collection_name
        )
        #删除集合
        vectorstore.delete_collection()
        print(f"✅ 集合 '{collection_name}' 已删除")
