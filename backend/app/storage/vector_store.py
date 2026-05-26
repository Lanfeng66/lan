"""Chroma 向量数据库操作。"""
import os
import time
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from app.embedding.embeddings import get_embedding_model, embed_batch


class VectorStoreManager:
    """管理 Chroma 向量存储：索引、检索、删除。"""

    def __init__(self, persist_dir: str):
        self.persist_dir = persist_dir
        self.embeddings = get_embedding_model()
        os.makedirs(persist_dir, exist_ok=True)

    def index_documents(
        self, docs: List[Document], collection_name: str = "default"
    ) -> Chroma:
        """索引文档（使用 Chroma 内置逐 batch 编码）。"""
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
            collection_name=collection_name,
        )
        print(f"✅ 已索引 {len(docs)} 个文档到集合 '{collection_name}'")
        return vectorstore

    def index_documents_batch(
        self, docs: List[Document], collection_name: str = "default"
    ) -> Chroma:
        """高效批量索引：一次性编码全部文本，再写入 Chroma。

        相比 index_documents（Chroma 内部逐 batch 调用 embed_documents），
        这个方法让 SentenceTransformer 自己管理 batch 和 GPU 传输，减少
        Python↔C 跨界开销，适合 100+ chunks 的批量导入。
        """
        t0 = time.time()
        texts = [d.page_content for d in docs]
        metadatas = [d.metadata for d in docs]
        ids = [
            f"{collection_name}_{i}"
            for i in range(len(docs))
        ]

        print(f"  🔤 编码 {len(texts)} 条文本...")
        embeddings = embed_batch(texts)
        print(f"  ⏱️  编码耗时: {time.time() - t0:.1f}s")

        vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            collection_name=collection_name,
        )
        vectorstore.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings,  # 直接传入预计算的向量，跳过 Chroma 再编码
        )
        print(f"✅ 批量索引完成: {len(docs)} 条 → 集合 '{collection_name}'，总耗时 {time.time() - t0:.1f}s")
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
