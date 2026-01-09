from __future__ import annotations

import os

import weaviate
from langchain_community.vectorstores import Weaviate

from .llm import embeddings_from_env


def weaviate_client_from_env():
    # Weaviate 客户端（可选 API Key）
    url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    api_key = os.getenv("WEAVIATE_API_KEY")
    if api_key:
        auth = weaviate.AuthApiKey(api_key=api_key)
        return weaviate.Client(url=url, auth_client_secret=auth)
    return weaviate.Client(url=url)


def build_retriever(*, search_k: int = 4):
    # 构建 Weaviate 检索器（使用同一 Embedding 模型）
    embeddings = embeddings_from_env()
    client = weaviate_client_from_env()
    class_name = os.getenv("WEAVIATE_CLASS", "RAGChunk")
    if not client.schema.exists(class_name):
        raise RuntimeError(
            f"Weaviate class {class_name} not found. Run `PYTHONPATH=src python -m services.ingest.processor` first."
        )
    vectorstore = Weaviate(
        client=client,
        index_name=class_name,
        text_key="text",
        embedding=embeddings,
        attributes=["source"],
    )
    return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": search_k})
