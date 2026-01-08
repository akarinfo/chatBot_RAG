"""Build a Weaviate vector index from documents in ./data."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import Weaviate
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

try:
    from .providers import embeddings_from_env, weaviate_client_from_env
except ImportError:  # allows `python src/rag/ingest.py`
    from providers import embeddings_from_env, weaviate_client_from_env


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

load_dotenv()


def load_documents(data_dir: Path) -> List[Document]:
    # 只读取指定后缀的文本/Markdown 文件
    allowed_suffixes = {".txt", ".md", ".mdx"}
    docs: List[Document] = []
    for path in data_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in allowed_suffixes:
            text = path.read_text(encoding="utf-8")
            docs.append(Document(page_content=text, metadata={"source": str(path)}))
    return docs


def select_embeddings():
    # 通过环境变量选择 Embedding 提供方
    return embeddings_from_env()


def ingest():
    # 入库主流程：读文档 → 分块 → 向量化 → 写入 Weaviate
    docs = load_documents(DATA_DIR)
    if not docs:
        raise SystemExit(f"No documents found in {DATA_DIR}")

    # Markdown 先按标题分块，再做递归分块
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
    )
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks: List[Document] = []
    for doc in docs:
        source = doc.metadata.get("source", "")
        if source.lower().endswith((".md", ".mdx")):
            md_docs = header_splitter.split_text(doc.page_content)
            # Preserve source path on all header-split docs
            for md_doc in md_docs:
                md_doc.metadata["source"] = source
            chunks.extend(splitter.split_documents(md_docs))
        else:
            chunks.extend(splitter.split_documents([doc]))

    # 写入 Weaviate（可通过环境变量控制类名与是否重建）
    embeddings = select_embeddings()
    client = weaviate_client_from_env()
    class_name = os.getenv("WEAVIATE_CLASS", "RAGChunk")
    rebuild = os.getenv("WEAVIATE_REBUILD", "1") == "1"

    if rebuild and client.schema.exists(class_name):
        client.schema.delete_class(class_name)

    Weaviate.from_documents(
        chunks,
        embedding=embeddings,
        client=client,
        index_name=class_name,
        text_key="text",
    )
    print(f"Ingested {len(chunks)} chunks from {len(docs)} files into Weaviate:{class_name}")


if __name__ == "__main__":
    ingest()
