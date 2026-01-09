from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


ChunkingMethod = str


def build_splitters(
    *, chunk_size: int = 800, chunk_overlap: int = 120
) -> tuple[MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter]:
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return header_splitter, splitter


def _split_documents_auto(
    docs: List[Document],
    *,
    header_splitter: MarkdownHeaderTextSplitter,
    splitter: RecursiveCharacterTextSplitter,
) -> List[Document]:
    chunks: List[Document] = []
    for doc in docs:
        source = doc.metadata.get("source", "")
        if str(source).lower().endswith((".md", ".mdx")):
            md_docs = header_splitter.split_text(doc.page_content)
            for md_doc in md_docs:
                md_doc.metadata["source"] = source
            md_chunks = splitter.split_documents(md_docs)
            for c in md_chunks:
                c.metadata.setdefault("source", source)
            chunks.extend(md_chunks)
        else:
            doc_chunks = splitter.split_documents([doc])
            for c in doc_chunks:
                c.metadata.setdefault("source", source)
            chunks.extend(doc_chunks)
    return chunks


def _split_documents_recursive_only(
    docs: List[Document], *, splitter: RecursiveCharacterTextSplitter
) -> List[Document]:
    chunks = splitter.split_documents(docs)
    if len(docs) == 1:
        source = docs[0].metadata.get("source", "")
        for c in chunks:
            c.metadata.setdefault("source", source)
    return chunks


def add_spans_inplace(original_text: str, chunks: List[Document]) -> None:
    cursor = 0
    for c in chunks:
        text = c.page_content or ""
        start = original_text.find(text, cursor)
        end = None
        if start != -1:
            end = start + len(text)
        else:
            stripped = text.strip()
            if stripped and stripped != text:
                start2 = original_text.find(stripped, cursor)
                if start2 != -1:
                    start = start2
                    end = start + len(stripped)
            if end is None:
                prefix = text[: min(200, len(text))].strip()
                if prefix:
                    start2 = original_text.find(prefix, cursor)
                    if start2 != -1:
                        start = start2
                        end = min(len(original_text), start + len(text))
        if end is None or start == -1:
            c.metadata["span_start"] = None
            c.metadata["span_end"] = None
            continue
        c.metadata["span_start"] = int(start)
        c.metadata["span_end"] = int(end)
        cursor = int(end)


def chunk_documents(
    docs: List[Document],
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    method: ChunkingMethod = "auto",
    include_preview_metadata: bool = False,
    include_spans: bool = False,
) -> List[Document]:
    header_splitter, splitter = build_splitters(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    if method == "recursive_only":
        chunks = _split_documents_recursive_only(docs, splitter=splitter)
    else:
        # default: auto (md header split + recursive for md/mdx; recursive for others)
        chunks = _split_documents_auto(
            docs, header_splitter=header_splitter, splitter=splitter
        )

    if include_preview_metadata:
        for idx, c in enumerate(chunks):
            c.metadata.setdefault("source", "")
            c.metadata["chunk_index"] = idx
            c.metadata["chunk_chars"] = len(c.page_content or "")
            c.metadata["chunking_method"] = method
            c.metadata["chunk_size"] = chunk_size
            c.metadata["chunk_overlap"] = chunk_overlap

    if include_spans:
        if len(docs) != 1:
            raise ValueError("include_spans requires a single-document input")
        add_spans_inplace(docs[0].page_content or "", chunks)
    return chunks


def chunk_file(
    path: Path,
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    method: ChunkingMethod = "auto",
    include_preview_metadata: bool = True,
    include_spans: bool = True,
) -> List[Document]:
    text = path.read_text(encoding="utf-8")
    doc = Document(page_content=text, metadata={"source": str(path)})
    return chunk_documents(
        [doc],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        method=method,
        include_preview_metadata=include_preview_metadata,
        include_spans=include_spans,
    )
