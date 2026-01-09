from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from .state import RAGState


def _format_source(meta: dict) -> str:
    for key in ("source", "file_path", "path", "filename", "file_name", "file"):
        value = meta.get(key)
        if not value:
            continue
        if isinstance(value, list):
            value = value[0] if value else ""
        value = str(value)
        if not value:
            continue
        try:
            return Path(value).name
        except Exception:
            return value
    return "unknown"


def format_docs(docs: List[Document]) -> str:
    # 将检索到的文档拼成模型可读的上下文文本
    return "\n\n".join(
        f"Source: {_format_source(doc.metadata or {})}\n{doc.page_content}"
        for doc in docs
    )


def build_prompt() -> ChatPromptTemplate:
    # 约束模型必须基于上下文回答
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个严谨的中文助手。你只能基于给定的“上下文”回答问题，禁止编造。"
                "如果上下文中没有答案，请直接说明“我不知道/资料不足”。"
                "如果能从上下文中定位到来源文件名，请在回答中引用文件名。\n\n"
                "用户记忆（可能为空）：\n{memory}\n\n"
                "上下文：\n{context}",
            ),
            ("user", "{question}"),
        ]
    )


def retrieve(state: RAGState, *, retriever) -> dict:
    # 取用户最后一句作为检索 query
    question = state["messages"][-1].content
    docs = retriever.invoke(question)
    return {"context": docs}


def generate(state: RAGState, *, rag_chain) -> dict:
    # 用检索到的上下文调用 LLM
    question = state["messages"][-1].content
    context = format_docs(state.get("context", []))
    memory = state.get("memory", "")
    ai_msg: AIMessage = rag_chain.invoke(
        {"context": context, "question": question, "memory": memory}
    )
    return {"messages": state["messages"] + [ai_msg]}
