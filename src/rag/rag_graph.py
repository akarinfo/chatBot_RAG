"""LangGraph-based RAG question-answering app."""

from __future__ import annotations

import os
from typing import List, TypedDict
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Weaviate
from langgraph.graph import END, StateGraph

try:
    from .providers import embeddings_from_env, llm_from_env, weaviate_client_from_env
except ImportError:  # allows `python src/rag/rag_graph.py`
    from providers import embeddings_from_env, llm_from_env, weaviate_client_from_env

load_dotenv()


class RAGState(TypedDict):
    messages: List[BaseMessage]
    context: List[Document]
    memory: str


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


def build_retriever():
    # 构建 Weaviate 检索器（使用同一 Embedding 模型）
    embeddings = embeddings_from_env()
    client = weaviate_client_from_env()
    class_name = os.getenv("WEAVIATE_CLASS", "RAGChunk")
    if not client.schema.exists(class_name):
        raise RuntimeError(
            f"Weaviate class {class_name} not found. Run `python src/rag/ingest.py` first."
        )
    vectorstore = Weaviate(
        client=client,
        index_name=class_name,
        text_key="text",
        embedding=embeddings,
        attributes=["source"],
    )
    return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4})


def build_llm():
    # 根据环境变量创建 LLM 实例
    return llm_from_env()


def build_prompt():
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


def build_graph():
    # LangGraph：retrieve → generate
    retriever = build_retriever()
    llm = build_llm()
    prompt = build_prompt()
    rag_chain = prompt | llm

    def retrieve(state: RAGState):
        # 取用户最后一句作为检索 query
        question = state["messages"][-1].content
        docs = retriever.invoke(question)
        return {"context": docs}

    def generate(state: RAGState):
        # 用检索到的上下文调用 LLM
        question = state["messages"][-1].content
        context = format_docs(state.get("context", []))
        memory = state.get("memory", "")
        ai_msg: AIMessage = rag_chain.invoke(
            {"context": context, "question": question, "memory": memory}
        )
        return {"messages": state["messages"] + [ai_msg]}

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    app = graph.compile()

    def ask(
        question: str, history: List[BaseMessage] | None = None, *, memory: str = ""
    ) -> AIMessage:
        # 直接调用图执行一次完整的 RAG
        history = history or []
        state: RAGState = {
            "messages": history + [HumanMessage(content=question)],
            "memory": memory or "",
        }
        result = app.invoke(state)
        return result["messages"][-1]

    def ask_stream(
        question: str, history: List[BaseMessage] | None = None, *, memory: str = ""
    ):
        # 只做检索并流式生成（用于 Web UI）
        docs = retriever.invoke(question)
        context = format_docs(docs)
        stream = rag_chain.stream(
            {"context": context, "question": question, "memory": memory or ""}
        )
        for chunk in stream:
            if isinstance(chunk, AIMessageChunk):
                if chunk.content:
                    yield chunk.content
            elif isinstance(chunk, AIMessage):
                if chunk.content:
                    yield chunk.content
            else:
                content = getattr(chunk, "content", None)
                if content:
                    yield content

    return app, ask, ask_stream


if __name__ == "__main__":
    _, ask, _ = build_graph()
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not user_input:
            continue
        answer = ask(user_input)
        print(f"Bot: {answer.content}\n")
