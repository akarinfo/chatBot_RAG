"""LangGraph-based RAG question-answering app."""

from __future__ import annotations

import os
from typing import List, TypedDict

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


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
        for doc in docs
    )


def build_retriever():
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
    )
    return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4})


def build_llm():
    return llm_from_env()


def build_prompt():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个严谨的中文助手。你只能基于给定的“上下文”回答问题，禁止编造。"
                "如果上下文中没有答案，请直接说明“我不知道/资料不足”。"
                "如果能从上下文中定位到来源文件名，请在回答中引用文件名。\n\n"
                "上下文：\n{context}",
            ),
            ("user", "{question}"),
        ]
    )


def build_graph():
    retriever = build_retriever()
    llm = build_llm()
    prompt = build_prompt()
    rag_chain = prompt | llm

    def retrieve(state: RAGState):
        question = state["messages"][-1].content
        docs = retriever.get_relevant_documents(question)
        return {"context": docs}

    def generate(state: RAGState):
        question = state["messages"][-1].content
        context = format_docs(state.get("context", []))
        ai_msg: AIMessage = rag_chain.invoke({"context": context, "question": question})
        return {"messages": state["messages"] + [ai_msg]}

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    app = graph.compile()

    def ask(question: str, history: List[BaseMessage] | None = None) -> AIMessage:
        history = history or []
        state: RAGState = {"messages": history + [HumanMessage(content=question)]}
        result = app.invoke(state)
        return result["messages"][-1]

    def ask_stream(question: str, history: List[BaseMessage] | None = None):
        docs = retriever.get_relevant_documents(question)
        context = format_docs(docs)
        stream = rag_chain.stream({"context": context, "question": question})
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
