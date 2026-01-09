"""LangGraph-based RAG question-answering app."""

from __future__ import annotations

from functools import partial
from typing import List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.messages.ai import AIMessageChunk
from langgraph.graph import END, StateGraph

from core.config import load_env
from core.llm import llm_from_env
from core.vectordb import build_retriever
from .nodes import build_prompt, format_docs, generate, retrieve
from .state import RAGState

load_env()


def build_graph():
    # LangGraph：retrieve → generate
    retriever = build_retriever()
    llm = llm_from_env()
    prompt = build_prompt()
    rag_chain = prompt | llm

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", partial(retrieve, retriever=retriever))
    graph.add_node("generate", partial(generate, rag_chain=rag_chain))

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


def build_rag_graph():
    return build_graph()


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
