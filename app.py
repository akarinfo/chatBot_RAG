from __future__ import annotations

import sys
import traceback
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rag.ingest import ingest  # noqa: E402
from rag.kb import delete_kb_file, list_kb_files, save_upload  # noqa: E402
from rag.rag_graph import build_graph  # noqa: E402


st.set_page_config(page_title="RAG 知识库问答", layout="wide")


@st.cache_resource
def _rag():
    _app, ask, ask_stream = build_graph()
    return ask, ask_stream


def _reset_rag_cache():
    _rag.clear()


def sidebar_kb_manager():
    st.sidebar.header("知识库管理")

    uploaded = st.sidebar.file_uploader(
        "上传文档（.txt/.md/.mdx）",
        type=["txt", "md", "mdx"],
        accept_multiple_files=True,
    )
    if uploaded and st.sidebar.button("保存上传文件"):
        for f in uploaded:
            save_upload(f.name, f.getvalue())
        st.sidebar.success("已保存到 data/（需要重新入库）")

    files = list_kb_files()
    st.sidebar.subheader("当前文档")
    if not files:
        st.sidebar.caption("data/ 里还没有文档")
    else:
        selected = st.sidebar.selectbox(
            "选择一个文档", options=[f.name for f in files], index=0
        )
        if st.sidebar.button("删除选中文档", type="secondary"):
            delete_kb_file(selected)
            st.sidebar.success(f"已删除：{selected}（需要重新入库）")

    st.sidebar.divider()
    st.sidebar.subheader("索引/入库")
    if st.sidebar.button("重新入库（重建向量库）", type="primary"):
        with st.sidebar.status("正在入库...", expanded=True):
            try:
                ingest()
                _reset_rag_cache()
                st.sidebar.success("入库完成，可以开始提问了")
            except Exception:
                st.sidebar.error("入库失败")
                st.sidebar.code(traceback.format_exc())


def main_chat():
    st.title("RAG 知识库问答机器人")
    st.caption("流程：检索（Weaviate）→ 生成（DeepSeek LLM），回答只基于知识库上下文。")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    col1, col2 = st.columns([1, 1], vertical_alignment="center")
    with col1:
        if st.button("清空对话"):
            st.session_state.messages = []
            st.rerun()
    with col2:
        st.write("")

    for msg in st.session_state.messages:
        role = "assistant" if isinstance(msg, AIMessage) else "user"
        with st.chat_message(role):
            st.markdown(msg.content)

    prompt = st.chat_input("请输入你的问题…")
    if not prompt:
        return

    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    ask, ask_stream = _rag()
    with st.chat_message("assistant"):
        try:
            def _gen():
                for token in ask_stream(prompt, history=st.session_state.messages[:-1]):
                    yield token

            content = st.write_stream(_gen())
        except Exception:
            # fallback to non-stream call
            ai = ask(prompt, history=st.session_state.messages[:-1])
            content = ai.content
            st.markdown(content)

    st.session_state.messages.append(AIMessage(content=str(content)))


def main():
    sidebar_kb_manager()
    main_chat()


if __name__ == "__main__":
    main()
