from __future__ import annotations

import sys
import traceback
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from langchain_core.messages import AIMessage, HumanMessage

# 允许从项目根目录直接运行 app.py
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from core.db import storage  # noqa: E402
from core.vectordb import weaviate_client_from_env  # noqa: E402
from services.ingest.chunking import chunk_file  # noqa: E402
from services.ingest.processor import ingest  # noqa: E402
from services.kb import delete_kb_file, list_kb_files, save_upload  # noqa: E402
from workflows.rag_bot.graph import build_graph  # noqa: E402


st.set_page_config(page_title="RAG 知识库问答", layout="wide")


@st.cache_resource
def _rag():
    _app, ask, ask_stream = build_graph()
    return ask, ask_stream


def _reset_rag_cache():
    # 重新入库后需要重建 retriever/LLM 等资源
    _rag.clear()


def _logout():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()


def require_login():
    s = storage()
    s.init_db()
    if "user" in st.session_state and st.session_state.user is not None:
        return st.session_state.user

    st.title("登录")

    if not s.has_any_users():
        st.info("首次启动：请创建管理员账号。")
        with st.form("bootstrap_admin"):
            username = st.text_input("管理员用户名", placeholder="admin")
            password = st.text_input("管理员密码（至少 8 位）", type="password")
            password2 = st.text_input("确认密码", type="password")
            submitted = st.form_submit_button("创建管理员")
        if submitted:
            if password != password2:
                st.error("两次密码不一致")
            else:
                try:
                    user = s.create_user(username, password, is_admin=True)
                    s.log_audit(user.id, "user.bootstrap_admin", user.username, None)
                    st.session_state.user = user
                    st.success("管理员创建成功")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
        st.stop()

    with st.form("login"):
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        submitted = st.form_submit_button("登录")
    if submitted:
        user = s.authenticate(username, password)
        if not user:
            st.error("用户名或密码错误")
        else:
            s.log_audit(user.id, "user.login", user.username, None)
            st.session_state.user = user
            st.rerun()

    st.stop()


def _weaviate_chunk_count_for_source(source_path: str) -> int | None:
    import os

    class_name = os.getenv("WEAVIATE_CLASS", "RAGChunk")
    try:
        client = weaviate_client_from_env()
        if not client.schema.exists(class_name):
            return None
        where_filter = {"path": ["source"], "operator": "Equal", "valueString": source_path}
        res = (
            client.query.aggregate(class_name)
            .with_where(where_filter)
            .with_fields("meta { count }")
            .do()
        )
        return int(res["data"]["Aggregate"][class_name][0]["meta"]["count"])
    except Exception:
        return None


def sidebar_nav(user):
    st.sidebar.header("导航")
    st.sidebar.caption(f"当前用户：{user.username}" + ("（管理员）" if user.is_admin else ""))
    if st.sidebar.button("退出登录", type="secondary"):
        _logout()

    pages = ["聊天", "文档管理", "用户记忆"]
    if user.is_admin:
        pages.append("用户管理")
    page = st.sidebar.radio("页面", pages, index=0)

    if page == "聊天":
        st.sidebar.divider()
        st.sidebar.subheader("对话")
        s = storage()
        convs = s.list_conversations(user.id)
        if not convs:
            conv_id = s.create_conversation(user.id, "新对话")
            st.session_state.active_conversation_id = conv_id
            st.rerun()

        convs = s.list_conversations(user.id)
        labels = [f"{c.id} · {c.title}  ·  {c.updated_at}" for c in convs]
        id_by_label = {labels[i]: convs[i].id for i in range(len(convs))}

        current_id = st.session_state.get("active_conversation_id", convs[0].id)
        current_label = next((lab for lab, cid in id_by_label.items() if cid == current_id), labels[0])
        selected_label = st.sidebar.selectbox("选择对话", options=labels, index=labels.index(current_label))
        st.session_state.active_conversation_id = id_by_label[selected_label]

        col1, col2 = st.sidebar.columns([1, 1])
        with col1:
            if st.button("新建对话"):
                conv_id = s.create_conversation(user.id, "新对话")
                st.session_state.active_conversation_id = conv_id
                st.rerun()
        with col2:
            if st.button("删除对话", type="secondary"):
                s.delete_conversation(user.id, int(st.session_state.active_conversation_id))
                st.session_state.active_conversation_id = None
                st.rerun()

    return page


def page_chat(user):
    st.title("RAG 知识库问答机器人")
    st.caption("流程：检索（Weaviate）→ 生成（DeepSeek LLM），回答只基于知识库上下文。")

    conv_id = st.session_state.get("active_conversation_id")
    if not conv_id:
        s = storage()
        conv_id = s.create_conversation(user.id, "新对话")
        st.session_state.active_conversation_id = conv_id

    s = storage()
    stored = s.list_messages(user.id, int(conv_id))
    history = []
    for msg in stored:
        if msg.role == "assistant":
            history.append(AIMessage(content=msg.content))
        else:
            history.append(HumanMessage(content=msg.content))

    for msg in history:
        role = "assistant" if isinstance(msg, AIMessage) else "user"
        with st.chat_message(role):
            st.markdown(msg.content)

    prompt = st.chat_input("请输入你的问题…")
    if not prompt:
        return

    s.add_message(user.id, int(conv_id), role="user", content=prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    ask, ask_stream = _rag()
    with st.chat_message("assistant"):
        try:
            # 优先使用流式输出
            def _gen():
                memory = s.get_user_memory(user.id)
                for token in ask_stream(prompt, history=history, memory=memory):
                    yield token

            content = st.write_stream(_gen())
        except Exception:
            # 兼容不支持 stream 的情况
            memory = s.get_user_memory(user.id)
            ai = ask(prompt, history=history, memory=memory)
            content = ai.content
            st.markdown(content)

    s.add_message(user.id, int(conv_id), role="assistant", content=str(content))


def page_memory(user):
    st.title("用户记忆")
    st.caption("这里存放与你相关、需要长期记住的信息（偏好/背景/约束等）。")

    s = storage()
    memory = s.get_user_memory(user.id)
    updated = st.text_area("记忆内容", value=memory, height=260)
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("保存记忆", type="primary"):
            s.set_user_memory(user.id, updated)
            s.log_audit(user.id, "user.memory.update", user.username, None)
            st.success("已保存")
    with col2:
        if st.button("清空记忆", type="secondary"):
            s.set_user_memory(user.id, "")
            s.log_audit(user.id, "user.memory.clear", user.username, None)
            st.success("已清空")
            st.rerun()


def page_users(user):
    if not user.is_admin:
        st.error("无权限")
        return

    st.title("用户管理")
    st.caption("仅管理员可创建用户。")

    st.subheader("创建用户")
    with st.form("create_user"):
        username = st.text_input("用户名")
        password = st.text_input("密码（至少 8 位）", type="password")
        is_admin = st.checkbox("设为管理员", value=False)
        submitted = st.form_submit_button("创建")
    if submitted:
        try:
            s = storage()
            created = s.create_user(username, password, is_admin=is_admin)
            s.log_audit(user.id, "user.create", created.username, {"is_admin": is_admin})
            st.success("用户已创建")
            st.rerun()
        except Exception as e:
            st.error(str(e))

    st.subheader("现有用户")
    s = storage()
    rows = [{"username": u.username, "is_admin": u.is_admin} for u in s.list_users()]
    st.table(rows)


def page_docs(user):
    st.title("文档管理")
    st.caption(
        "全局共享一套向量库。左侧选择文件，右侧预览原文与 chunks；上传/删除/重新入库都会记录审计日志。"
    )

    s = storage()

    if user.is_admin:
        with st.expander("权限设置（管理员）", expanded=False):
            delete_policy = s.get_kb_delete_policy()
            reindex_policy = s.get_kb_reindex_policy()
            delete_label = {
                "admin_only": "仅管理员可删除",
                "all_users": "所有用户可删除",
                "uploader_only": "上传者可删除（管理员也可删）",
            }
            reindex_label = {
                "admin_only": "仅管理员可重新入库",
                "all_users": "所有用户可重新入库",
            }
            new_delete = st.selectbox(
                "删除权限",
                options=list(delete_label.keys()),
                index=list(delete_label.keys()).index(delete_policy),
                format_func=lambda k: delete_label[k],
            )
            new_reindex = st.selectbox(
                "重新入库权限",
                options=list(reindex_label.keys()),
                index=list(reindex_label.keys()).index(reindex_policy),
                format_func=lambda k: reindex_label[k],
            )
            if st.button("保存权限设置", type="primary"):
                s.set_setting(user.id, "kb_delete_policy", new_delete)
                s.set_setting(user.id, "kb_reindex_policy", new_reindex)
                s.log_audit(
                    user.id,
                    "kb.policy.update",
                    "kb",
                    {"delete": new_delete, "reindex": new_reindex},
                )
                st.success("已保存")
                st.rerun()

    col_files, col_preview = st.columns([0.30, 0.70], gap="large")

    with col_files:
        st.subheader("文件")
        uploaded = st.file_uploader(
            "上传文档（.txt/.md/.mdx）",
            type=["txt", "md", "mdx"],
            accept_multiple_files=True,
        )
        if uploaded and st.button("保存上传文件", type="primary"):
            for f in uploaded:
                content = f.getvalue()
                target = save_upload(f.name, content)
                s.upsert_kb_file(
                    target.name, uploader_user_id=user.id, size_bytes=len(content)
                )
                s.log_audit(
                    user.id,
                    "kb.upload",
                    target.name,
                    {"size_bytes": len(content)},
                )
            st.success("已保存到 data/（需要重新入库）")
            st.rerun()

        st.divider()
        files = list_kb_files()
        if not files:
            st.info("data/ 里还没有文档")
            return

        query = st.text_input("搜索文件", placeholder="输入文件名关键词")
        file_names = [f.name for f in files]
        if query.strip():
            q = query.strip().lower()
            file_names = [n for n in file_names if q in n.lower()]
            if not file_names:
                st.warning("没有匹配的文件")
                return

        active_name = st.session_state.get("kb_active_file") or file_names[0]
        if active_name not in file_names:
            active_name = file_names[0]
            st.session_state.kb_active_file = active_name

        with st.container(height=420):
            selected_name = st.radio(
                "文件列表",
                options=file_names,
                index=file_names.index(active_name),
                label_visibility="collapsed",
            )
        st.session_state.kb_active_file = selected_name
        selected = next(f for f in files if f.name == selected_name)

        meta = s.get_kb_file_meta(selected.name)
        if meta:
            st.caption(
                f"上传者 user_id：{meta.uploader_user_id if meta.uploader_user_id is not None else '未知'}"
                f" · 上传时间：{meta.uploaded_at}"
            )
        else:
            st.caption("元数据：未记录（可能是历史文件）。")

        can_delete = s.can_delete_kb_file(user, meta)
        can_reindex = s.can_reindex_kb(user)
        b1, b2 = st.columns([1, 1])
        with b1:
            if st.button(
                "删除",
                type="secondary",
                disabled=not can_delete,
                help=None if can_delete else "当前权限不允许删除",
            ):
                delete_kb_file(selected_name)
                s.log_audit(user.id, "kb.delete", selected_name, None)
                st.success(f"已删除：{selected_name}（需要重新入库）")
                st.rerun()
        with b2:
            if st.button(
                "重新入库",
                type="primary",
                disabled=not can_reindex,
                help=None if can_reindex else "当前权限不允许重新入库",
            ):
                s.log_audit(user.id, "kb.reindex.start", "kb", None)
                with st.status("正在入库...", expanded=True):
                    try:
                        ingest()
                        _reset_rag_cache()
                        s.log_audit(user.id, "kb.reindex.ok", "kb", None)
                        st.success("入库完成")
                    except Exception:
                        s.log_audit(
                            user.id,
                            "kb.reindex.fail",
                            "kb",
                            {"traceback": traceback.format_exc()},
                        )
                        st.error("入库失败")
                        st.code(traceback.format_exc())

        count = _weaviate_chunk_count_for_source(str(selected.path))
        if count is None:
            st.caption("向量库 chunk 数：未知（可能未入库或 Weaviate 未就绪）")
        else:
            st.caption(f"向量库 chunk 数：{count}")

    with col_preview:
        st.subheader("预览")
        method = st.selectbox(
            "切分方法",
            options=["auto", "recursive_only"],
            index=0,
            format_func=lambda m: "Markdown 标题分块 + 递归分块（auto）"
            if m == "auto"
            else "仅递归分块（recursive_only）",
        )
        chunk_size = st.number_input(
            "chunk_size", min_value=200, max_value=4000, value=800, step=50
        )
        chunk_overlap = st.number_input(
            "chunk_overlap",
            min_value=0,
            max_value=int(chunk_size) - 1,
            value=120,
            step=10,
        )

        try:
            chunks = chunk_file(
                selected.path,
                chunk_size=int(chunk_size),
                chunk_overlap=int(chunk_overlap),
                method=method,
                include_preview_metadata=True,
                include_spans=True,
            )
        except Exception:
            st.error("分块失败（请确认文件编码为 UTF-8）")
            st.code(traceback.format_exc())
            return

        original_text = selected.path.read_text(encoding="utf-8")
        st.caption(f"共 {len(chunks)} 个 chunk（点击 chunk 会在原文中定位并高亮）")

        import json as _json

        chunk_rows = []
        for c in chunks:
            meta2 = c.metadata or {}
            title_bits = [
                f"#{meta2.get('chunk_index', '-')}",
                f"{meta2.get('chunk_chars', 0)} chars",
            ]
            for k in ("h1", "h2", "h3"):
                if meta2.get(k):
                    title_bits.append(f"{k}:{meta2.get(k)}")
            title = " · ".join(title_bits)
            chunk_rows.append(
                {
                    "title": title,
                    "content": c.page_content or "",
                    "start": meta2.get("span_start"),
                    "end": meta2.get("span_end"),
                }
            )

        data_json = _json.dumps(
            {"original": original_text, "chunks": chunk_rows}, ensure_ascii=False
        ).replace("</", "<\\/")

        components.html(
            f"""
        <div id="rag-preview-root">
          <style>
            #rag-preview-root {{
              font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
            }}
            .wrap {{
              display: flex;
              gap: 12px;
              height: 720px;
            }}
            .panel {{
              border: 1px solid #e5e7eb;
              border-radius: 10px;
              overflow: hidden;
              display: flex;
              flex-direction: column;
              min-width: 0;
            }}
            .panel-header {{
              padding: 10px 12px;
              font-weight: 600;
              background: #f9fafb;
              border-bottom: 1px solid #e5e7eb;
            }}
            .panel-body {{
              flex: 1;
              overflow: auto;
              padding: 12px;
              background: #fff;
            }}
            pre {{
              margin: 0;
              white-space: pre-wrap;
              word-break: break-word;
              line-height: 1.5;
              font-size: 13px;
            }}
            mark {{
              background: #fde68a;
              border-radius: 4px;
              padding: 0 2px;
            }}
            .chunks {{
              display: flex;
              flex-direction: column;
              gap: 8px;
            }}
            .chunk-item {{
              border: 1px solid #e5e7eb;
              border-radius: 10px;
              padding: 10px;
              cursor: pointer;
              background: #fff;
            }}
            .chunk-item.active {{
              border-color: #60a5fa;
              box-shadow: 0 0 0 3px rgba(96, 165, 250, 0.25);
            }}
            .chunk-title {{
              font-weight: 600;
              font-size: 13px;
              margin-bottom: 6px;
            }}
            .chunk-snippet {{
              font-size: 12px;
              color: #374151;
              opacity: 0.9;
              max-height: 4.5em;
              overflow: hidden;
            }}
            .detail {{
              border-top: 1px solid #e5e7eb;
              padding: 10px 12px;
              background: #f9fafb;
            }}
            .detail pre {{
              max-height: 200px;
              overflow: auto;
              background: #fff;
              border: 1px solid #e5e7eb;
              border-radius: 10px;
              padding: 10px;
            }}
          </style>

          <div class="wrap">
            <div class="panel" style="flex: 1.2">
              <div class="panel-header">原文</div>
              <div class="panel-body"><pre id="doc"></pre></div>
            </div>
            <div class="panel" style="flex: 1.0">
              <div class="panel-header">Chunks（点击定位/高亮）</div>
              <div class="panel-body">
                <div id="chunkList" class="chunks"></div>
              </div>
              <div class="detail">
                <div style="font-weight:600; margin-bottom:6px;">Chunk 内容</div>
                <pre id="chunkDetail"></pre>
              </div>
            </div>
          </div>
        </div>

        <script>
          const data = {data_json};
          const original = data.original || "";
          const chunks = data.chunks || [];

          const docEl = document.getElementById("doc");
          const listEl = document.getElementById("chunkList");
          const detailEl = document.getElementById("chunkDetail");

          function esc(s) {{
            return (s || "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
          }}

          function renderDoc(highlight) {{
            if (!highlight || highlight.start == null || highlight.end == null) {{
              docEl.innerHTML = esc(original);
              return;
            }}
            const s = Math.max(0, Math.min(original.length, highlight.start));
            const e = Math.max(0, Math.min(original.length, highlight.end));
            if (e <= s) {{
              docEl.innerHTML = esc(original);
              return;
            }}
            docEl.innerHTML = esc(original.slice(0, s)) + '<mark id="hl">' + esc(original.slice(s, e)) + "</mark>" + esc(original.slice(e));
            const hl = document.getElementById("hl");
            if (hl) {{
              hl.scrollIntoView({{block: "center"}});
            }}
          }}

          function renderList() {{
            listEl.innerHTML = "";
            chunks.forEach((c, idx) => {{
              const div = document.createElement("div");
              div.className = "chunk-item";
              div.dataset.index = String(idx);
              const snippet = (c.content || "").slice(0, 240);
              div.innerHTML = '<div class="chunk-title">' + esc(c.title || ("Chunk " + idx)) + "</div>" +
                              '<div class="chunk-snippet">' + esc(snippet) + ( (c.content||"").length > 240 ? "…" : "" ) + "</div>";
              div.addEventListener("click", () => selectChunk(idx));
              listEl.appendChild(div);
            }});
          }}

          function selectChunk(idx) {{
            const items = listEl.querySelectorAll(".chunk-item");
            items.forEach(el => el.classList.remove("active"));
            const active = listEl.querySelector('.chunk-item[data-index="' + idx + '"]');
            if (active) {{
              active.classList.add("active");
              active.scrollIntoView({{block: "nearest"}});
            }}
            const c = chunks[idx];
            detailEl.textContent = c && c.content ? c.content : "";
            renderDoc(c);
          }}

          renderList();
          renderDoc(null);
          if (chunks.length) {{
            selectChunk(0);
          }}
        </script>
        """,
            height=760,
            scrolling=False,
        )

    st.divider()
    with st.expander("操作记录（默认折叠）", expanded=False):
        events = s.list_audit_events(limit=500)
        user_map = {u.id: u.username for u in s.list_users()}
        rows = []
        for e in events:
            rows.append(
                {
                    "id": e.id,
                    "time": e.created_at,
                    "user": user_map.get(e.user_id, str(e.user_id))
                    if e.user_id is not None
                    else "unknown",
                    "action": e.action,
                    "target": e.target,
                    "details": e.details,
                }
            )
        st.dataframe(rows, use_container_width=True, height=320)

        import csv as _csv
        import io as _io

        buf = _io.StringIO()
        w = _csv.DictWriter(
            buf, fieldnames=["id", "time", "user", "action", "target", "details"]
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
        st.download_button(
            "导出 CSV",
            data=buf.getvalue().encode("utf-8"),
            file_name="audit_log.csv",
            mime="text/csv",
        )


def main():
    user = require_login()
    page = sidebar_nav(user)
    if page == "聊天":
        page_chat(user)
    elif page == "文档管理":
        page_docs(user)
    elif page == "用户记忆":
        page_memory(user)
    elif page == "用户管理":
        page_users(user)


if __name__ == "__main__":
    main()
