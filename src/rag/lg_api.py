from __future__ import annotations

import json
import os
import uuid
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from rag.db import storage
from rag.rag_graph import build_graph


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\n" f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: List[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    texts.append(text)
        return " ".join(t for t in texts if t).strip()
    if isinstance(content, dict):
        # Fallback for single text block
        if content.get("type") == "text" and isinstance(content.get("text"), str):
            return str(content.get("text"))
    return str(content)


def _get_api_key(req: Request) -> Optional[str]:
    # SDK uses lowercase "x-api-key"
    return req.headers.get("x-api-key") or req.headers.get("X-Api-Key")


def current_user(req: Request):
    key = _get_api_key(req)
    if not key:
        raise HTTPException(status_code=401, detail="Missing x-api-key")
    user = storage().authenticate_api_token(key)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid x-api-key")
    return user


@lru_cache(maxsize=1)
def _rag():
    _app, ask, ask_stream = build_graph()
    return ask, ask_stream


def create_app() -> FastAPI:
    app = FastAPI(title="chatBot_RAG LangGraph-compatible API", version="0.1.0")
    storage().init_db()

    allow_origins = os.getenv("APP_CORS_ORIGINS", "http://localhost:3000").split(",")
    allow_origins = [o.strip() for o in allow_origins if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Location"],
    )

    @app.get("/info")
    def info():
        return {"ok": True, "name": "chatBot_RAG", "version": "0.1.0"}

    @app.post("/auth/login")
    async def login(payload: Dict[str, Any]):
        username = str(payload.get("username") or "")
        password = str(payload.get("password") or "")
        user = storage().authenticate(username, password)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid username/password")
        api_key = storage().create_api_token(user.id, name="agent-chat-ui")
        storage().log_audit(user.id, "api_token.create", user.username, {"name": "agent-chat-ui"})
        return {"api_key": api_key, "user": {"id": user.id, "username": user.username}}

    @app.post("/threads/search")
    async def threads_search(payload: Dict[str, Any], user=Depends(current_user)):
        meta = payload.get("metadata") or {}
        graph_id = meta.get("graph_id")
        assistant_id = meta.get("assistant_id")
        # We treat everything as a single local graph; accept any graph_id/assistant_id filter.
        limit = int(payload.get("limit") or 10)
        offset = int(payload.get("offset") or 0)
        limit = max(1, min(limit, 100))
        offset = max(0, offset)

        threads = storage().list_threads(user.id, limit=limit, offset=offset)
        result = []
        for t in threads:
            if graph_id and t.get("metadata", {}).get("graph_id") != graph_id:
                continue
            if assistant_id and t.get("metadata", {}).get("assistant_id") != assistant_id:
                continue
            result.append(t)
        return result

    @app.post("/threads")
    async def threads_create(payload: Dict[str, Any], user=Depends(current_user)):
        thread_id = payload.get("thread_id") or payload.get("threadId") or None
        if thread_id is None:
            thread_id = str(uuid.uuid4())

        meta = payload.get("metadata") or {}
        # SDK may send {graph_id: payload.graphId} or {assistant_id}; just store whatever.
        thread = storage().create_thread(
            user.id,
            thread_id=str(thread_id),
            metadata=meta,
        )
        storage().log_audit(user.id, "thread.create", thread["thread_id"], None)
        return thread

    @app.get("/threads/{thread_id}")
    async def threads_get(thread_id: str, user=Depends(current_user)):
        thread = storage().get_thread(user.id, thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")
        return thread

    @app.api_route("/threads/{thread_id}/history", methods=["GET", "POST"])
    async def threads_history(
        thread_id: str, limit: int = 10, user=Depends(current_user)
    ):
        # Minimal: return a single checkpoint containing full state.
        thread = storage().get_thread(user.id, thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")
        state = storage().get_thread_state(user.id, thread_id)
        checkpoint_id = state.get("checkpoint", {}).get("checkpoint_id") or "0"
        return [
            {
                "checkpoint": {"thread_id": thread_id, "checkpoint_id": str(checkpoint_id)},
                "parent_checkpoint": None,
                "values": state.get("values") or {"messages": []},
            }
        ]

    @app.api_route("/threads/{thread_id}/state", methods=["GET", "POST"])
    async def threads_state(thread_id: str, user=Depends(current_user)):
        state = storage().get_thread_state(user.id, thread_id)
        if not state:
            raise HTTPException(status_code=404, detail="Thread not found")
        return state

    @app.post("/threads/{thread_id}/runs/stream")
    async def runs_stream(thread_id: str, req: Request, user=Depends(current_user)):
        payload = await req.json()
        assistant_id = payload.get("assistant_id") or payload.get("assistantId") or "agent"
        input_obj = payload.get("input") or {}
        messages = input_obj.get("messages") or []
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="input.messages is required")
        last = messages[-1]
        question = _content_to_text(last.get("content")).strip()
        if not question:
            raise HTTPException(status_code=400, detail="Last message content is empty")

        # Persist human message into thread
        storage().append_thread_message(user.id, thread_id, role="human", content=question)
        storage().log_audit(user.id, "thread.message.human", thread_id, None)

        history = storage().get_thread_messages_as_lc(user.id, thread_id)
        # history includes the message we just saved, so exclude it for context.
        history_for_model = history[:-1]
        memory = storage().get_user_memory(user.id)

        ask, ask_stream = _rag()

        run_id = str(uuid.uuid4())
        content_location = f"/threads/{thread_id}/runs/{run_id}"

        async def gen():
            ai_msg_id = f"ai-{uuid.uuid4()}"
            answer_parts: List[str] = []
            try:
                for token in ask_stream(question, history=history_for_model, memory=memory):
                    answer_parts.append(str(token))
                    yield _sse("messages", ({"type": "ai", "content": str(token), "id": ai_msg_id}, None))
            except Exception as e:
                yield _sse("error", {"message": str(e)})
                return

            answer = "".join(answer_parts).strip()
            storage().append_thread_message(user.id, thread_id, role="ai", content=answer)
            storage().log_audit(user.id, "thread.message.ai", thread_id, {"chars": len(answer)})

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Content-Location": content_location},
        )

    return app


app = create_app()
