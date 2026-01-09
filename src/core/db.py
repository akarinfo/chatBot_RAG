from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Literal


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = BASE_DIR / "storage" / "app.db"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _db_path() -> Path:
    path = Path(os.getenv("APP_DB_PATH", str(DEFAULT_DB_PATH)))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    with connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_salt BLOB NOT NULL,
                password_hash BLOB NOT NULL,
                is_admin INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS user_memory (
                user_id INTEGER PRIMARY KEY,
                memory TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                thread_id TEXT,
                thread_metadata TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user','assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS kb_files (
                path TEXT PRIMARY KEY,
                uploader_user_id INTEGER,
                size_bytes INTEGER NOT NULL DEFAULT 0,
                uploaded_at TEXT NOT NULL,
                FOREIGN KEY(uploader_user_id) REFERENCES users(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                action TEXT NOT NULL,
                target TEXT NOT NULL,
                details TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                updated_by_user_id INTEGER,
                FOREIGN KEY(updated_by_user_id) REFERENCES users(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS api_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token_hash BLOB NOT NULL UNIQUE,
                token_prefix TEXT NOT NULL,
                name TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                revoked_at TEXT,
                last_used_at TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            """
        )

        def _ensure_column(table: str, col: str, ddl_fragment: str) -> None:
            cols = [r["name"] for r in conn.execute(f"PRAGMA table_info({table});").fetchall()]
            if col in cols:
                return
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl_fragment};")

        # Migrations: add missing columns for existing DBs.
        _ensure_column("conversations", "thread_id", "thread_id TEXT")
        _ensure_column(
            "conversations",
            "thread_metadata",
            "thread_metadata TEXT NOT NULL DEFAULT ''",
        )

        # Indices / migrations for thread_id uniqueness and backfill.
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_conversations_thread_id ON conversations(thread_id);"
        )
        rows = conn.execute(
            "SELECT id FROM conversations WHERE thread_id IS NULL OR thread_id = '';"
        ).fetchall()
        for r in rows:
            conn.execute(
                "UPDATE conversations SET thread_id = ? WHERE id = ?;",
                (str(uuid.uuid4()), int(r["id"])),
            )


@dataclass(frozen=True)
class User:
    id: int
    username: str
    is_admin: bool


def _hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)


def has_any_users() -> bool:
    with connect() as conn:
        row = conn.execute("SELECT 1 FROM users LIMIT 1;").fetchone()
        return row is not None


def create_user(username: str, password: str, *, is_admin: bool) -> User:
    username = username.strip()
    if not username:
        raise ValueError("username is required")
    if len(password) < 8:
        raise ValueError("password must be at least 8 characters")

    salt = secrets.token_bytes(16)
    pwd_hash = _hash_password(password, salt)
    created_at = _utc_now_iso()

    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO users (username, password_salt, password_hash, is_admin, created_at)
            VALUES (?, ?, ?, ?, ?);
            """,
            (username, salt, pwd_hash, 1 if is_admin else 0, created_at),
        )
        user_id = int(cur.lastrowid)
        # Ensure memory row exists
        conn.execute(
            """
            INSERT OR IGNORE INTO user_memory (user_id, memory, updated_at)
            VALUES (?, '', ?);
            """,
            (user_id, created_at),
        )
        return User(id=user_id, username=username, is_admin=is_admin)


def authenticate(username: str, password: str) -> Optional[User]:
    username = username.strip()
    if not username or not password:
        return None
    with connect() as conn:
        row = conn.execute(
            """
            SELECT id, username, password_salt, password_hash, is_admin
            FROM users
            WHERE username = ?;
            """,
            (username,),
        ).fetchone()
        if row is None:
            return None
        salt = bytes(row["password_salt"])
        expected = bytes(row["password_hash"])
        got = _hash_password(password, salt)
        if not secrets.compare_digest(expected, got):
            return None
        return User(
            id=int(row["id"]),
            username=str(row["username"]),
            is_admin=bool(int(row["is_admin"])),
        )


def _hash_token(token: str) -> bytes:
    return hashlib.sha256(token.encode("utf-8")).digest()


def create_api_token(user_id: int, name: str = "") -> str:
    token = secrets.token_urlsafe(32)
    token_hash = _hash_token(token)
    prefix = token[:8]
    now = _utc_now_iso()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO api_tokens (user_id, token_hash, token_prefix, name, created_at)
            VALUES (?, ?, ?, ?, ?);
            """,
            (user_id, token_hash, prefix, name or "", now),
        )
    return token


def authenticate_api_token(token: str) -> Optional[User]:
    token = (token or "").strip()
    if not token:
        return None
    token_hash = _hash_token(token)
    now = _utc_now_iso()
    with connect() as conn:
        row = conn.execute(
            """
            SELECT u.id, u.username, u.is_admin, t.id as token_id
            FROM api_tokens t
            JOIN users u ON u.id = t.user_id
            WHERE t.token_hash = ? AND t.revoked_at IS NULL;
            """,
            (token_hash,),
        ).fetchone()
        if row is None:
            return None
        conn.execute(
            "UPDATE api_tokens SET last_used_at = ? WHERE id = ?;",
            (now, int(row["token_id"])),
        )
        return User(
            id=int(row["id"]),
            username=str(row["username"]),
            is_admin=bool(int(row["is_admin"])),
        )


def list_users() -> List[User]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, username, is_admin FROM users ORDER BY username ASC;"
        ).fetchall()
        return [
            User(
                id=int(r["id"]),
                username=str(r["username"]),
                is_admin=bool(int(r["is_admin"])),
            )
            for r in rows
        ]


def get_user_memory(user_id: int) -> str:
    with connect() as conn:
        row = conn.execute(
            "SELECT memory FROM user_memory WHERE user_id = ?;",
            (user_id,),
        ).fetchone()
        return str(row["memory"]) if row else ""


def set_user_memory(user_id: int, memory: str) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO user_memory (user_id, memory, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                memory = excluded.memory,
                updated_at = excluded.updated_at;
            """,
            (user_id, memory or "", _utc_now_iso()),
        )


@dataclass(frozen=True)
class Conversation:
    id: int
    title: str
    updated_at: str


def list_conversations(user_id: int) -> List[Conversation]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT id, title, updated_at
            FROM conversations
            WHERE user_id = ?
            ORDER BY updated_at DESC;
            """,
            (user_id,),
        ).fetchall()
        return [
            Conversation(
                id=int(r["id"]),
                title=str(r["title"]),
                updated_at=str(r["updated_at"]),
            )
            for r in rows
        ]


def create_conversation(user_id: int, title: str) -> int:
    title = (title or "").strip() or "新对话"
    now = _utc_now_iso()
    thread_id = str(uuid.uuid4())
    meta = json.dumps({"graph_id": "agent"}, ensure_ascii=False)
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO conversations (user_id, title, thread_id, thread_metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (user_id, title, thread_id, meta, now, now),
        )
        return int(cur.lastrowid)


def delete_conversation(user_id: int, conversation_id: int) -> None:
    with connect() as conn:
        conn.execute(
            "DELETE FROM conversations WHERE id = ? AND user_id = ?;",
            (conversation_id, user_id),
        )


@dataclass(frozen=True)
class StoredMessage:
    role: str
    content: str
    created_at: str


@dataclass(frozen=True)
class KBFileMeta:
    path: str
    uploader_user_id: Optional[int]
    size_bytes: int
    uploaded_at: str


@dataclass(frozen=True)
class AuditEvent:
    id: int
    user_id: Optional[int]
    action: str
    target: str
    details: str
    created_at: str


KBDeletePolicy = Literal["admin_only", "all_users", "uploader_only"]
KBReindexPolicy = Literal["admin_only", "all_users"]


def _audit_details(details: object | None) -> str:
    if details is None:
        return ""
    try:
        return json.dumps(details, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(details)


def _get_setting(key: str) -> Optional[str]:
    with connect() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key = ?;", (key,)).fetchone()
        return str(row["value"]) if row else None


def get_kb_delete_policy() -> KBDeletePolicy:
    val = (_get_setting("kb_delete_policy") or "admin_only").strip()
    if val not in {"admin_only", "all_users", "uploader_only"}:
        return "admin_only"
    return val  # type: ignore[return-value]


def get_kb_reindex_policy() -> KBReindexPolicy:
    val = (_get_setting("kb_reindex_policy") or "admin_only").strip()
    if val not in {"admin_only", "all_users"}:
        return "admin_only"
    return val  # type: ignore[return-value]


def set_setting(user_id: int, key: str, value: str) -> None:
    now = _utc_now_iso()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO settings (key, value, updated_at, updated_by_user_id)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at,
                updated_by_user_id = excluded.updated_by_user_id;
            """,
            (key, value, now, user_id),
        )


def log_audit(user_id: Optional[int], action: str, target: str, details: object | None = None) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO audit_log (user_id, action, target, details, created_at)
            VALUES (?, ?, ?, ?, ?);
            """,
            (user_id, action, target, _audit_details(details), _utc_now_iso()),
        )


def list_audit_events(limit: int = 200) -> List[AuditEvent]:
    limit = max(1, min(int(limit), 5000))
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT id, user_id, action, target, details, created_at
            FROM audit_log
            ORDER BY id DESC
            LIMIT ?;
            """,
            (limit,),
        ).fetchall()
        return [
            AuditEvent(
                id=int(r["id"]),
                user_id=int(r["user_id"]) if r["user_id"] is not None else None,
                action=str(r["action"]),
                target=str(r["target"]),
                details=str(r["details"] or ""),
                created_at=str(r["created_at"]),
            )
            for r in rows
        ]


def upsert_kb_file(path: str, *, uploader_user_id: Optional[int], size_bytes: int) -> None:
    now = _utc_now_iso()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO kb_files (path, uploader_user_id, size_bytes, uploaded_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                uploader_user_id = excluded.uploader_user_id,
                size_bytes = excluded.size_bytes,
                uploaded_at = excluded.uploaded_at;
            """,
            (path, uploader_user_id, int(size_bytes), now),
        )


def get_kb_file_meta(path: str) -> Optional[KBFileMeta]:
    with connect() as conn:
        row = conn.execute(
            "SELECT path, uploader_user_id, size_bytes, uploaded_at FROM kb_files WHERE path = ?;",
            (path,),
        ).fetchone()
        if not row:
            return None
        return KBFileMeta(
            path=str(row["path"]),
            uploader_user_id=int(row["uploader_user_id"]) if row["uploader_user_id"] is not None else None,
            size_bytes=int(row["size_bytes"] or 0),
            uploaded_at=str(row["uploaded_at"]),
        )


def can_delete_kb_file(user: User, meta: Optional[KBFileMeta]) -> bool:
    policy = get_kb_delete_policy()
    if policy == "all_users":
        return True
    if policy == "uploader_only":
        if user.is_admin:
            return True
        return bool(meta and meta.uploader_user_id == user.id)
    return bool(user.is_admin)


def can_reindex_kb(user: User) -> bool:
    policy = get_kb_reindex_policy()
    if policy == "all_users":
        return True
    return bool(user.is_admin)


def list_messages(user_id: int, conversation_id: int) -> List[StoredMessage]:
    with connect() as conn:
        # Ensure ownership
        owner = conn.execute(
            "SELECT 1 FROM conversations WHERE id = ? AND user_id = ?;",
            (conversation_id, user_id),
        ).fetchone()
        if owner is None:
            return []
        rows = conn.execute(
            """
            SELECT role, content, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC;
            """,
            (conversation_id,),
        ).fetchall()
        return [
            StoredMessage(
                role=str(r["role"]),
                content=str(r["content"]),
                created_at=str(r["created_at"]),
            )
            for r in rows
        ]


def add_message(user_id: int, conversation_id: int, *, role: str, content: str) -> None:
    if role not in {"user", "assistant"}:
        raise ValueError("role must be 'user' or 'assistant'")
    now = _utc_now_iso()
    with connect() as conn:
        owner = conn.execute(
            "SELECT 1 FROM conversations WHERE id = ? AND user_id = ?;",
            (conversation_id, user_id),
        ).fetchone()
        if owner is None:
            raise ValueError("conversation not found")
        conn.execute(
            """
            INSERT INTO messages (conversation_id, role, content, created_at)
            VALUES (?, ?, ?, ?);
            """,
            (conversation_id, role, content, now),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?;",
            (now, conversation_id),
        )


def _get_conversation_id_by_thread_id(
    conn: sqlite3.Connection, user_id: int, thread_id: str
) -> Optional[int]:
    row = conn.execute(
        "SELECT id FROM conversations WHERE user_id = ? AND thread_id = ?;",
        (user_id, thread_id),
    ).fetchone()
    return int(row["id"]) if row else None


def _load_thread_row(
    conn: sqlite3.Connection, user_id: int, thread_id: str
) -> Optional[sqlite3.Row]:
    return conn.execute(
        """
        SELECT id, thread_id, thread_metadata, created_at, updated_at, title
        FROM conversations
        WHERE user_id = ? AND thread_id = ?;
        """,
        (user_id, thread_id),
    ).fetchone()


def _parse_thread_metadata(raw: str) -> Dict[str, Any]:
    raw = (raw or "").strip()
    if not raw:
        return {"graph_id": "agent"}
    try:
        val = json.loads(raw)
        return val if isinstance(val, dict) else {"graph_id": "agent"}
    except Exception:
        return {"graph_id": "agent"}


def list_threads(user_id: int, *, limit: int, offset: int) -> List[Dict[str, Any]]:
    limit = max(1, min(int(limit), 100))
    offset = max(0, int(offset))
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT id, thread_id, thread_metadata, created_at, updated_at
            FROM conversations
            WHERE user_id = ? AND thread_id IS NOT NULL AND thread_id != ''
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?;
            """,
            (user_id, limit, offset),
        ).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            conv_id = int(r["id"])
            first = conn.execute(
                """
                SELECT id, role, content
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id ASC
                LIMIT 1;
                """,
                (conv_id,),
            ).fetchone()
            messages: List[Dict[str, Any]] = []
            if first:
                msg_type = "human" if str(first["role"]) == "user" else "ai"
                messages.append(
                    {
                        "type": msg_type,
                        "content": str(first["content"]),
                        "id": f"m-{int(first['id'])}",
                    }
                )
            out.append(
                {
                    "thread_id": str(r["thread_id"]),
                    "metadata": _parse_thread_metadata(str(r["thread_metadata"])),
                    "values": {"messages": messages},
                    "created_at": str(r["created_at"]),
                    "updated_at": str(r["updated_at"]),
                }
            )
        return out


def create_thread(user_id: int, *, thread_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    thread_id = (thread_id or "").strip()
    if not thread_id:
        raise ValueError("thread_id is required")
    now = _utc_now_iso()
    meta = metadata or {}
    meta.setdefault("graph_id", meta.get("graph_id") or "agent")
    meta_raw = json.dumps(meta, ensure_ascii=False)
    title = str(meta.get("title") or "新对话")
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO conversations (user_id, title, thread_id, thread_metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (user_id, title, thread_id, meta_raw, now, now),
        )
    return {
        "thread_id": thread_id,
        "metadata": meta,
        "values": {"messages": []},
        "created_at": now,
        "updated_at": now,
    }


def get_thread(user_id: int, thread_id: str) -> Optional[Dict[str, Any]]:
    with connect() as conn:
        row = _load_thread_row(conn, user_id, thread_id)
        if row is None:
            return None
        return {
            "thread_id": str(row["thread_id"]),
            "metadata": _parse_thread_metadata(str(row["thread_metadata"])),
            "values": {"messages": []},
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
            "title": str(row["title"]),
        }


def get_thread_state(user_id: int, thread_id: str) -> Optional[Dict[str, Any]]:
    with connect() as conn:
        conv = _load_thread_row(conn, user_id, thread_id)
        if conv is None:
            return None
        conv_id = int(conv["id"])
        rows = conn.execute(
            """
            SELECT id, role, content
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC;
            """,
            (conv_id,),
        ).fetchall()
        msgs: List[Dict[str, Any]] = []
        last_id = 0
        for r in rows:
            last_id = int(r["id"])
            msg_type = "human" if str(r["role"]) == "user" else "ai"
            msgs.append(
                {
                    "type": msg_type,
                    "content": str(r["content"]),
                    "id": f"m-{int(r['id'])}",
                }
            )
        return {
            "checkpoint": {"thread_id": thread_id, "checkpoint_id": str(last_id)},
            "parent_checkpoint": None,
            "values": {"messages": msgs},
        }


def append_thread_message(user_id: int, thread_id: str, *, role: str, content: str) -> None:
    role = (role or "").strip().lower()
    if role in {"human", "user"}:
        db_role = "user"
    elif role in {"ai", "assistant"}:
        db_role = "assistant"
    else:
        raise ValueError("role must be human/ai (or user/assistant)")

    content = content or ""
    now = _utc_now_iso()
    with connect() as conn:
        conv_id = _get_conversation_id_by_thread_id(conn, user_id, thread_id)
        if conv_id is None:
            raise ValueError("thread not found")
        conn.execute(
            """
            INSERT INTO messages (conversation_id, role, content, created_at)
            VALUES (?, ?, ?, ?);
            """,
            (conv_id, db_role, content, now),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?;",
            (now, conv_id),
        )


def get_thread_messages_as_lc(user_id: int, thread_id: str):
    from langchain_core.messages import AIMessage, HumanMessage

    with connect() as conn:
        conv_id = _get_conversation_id_by_thread_id(conn, user_id, thread_id)
        if conv_id is None:
            return []
        rows = conn.execute(
            """
            SELECT role, content
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC;
            """,
            (conv_id,),
        ).fetchall()
        out = []
        for r in rows:
            if str(r["role"]) == "assistant":
                out.append(AIMessage(content=str(r["content"])))
            else:
                out.append(HumanMessage(content=str(r["content"])))
        return out


class Storage(Protocol):
    def init_db(self) -> None: ...
    def has_any_users(self) -> bool: ...
    def create_user(self, username: str, password: str, *, is_admin: bool) -> User: ...
    def authenticate(self, username: str, password: str) -> Optional[User]: ...
    def list_users(self) -> List[User]: ...

    def get_user_memory(self, user_id: int) -> str: ...
    def set_user_memory(self, user_id: int, memory: str) -> None: ...

    def list_conversations(self, user_id: int) -> List[Conversation]: ...
    def create_conversation(self, user_id: int, title: str) -> int: ...
    def delete_conversation(self, user_id: int, conversation_id: int) -> None: ...
    def list_messages(self, user_id: int, conversation_id: int) -> List[StoredMessage]: ...
    def add_message(self, user_id: int, conversation_id: int, *, role: str, content: str) -> None: ...

    def get_kb_delete_policy(self) -> KBDeletePolicy: ...
    def get_kb_reindex_policy(self) -> KBReindexPolicy: ...
    def set_setting(self, user_id: int, key: str, value: str) -> None: ...
    def log_audit(self, user_id: Optional[int], action: str, target: str, details: object | None = None) -> None: ...
    def list_audit_events(self, limit: int = 200) -> List[AuditEvent]: ...

    def upsert_kb_file(self, path: str, *, uploader_user_id: Optional[int], size_bytes: int) -> None: ...
    def get_kb_file_meta(self, path: str) -> Optional[KBFileMeta]: ...
    def can_delete_kb_file(self, user: User, meta: Optional[KBFileMeta]) -> bool: ...
    def can_reindex_kb(self, user: User) -> bool: ...

    def create_api_token(self, user_id: int, name: str = "") -> str: ...
    def authenticate_api_token(self, token: str) -> Optional[User]: ...

    def list_threads(self, user_id: int, *, limit: int, offset: int) -> List[Dict[str, Any]]: ...
    def create_thread(self, user_id: int, *, thread_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]: ...
    def get_thread(self, user_id: int, thread_id: str) -> Optional[Dict[str, Any]]: ...
    def get_thread_state(self, user_id: int, thread_id: str) -> Optional[Dict[str, Any]]: ...
    def append_thread_message(self, user_id: int, thread_id: str, *, role: str, content: str) -> None: ...
    def get_thread_messages_as_lc(self, user_id: int, thread_id: str): ...


class SQLiteStorage:
    def init_db(self) -> None:
        init_db()

    def has_any_users(self) -> bool:
        return has_any_users()

    def create_user(self, username: str, password: str, *, is_admin: bool) -> User:
        return create_user(username, password, is_admin=is_admin)

    def authenticate(self, username: str, password: str) -> Optional[User]:
        return authenticate(username, password)

    def list_users(self) -> List[User]:
        return list_users()

    def get_user_memory(self, user_id: int) -> str:
        return get_user_memory(user_id)

    def set_user_memory(self, user_id: int, memory: str) -> None:
        set_user_memory(user_id, memory)

    def list_conversations(self, user_id: int) -> List[Conversation]:
        return list_conversations(user_id)

    def create_conversation(self, user_id: int, title: str) -> int:
        return create_conversation(user_id, title)

    def delete_conversation(self, user_id: int, conversation_id: int) -> None:
        delete_conversation(user_id, conversation_id)

    def list_messages(self, user_id: int, conversation_id: int) -> List[StoredMessage]:
        return list_messages(user_id, conversation_id)

    def add_message(self, user_id: int, conversation_id: int, *, role: str, content: str) -> None:
        add_message(user_id, conversation_id, role=role, content=content)

    def get_kb_delete_policy(self) -> KBDeletePolicy:
        return get_kb_delete_policy()

    def get_kb_reindex_policy(self) -> KBReindexPolicy:
        return get_kb_reindex_policy()

    def set_setting(self, user_id: int, key: str, value: str) -> None:
        set_setting(user_id, key, value)

    def log_audit(
        self, user_id: Optional[int], action: str, target: str, details: object | None = None
    ) -> None:
        log_audit(user_id, action, target, details)

    def list_audit_events(self, limit: int = 200) -> List[AuditEvent]:
        return list_audit_events(limit)

    def upsert_kb_file(self, path: str, *, uploader_user_id: Optional[int], size_bytes: int) -> None:
        upsert_kb_file(path, uploader_user_id=uploader_user_id, size_bytes=size_bytes)

    def get_kb_file_meta(self, path: str) -> Optional[KBFileMeta]:
        return get_kb_file_meta(path)

    def can_delete_kb_file(self, user: User, meta: Optional[KBFileMeta]) -> bool:
        return can_delete_kb_file(user, meta)

    def can_reindex_kb(self, user: User) -> bool:
        return can_reindex_kb(user)

    def create_api_token(self, user_id: int, name: str = "") -> str:
        return create_api_token(user_id, name=name)

    def authenticate_api_token(self, token: str) -> Optional[User]:
        return authenticate_api_token(token)

    def list_threads(self, user_id: int, *, limit: int, offset: int) -> List[Dict[str, Any]]:
        return list_threads(user_id, limit=limit, offset=offset)

    def create_thread(self, user_id: int, *, thread_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        return create_thread(user_id, thread_id=thread_id, metadata=metadata)

    def get_thread(self, user_id: int, thread_id: str) -> Optional[Dict[str, Any]]:
        return get_thread(user_id, thread_id)

    def get_thread_state(self, user_id: int, thread_id: str) -> Optional[Dict[str, Any]]:
        return get_thread_state(user_id, thread_id)

    def append_thread_message(self, user_id: int, thread_id: str, *, role: str, content: str) -> None:
        append_thread_message(user_id, thread_id, role=role, content=content)

    def get_thread_messages_as_lc(self, user_id: int, thread_id: str):
        return get_thread_messages_as_lc(user_id, thread_id)


_STORAGE: SQLiteStorage | None = None


def storage() -> SQLiteStorage:
    global _STORAGE
    if _STORAGE is None:
        backend = os.getenv("APP_STORAGE_BACKEND", "sqlite").lower()
        if backend != "sqlite":
            raise RuntimeError(
                f"Unsupported APP_STORAGE_BACKEND={backend!r}. Only 'sqlite' is implemented."
            )
        _STORAGE = SQLiteStorage()
    return _STORAGE
