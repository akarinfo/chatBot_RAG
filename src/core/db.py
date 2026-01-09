from __future__ import annotations

import hashlib
import json
import os
import secrets
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Literal

from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import QueuePool
from sqlmodel import SQLModel, Session, create_engine, select

from core.db_models import (
    ApiToken as ApiTokenModel,
    AuditLog as AuditLogModel,
    Conversation as ConversationModel,
    Department as DepartmentModel,
    KBFile as KBFileModel,
    Message as MessageModel,
    Setting as SettingModel,
    User as UserModel,
    UserMemory as UserMemoryModel,
    utc_now_iso,
)


DEFAULT_DEPARTMENT = os.getenv("APP_DEFAULT_DEPARTMENT", "default").strip() or "default"


def _database_url() -> str:
    return (
        os.getenv("DATABASE_URL")
        or os.getenv("APP_DATABASE_URL")
        or "postgresql+psycopg2://postgres:postgres@localhost:5432/chatbot_rag"
    )


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError:
        return default


_ENGINE = None


def _engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = create_engine(
            _database_url(),
            echo=os.getenv("APP_DB_ECHO", "0").lower() in {"1", "true", "yes"},
            poolclass=QueuePool,
            pool_size=_env_int("APP_DB_POOL_SIZE", 5),
            max_overflow=_env_int("APP_DB_MAX_OVERFLOW", 10),
            pool_recycle=_env_int("APP_DB_POOL_RECYCLE", 1800),
            pool_pre_ping=True,
        )
    return _ENGINE


def _ensure_default_department() -> None:
    with Session(_engine()) as session:
        dept = session.exec(
            select(DepartmentModel).where(DepartmentModel.name == DEFAULT_DEPARTMENT)
        ).first()
        if dept is None:
            session.add(
                DepartmentModel(name=DEFAULT_DEPARTMENT, created_at=utc_now_iso())
            )
            session.commit()


def init_db() -> None:
    auto_create = os.getenv("APP_AUTO_CREATE_SCHEMA", "0").lower() in {"1", "true", "yes"}
    if auto_create:
        SQLModel.metadata.create_all(_engine())
        _ensure_default_department()
        return

    with _engine().connect() as conn:
        inspector = inspect(conn)
        tables = set(inspector.get_table_names())
        if "alembic_version" not in tables:
            raise RuntimeError(
                "Database schema not initialized. Run `alembic upgrade head` first "
                "or set APP_AUTO_CREATE_SCHEMA=1 for auto-create."
            )
    _ensure_default_department()


@dataclass(frozen=True)
class Department:
    id: int
    name: str
    created_at: str


@dataclass(frozen=True)
class User:
    id: int
    username: str
    is_admin: bool
    department_id: int
    department_name: str


def _hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)


def has_any_users() -> bool:
    with Session(_engine()) as session:
        row = session.exec(select(UserModel.id).limit(1)).first()
        return row is not None


def _get_or_create_department(session: Session, name: Optional[str]) -> DepartmentModel:
    dept_name = (name or DEFAULT_DEPARTMENT).strip() or DEFAULT_DEPARTMENT
    dept = session.exec(
        select(DepartmentModel).where(DepartmentModel.name == dept_name)
    ).first()
    if dept is None:
        dept = DepartmentModel(name=dept_name, created_at=utc_now_iso())
        session.add(dept)
        session.commit()
        session.refresh(dept)
    return dept


def list_departments() -> List[Department]:
    with Session(_engine()) as session:
        rows = session.exec(select(DepartmentModel).order_by(DepartmentModel.name)).all()
        return [
            Department(id=int(r.id), name=str(r.name), created_at=str(r.created_at))
            for r in rows
        ]


def _department_name(session: Session, department_id: int) -> str:
    dept = session.get(DepartmentModel, int(department_id))
    return str(dept.name) if dept else DEFAULT_DEPARTMENT


def create_user(
    username: str,
    password: str,
    *,
    is_admin: bool,
    department_name: Optional[str] = None,
) -> User:
    username = username.strip()
    if not username:
        raise ValueError("username is required")
    if len(password) < 8:
        raise ValueError("password must be at least 8 characters")

    salt = secrets.token_bytes(16)
    pwd_hash = _hash_password(password, salt)
    created_at = utc_now_iso()

    with Session(_engine()) as session:
        dept = _get_or_create_department(session, department_name)
        user = UserModel(
            username=username,
            password_salt=salt,
            password_hash=pwd_hash,
            is_admin=bool(is_admin),
            department_id=dept.id,
            created_at=created_at,
        )
        session.add(user)
        try:
            session.flush()
        except IntegrityError as exc:
            session.rollback()
            raise ValueError("username already exists") from exc

        session.add(
            UserMemoryModel(
                user_id=int(user.id),
                memory="",
                updated_at=created_at,
            )
        )
        session.commit()
        session.refresh(user)

        return User(
            id=int(user.id),
            username=str(user.username),
            is_admin=bool(user.is_admin),
            department_id=int(user.department_id),
            department_name=str(dept.name),
        )


def authenticate(username: str, password: str) -> Optional[User]:
    username = username.strip()
    if not username or not password:
        return None
    with Session(_engine()) as session:
        user = session.exec(
            select(UserModel).where(UserModel.username == username)
        ).first()
        if user is None:
            return None
        salt = bytes(user.password_salt)
        expected = bytes(user.password_hash)
        got = _hash_password(password, salt)
        if not secrets.compare_digest(expected, got):
            return None
        dept_name = _department_name(session, int(user.department_id))
        return User(
            id=int(user.id),
            username=str(user.username),
            is_admin=bool(user.is_admin),
            department_id=int(user.department_id),
            department_name=dept_name,
        )


def _hash_token(token: str) -> bytes:
    return hashlib.sha256(token.encode("utf-8")).digest()


def create_api_token(user_id: int, name: str = "") -> str:
    token = secrets.token_urlsafe(32)
    token_hash = _hash_token(token)
    prefix = token[:8]
    now = utc_now_iso()
    with Session(_engine()) as session:
        session.add(
            ApiTokenModel(
                user_id=int(user_id),
                token_hash=token_hash,
                token_prefix=prefix,
                name=name or "",
                created_at=now,
            )
        )
        session.commit()
    return token


def authenticate_api_token(token: str) -> Optional[User]:
    token = (token or "").strip()
    if not token:
        return None
    token_hash = _hash_token(token)
    now = utc_now_iso()
    with Session(_engine()) as session:
        row = session.exec(
            select(ApiTokenModel, UserModel)
            .join(UserModel, UserModel.id == ApiTokenModel.user_id)
            .where(ApiTokenModel.token_hash == token_hash)
            .where(ApiTokenModel.revoked_at.is_(None))
        ).first()
        if row is None:
            return None
        api_token, user = row
        api_token.last_used_at = now
        session.add(api_token)
        session.commit()
        dept_name = _department_name(session, int(user.department_id))
        return User(
            id=int(user.id),
            username=str(user.username),
            is_admin=bool(user.is_admin),
            department_id=int(user.department_id),
            department_name=dept_name,
        )


def list_users() -> List[User]:
    with Session(_engine()) as session:
        rows = session.exec(select(UserModel).order_by(UserModel.username)).all()
        out: List[User] = []
        for u in rows:
            dept_name = _department_name(session, int(u.department_id))
            out.append(
                User(
                    id=int(u.id),
                    username=str(u.username),
                    is_admin=bool(u.is_admin),
                    department_id=int(u.department_id),
                    department_name=dept_name,
                )
            )
        return out


def get_user_memory(user_id: int) -> str:
    with Session(_engine()) as session:
        row = session.get(UserMemoryModel, int(user_id))
        return str(row.memory) if row else ""


def set_user_memory(user_id: int, memory: str) -> None:
    with Session(_engine()) as session:
        row = session.get(UserMemoryModel, int(user_id))
        now = utc_now_iso()
        if row is None:
            row = UserMemoryModel(
                user_id=int(user_id), memory=memory or "", updated_at=now
            )
            session.add(row)
        else:
            row.memory = memory or ""
            row.updated_at = now
            session.add(row)
        session.commit()


@dataclass(frozen=True)
class Conversation:
    id: int
    title: str
    updated_at: str


def list_conversations(user_id: int) -> List[Conversation]:
    with Session(_engine()) as session:
        rows = session.exec(
            select(ConversationModel)
            .where(ConversationModel.user_id == int(user_id))
            .order_by(ConversationModel.updated_at.desc())
        ).all()
        return [
            Conversation(id=int(r.id), title=str(r.title), updated_at=str(r.updated_at))
            for r in rows
        ]


def create_conversation(user_id: int, title: str) -> int:
    title = (title or "").strip() or "新对话"
    now = utc_now_iso()
    thread_id = str(uuid.uuid4())
    meta = json.dumps({"graph_id": "agent"}, ensure_ascii=False)
    with Session(_engine()) as session:
        conv = ConversationModel(
            user_id=int(user_id),
            title=title,
            thread_id=thread_id,
            thread_metadata=meta,
            created_at=now,
            updated_at=now,
        )
        session.add(conv)
        session.commit()
        session.refresh(conv)
        return int(conv.id)


def delete_conversation(user_id: int, conversation_id: int) -> None:
    with Session(_engine()) as session:
        conv = session.exec(
            select(ConversationModel)
            .where(ConversationModel.id == int(conversation_id))
            .where(ConversationModel.user_id == int(user_id))
        ).first()
        if conv is None:
            return
        session.delete(conv)
        session.commit()


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
    with Session(_engine()) as session:
        row = session.get(SettingModel, key)
        return str(row.value) if row else None


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
    now = utc_now_iso()
    with Session(_engine()) as session:
        row = session.get(SettingModel, key)
        if row is None:
            row = SettingModel(
                key=key,
                value=value,
                updated_at=now,
                updated_by_user_id=int(user_id),
            )
            session.add(row)
        else:
            row.value = value
            row.updated_at = now
            row.updated_by_user_id = int(user_id)
            session.add(row)
        session.commit()


def log_audit(
    user_id: Optional[int], action: str, target: str, details: object | None = None
) -> None:
    with Session(_engine()) as session:
        session.add(
            AuditLogModel(
                user_id=int(user_id) if user_id is not None else None,
                action=action,
                target=target,
                details=_audit_details(details),
                created_at=utc_now_iso(),
            )
        )
        session.commit()


def list_audit_events(limit: int = 200) -> List[AuditEvent]:
    limit = max(1, min(int(limit), 5000))
    with Session(_engine()) as session:
        rows = session.exec(
            select(AuditLogModel)
            .order_by(AuditLogModel.id.desc())
            .limit(limit)
        ).all()
        return [
            AuditEvent(
                id=int(r.id),
                user_id=int(r.user_id) if r.user_id is not None else None,
                action=str(r.action),
                target=str(r.target),
                details=str(r.details or ""),
                created_at=str(r.created_at),
            )
            for r in rows
        ]


def upsert_kb_file(path: str, *, uploader_user_id: Optional[int], size_bytes: int) -> None:
    now = utc_now_iso()
    with Session(_engine()) as session:
        row = session.get(KBFileModel, path)
        if row is None:
            row = KBFileModel(
                path=path,
                uploader_user_id=int(uploader_user_id)
                if uploader_user_id is not None
                else None,
                size_bytes=int(size_bytes),
                uploaded_at=now,
            )
            session.add(row)
        else:
            row.uploader_user_id = (
                int(uploader_user_id) if uploader_user_id is not None else None
            )
            row.size_bytes = int(size_bytes)
            row.uploaded_at = now
            session.add(row)
        session.commit()


def get_kb_file_meta(path: str) -> Optional[KBFileMeta]:
    with Session(_engine()) as session:
        row = session.get(KBFileModel, path)
        if not row:
            return None
        return KBFileMeta(
            path=str(row.path),
            uploader_user_id=int(row.uploader_user_id)
            if row.uploader_user_id is not None
            else None,
            size_bytes=int(row.size_bytes or 0),
            uploaded_at=str(row.uploaded_at),
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
    with Session(_engine()) as session:
        owner = session.exec(
            select(ConversationModel.id)
            .where(ConversationModel.id == int(conversation_id))
            .where(ConversationModel.user_id == int(user_id))
        ).first()
        if owner is None:
            return []
        rows = session.exec(
            select(MessageModel)
            .where(MessageModel.conversation_id == int(conversation_id))
            .order_by(MessageModel.id)
        ).all()
        return [
            StoredMessage(
                role=str(r.role),
                content=str(r.content),
                created_at=str(r.created_at),
            )
            for r in rows
        ]


def add_message(user_id: int, conversation_id: int, *, role: str, content: str) -> None:
    if role not in {"user", "assistant"}:
        raise ValueError("role must be 'user' or 'assistant'")
    now = utc_now_iso()
    with Session(_engine()) as session:
        owner = session.exec(
            select(ConversationModel)
            .where(ConversationModel.id == int(conversation_id))
            .where(ConversationModel.user_id == int(user_id))
        ).first()
        if owner is None:
            raise ValueError("conversation not found")
        session.add(
            MessageModel(
                conversation_id=int(conversation_id),
                role=role,
                content=content,
                created_at=now,
            )
        )
        owner.updated_at = now
        session.add(owner)
        session.commit()


def _get_conversation_id_by_thread_id(
    session: Session, user_id: int, thread_id: str
) -> Optional[int]:
    row = session.exec(
        select(ConversationModel.id)
        .where(ConversationModel.user_id == int(user_id))
        .where(ConversationModel.thread_id == thread_id)
    ).first()
    return int(row) if row is not None else None


def _load_thread_row(
    session: Session, user_id: int, thread_id: str
) -> Optional[ConversationModel]:
    return session.exec(
        select(ConversationModel)
        .where(ConversationModel.user_id == int(user_id))
        .where(ConversationModel.thread_id == thread_id)
    ).first()


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
    with Session(_engine()) as session:
        rows = session.exec(
            select(ConversationModel)
            .where(ConversationModel.user_id == int(user_id))
            .where(ConversationModel.thread_id.is_not(None))
            .where(ConversationModel.thread_id != "")
            .order_by(ConversationModel.updated_at.desc())
            .limit(limit)
            .offset(offset)
        ).all()
        out: List[Dict[str, Any]] = []
        for r in rows:
            conv_id = int(r.id)
            first = session.exec(
                select(MessageModel)
                .where(MessageModel.conversation_id == conv_id)
                .order_by(MessageModel.id)
                .limit(1)
            ).first()
            messages: List[Dict[str, Any]] = []
            if first:
                msg_type = "human" if str(first.role) == "user" else "ai"
                messages.append(
                    {
                        "type": msg_type,
                        "content": str(first.content),
                        "id": f"m-{int(first.id)}",
                    }
                )
            out.append(
                {
                    "thread_id": str(r.thread_id),
                    "metadata": _parse_thread_metadata(str(r.thread_metadata)),
                    "values": {"messages": messages},
                    "created_at": str(r.created_at),
                    "updated_at": str(r.updated_at),
                }
            )
        return out


def create_thread(
    user_id: int, *, thread_id: str, metadata: Dict[str, Any]
) -> Dict[str, Any]:
    thread_id = (thread_id or "").strip()
    if not thread_id:
        raise ValueError("thread_id is required")
    now = utc_now_iso()
    meta = metadata or {}
    meta.setdefault("graph_id", meta.get("graph_id") or "agent")
    meta_raw = json.dumps(meta, ensure_ascii=False)
    title = str(meta.get("title") or "新对话")
    with Session(_engine()) as session:
        session.add(
            ConversationModel(
                user_id=int(user_id),
                title=title,
                thread_id=thread_id,
                thread_metadata=meta_raw,
                created_at=now,
                updated_at=now,
            )
        )
        session.commit()
    return {
        "thread_id": thread_id,
        "metadata": meta,
        "values": {"messages": []},
        "created_at": now,
        "updated_at": now,
    }


def get_thread(user_id: int, thread_id: str) -> Optional[Dict[str, Any]]:
    with Session(_engine()) as session:
        row = _load_thread_row(session, user_id, thread_id)
        if row is None:
            return None
        return {
            "thread_id": str(row.thread_id),
            "metadata": _parse_thread_metadata(str(row.thread_metadata)),
            "values": {"messages": []},
            "created_at": str(row.created_at),
            "updated_at": str(row.updated_at),
            "title": str(row.title),
        }


def get_thread_state(user_id: int, thread_id: str) -> Optional[Dict[str, Any]]:
    with Session(_engine()) as session:
        conv = _load_thread_row(session, user_id, thread_id)
        if conv is None:
            return None
        conv_id = int(conv.id)
        rows = session.exec(
            select(MessageModel)
            .where(MessageModel.conversation_id == conv_id)
            .order_by(MessageModel.id)
        ).all()
        msgs: List[Dict[str, Any]] = []
        last_id = 0
        for r in rows:
            last_id = int(r.id)
            msg_type = "human" if str(r.role) == "user" else "ai"
            msgs.append(
                {
                    "type": msg_type,
                    "content": str(r.content),
                    "id": f"m-{int(r.id)}",
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
    now = utc_now_iso()
    with Session(_engine()) as session:
        conv_id = _get_conversation_id_by_thread_id(session, user_id, thread_id)
        if conv_id is None:
            raise ValueError("thread not found")
        session.add(
            MessageModel(
                conversation_id=int(conv_id),
                role=db_role,
                content=content,
                created_at=now,
            )
        )
        conv = session.get(ConversationModel, int(conv_id))
        if conv is not None:
            conv.updated_at = now
            session.add(conv)
        session.commit()


def get_thread_messages_as_lc(user_id: int, thread_id: str):
    from langchain_core.messages import AIMessage, HumanMessage

    with Session(_engine()) as session:
        conv_id = _get_conversation_id_by_thread_id(session, user_id, thread_id)
        if conv_id is None:
            return []
        rows = session.exec(
            select(MessageModel)
            .where(MessageModel.conversation_id == int(conv_id))
            .order_by(MessageModel.id)
        ).all()
        out = []
        for r in rows:
            if str(r.role) == "assistant":
                out.append(AIMessage(content=str(r.content)))
            else:
                out.append(HumanMessage(content=str(r.content)))
        return out


class Storage(Protocol):
    def init_db(self) -> None: ...
    def has_any_users(self) -> bool: ...
    def create_user(
        self,
        username: str,
        password: str,
        *,
        is_admin: bool,
        department_name: Optional[str] = None,
    ) -> User: ...
    def authenticate(self, username: str, password: str) -> Optional[User]: ...
    def list_users(self) -> List[User]: ...
    def list_departments(self) -> List[Department]: ...

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


class SQLStorage:
    def init_db(self) -> None:
        init_db()

    def has_any_users(self) -> bool:
        return has_any_users()

    def create_user(
        self,
        username: str,
        password: str,
        *,
        is_admin: bool,
        department_name: Optional[str] = None,
    ) -> User:
        return create_user(
            username, password, is_admin=is_admin, department_name=department_name
        )

    def authenticate(self, username: str, password: str) -> Optional[User]:
        return authenticate(username, password)

    def list_users(self) -> List[User]:
        return list_users()

    def list_departments(self) -> List[Department]:
        return list_departments()

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


_STORAGE: SQLStorage | None = None


def storage() -> SQLStorage:
    global _STORAGE
    if _STORAGE is None:
        backend = os.getenv("APP_STORAGE_BACKEND", "postgres").lower()
        if backend != "postgres":
            raise RuntimeError(
                f"Unsupported APP_STORAGE_BACKEND={backend!r}. Only 'postgres' is implemented."
            )
        _STORAGE = SQLStorage()
    return _STORAGE
