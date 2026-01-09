from __future__ import annotations

from typing import Optional

from sqlalchemy import Column, ForeignKey, Integer, LargeBinary, String, Text
from sqlmodel import Field, SQLModel


def utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Department(SQLModel, table=True):
    __tablename__ = "departments"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(
        sa_column=Column(String(100), unique=True, index=True, nullable=False)
    )
    created_at: str = Field(default_factory=utc_now_iso, nullable=False)


class User(SQLModel, table=True):
    __tablename__ = "users"

    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(
        sa_column=Column(String(150), unique=True, index=True, nullable=False)
    )
    password_salt: bytes = Field(sa_column=Column(LargeBinary, nullable=False))
    password_hash: bytes = Field(sa_column=Column(LargeBinary, nullable=False))
    is_admin: bool = Field(default=False, nullable=False)
    department_id: int = Field(
        sa_column=Column(
            Integer, ForeignKey("departments.id", ondelete="RESTRICT"), nullable=False
        )
    )
    created_at: str = Field(default_factory=utc_now_iso, nullable=False)


class UserMemory(SQLModel, table=True):
    __tablename__ = "user_memory"

    user_id: int = Field(
        sa_column=Column(
            Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
        )
    )
    memory: str = Field(default="", sa_column=Column(Text, nullable=False))
    updated_at: str = Field(default_factory=utc_now_iso, nullable=False)


class Conversation(SQLModel, table=True):
    __tablename__ = "conversations"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(
        sa_column=Column(
            Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
        )
    )
    title: str = Field(nullable=False)
    thread_id: Optional[str] = Field(default=None, index=True, unique=True)
    thread_metadata: str = Field(default="", sa_column=Column(Text, nullable=False))
    created_at: str = Field(default_factory=utc_now_iso, nullable=False)
    updated_at: str = Field(default_factory=utc_now_iso, nullable=False)


class Message(SQLModel, table=True):
    __tablename__ = "messages"

    id: Optional[int] = Field(default=None, primary_key=True)
    conversation_id: int = Field(
        sa_column=Column(
            Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
        )
    )
    role: str = Field(nullable=False)
    content: str = Field(sa_column=Column(Text, nullable=False))
    created_at: str = Field(default_factory=utc_now_iso, nullable=False)


class KBFile(SQLModel, table=True):
    __tablename__ = "kb_files"

    path: str = Field(primary_key=True)
    uploader_user_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("users.id", ondelete="SET NULL")),
    )
    size_bytes: int = Field(default=0, nullable=False)
    uploaded_at: str = Field(default_factory=utc_now_iso, nullable=False)


class AuditLog(SQLModel, table=True):
    __tablename__ = "audit_log"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("users.id", ondelete="SET NULL")),
    )
    action: str = Field(nullable=False)
    target: str = Field(nullable=False)
    details: str = Field(default="", sa_column=Column(Text, nullable=False))
    created_at: str = Field(default_factory=utc_now_iso, nullable=False)


class Setting(SQLModel, table=True):
    __tablename__ = "settings"

    key: str = Field(primary_key=True)
    value: str = Field(sa_column=Column(Text, nullable=False))
    updated_at: str = Field(default_factory=utc_now_iso, nullable=False)
    updated_by_user_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("users.id", ondelete="SET NULL")),
    )


class ApiToken(SQLModel, table=True):
    __tablename__ = "api_tokens"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(
        sa_column=Column(
            Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
        )
    )
    token_hash: bytes = Field(sa_column=Column(LargeBinary, unique=True, nullable=False))
    token_prefix: str = Field(nullable=False)
    name: str = Field(default="", nullable=False)
    created_at: str = Field(default_factory=utc_now_iso, nullable=False)
    revoked_at: Optional[str] = Field(default=None)
    last_used_at: Optional[str] = Field(default=None)
