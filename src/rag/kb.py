from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable, List


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"


@dataclass(frozen=True)
class KBFile:
    name: str
    path: Path
    size_bytes: int


def list_kb_files() -> List[KBFile]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    files: List[KBFile] = []
    for path in sorted(DATA_DIR.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(DATA_DIR)
        files.append(KBFile(name=str(rel), path=path, size_bytes=path.stat().st_size))
    return files


def save_upload(filename: str, content: bytes) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename).name
    target = DATA_DIR / safe_name
    target.write_bytes(content)
    return target


def delete_kb_file(name: str) -> None:
    target = (DATA_DIR / name).resolve()
    if DATA_DIR.resolve() not in target.parents:
        raise ValueError("Invalid file path")
    if target.exists() and target.is_file():
        target.unlink()


def clear_vectorstore() -> None:
    try:
        from .providers import weaviate_client_from_env
    except ImportError:
        from providers import weaviate_client_from_env

    client = weaviate_client_from_env()
    class_name = os.getenv("WEAVIATE_CLASS", "RAGChunk")
    if client.schema.exists(class_name):
        client.schema.delete_class(class_name)
