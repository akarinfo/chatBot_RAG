from __future__ import annotations

from typing import List, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class RAGState(TypedDict):
    messages: List[BaseMessage]
    context: List[Document]
    memory: str
