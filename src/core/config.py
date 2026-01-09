from __future__ import annotations

from dotenv import load_dotenv


def load_env() -> None:
    # 统一加载 .env
    load_dotenv()
