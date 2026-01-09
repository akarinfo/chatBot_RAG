from __future__ import annotations

import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def embeddings_from_env():
    # Embedding 提供方：ModelScope（默认）或 OpenAI
    provider = os.getenv("EMBED_PROVIDER", "modelscope").lower()
    if provider == "openai":
        model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
        return OpenAIEmbeddings(model=model)

    # ModelScope embeddings via OpenAI-compatible endpoint
    api_key = os.getenv("MODELSCOPE_API_TOKEN")
    base_url = os.getenv("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1")
    model = os.getenv("MODELSCOPE_EMBED_MODEL")
    if not api_key or not model:
        raise ValueError(
            "MODELSCOPE_API_TOKEN and MODELSCOPE_EMBED_MODEL are required"
        )
    return OpenAIEmbeddings(api_key=api_key, base_url=base_url, model=model)


def llm_from_env():
    # LLM 提供方：DeepSeek（默认）或 OpenAI
    provider = os.getenv("LLM_PROVIDER", "deepseek").lower()
    if provider == "openai":
        model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
        return ChatOpenAI(model=model, temperature=temperature)

    # DeepSeek provides an OpenAI-compatible endpoint.
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    temperature = float(os.getenv("DEEPSEEK_TEMPERATURE", "0.2"))
    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
    )

