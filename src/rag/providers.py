from __future__ import annotations

import os

import weaviate
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def embeddings_from_env():
    provider = os.getenv("EMBED_PROVIDER", "dashscope").lower()
    if provider == "openai":
        model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
        return OpenAIEmbeddings(model=model)

    # DashScope native embeddings (Tongyi/Qwen)
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("TONGYI_API_KEY")
    model = os.getenv("DASHSCOPE_EMBED_MODEL", "text-embedding-v2")
    return DashScopeEmbeddings(dashscope_api_key=api_key, model=model)


def llm_from_env():
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


def weaviate_client_from_env():
    url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    api_key = os.getenv("WEAVIATE_API_KEY")
    if api_key:
        auth = weaviate.AuthApiKey(api_key=api_key)
        return weaviate.Client(url=url, auth_client_secret=auth)
    return weaviate.Client(url=url)
