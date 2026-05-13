"""Shared pytest fixtures and config."""
import os
import pytest
import httpx

BASE_URL = os.getenv("PROXY_URL", "http://localhost:11434")
TIMEOUT = float(os.getenv("PROXY_TIMEOUT", "120"))

CHAT_MODEL   = os.getenv("CHAT_MODEL",  "qwen35-122b-a10b-fp8")
EMBED_MODEL  = os.getenv("EMBED_MODEL", "qwen3-embedding-4b")
RERANK_MODEL = os.getenv("RERANK_MODEL","qwen3-reranker-4b")


@pytest.fixture(scope="session")
def client() -> httpx.Client:
    with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as c:
        yield c
