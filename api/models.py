from typing import Any

import asyncio
from fastapi import APIRouter

from schemas import ModelsResponse, TagsResponse
from services.limiter import limiter_status
from services.model_catalog import get_models_snapshot, to_ollama_tag_item
from services.upstream import ping_model

router = APIRouter(tags=["models"])


@router.get("/", tags=["chat"], summary="Lite Proxy Status")
async def root_status() -> dict[str, str]:
    """
    Параметры:
    - отсутствуют.

    Что делает:
    - Возвращает служебный статус liteProxy и ссылки на документацию.

    Выходные данные:
    - Словарь со статусом сервиса.
    """
    return {
        "status": "ok",
        "name": "liteProxy",
        "docs": "/docs",
        "openapi": "/openapi.json",
    }


@router.get("/api/models", response_model=ModelsResponse, summary="List configured models with health status")
async def api_models() -> dict[str, list[dict[str, Any]]]:
    """
    Параметры:
    - отсутствуют.

    Что делает:
    - Пингует upstream каждой модели параллельно через GET /v1/models.
    - Возвращает список моделей с полями status (available/error) и detail.

    Выходные данные:
    - Словарь вида `{ "models": [...] }`.
    """
    models: list[dict[str, Any]] = list(get_models_snapshot())  # type: ignore[assignment]

    # Collect unique base_url values and ping them all in parallel
    unique_urls: list[str] = list({str(m.get("base_url", "")) for m in models if m.get("base_url")})
    results = await asyncio.gather(*[ping_model(url) for url in unique_urls], return_exceptions=False)
    ping_map: dict[str, tuple[bool, str]] = dict(zip(unique_urls, results))  # type: ignore[arg-type]

    for model in models:
        base_url = str(model.get("base_url", ""))
        ok, detail = ping_map.get(base_url, (False, "unknown base_url"))
        model["status"] = "available" if ok else "error"
        model["detail"] = detail if not ok else ""

    return {"models": models}


@router.get("/api/tags", response_model=TagsResponse, summary="List models in Ollama tags format")
async def api_tags() -> dict[str, list[dict[str, Any]]]:
    """
    Параметры:
    - отсутствуют.

    Что делает:
    - Преобразует доступные chat/embeddings модели в Ollama tags-формат.

    Выходные данные:
    - Словарь вида `{ "models": [...] }` в tags-формате.
    """
    snapshot = get_models_snapshot()
    tags = [
        to_ollama_tag_item(item)
        for item in snapshot
        if str(item.get("type")) in {"chat", "embeddings"}
    ]
    return {"models": tags}


@router.get("/api/queue/status", summary="Upstream concurrency queue metrics")
async def api_queue_status() -> dict[str, dict[str, int]]:
    """
    Параметры:
    - отсутствуют.

    Что делает:
    - Возвращает текущее состояние очереди запросов к upstream vLLM по каждому типу модели.

    Выходные данные:
    - Словарь вида `{ "chat": {...}, "embeddings": {...}, "reranker": {...} }`.
    """
    return limiter_status()
