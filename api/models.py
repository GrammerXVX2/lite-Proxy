from typing import Any

from fastapi import APIRouter

from schemas import ModelsResponse, TagsResponse
from services.limiter import limiter_status
from services.model_catalog import get_models_snapshot, to_ollama_tag_item

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


@router.get("/api/models", response_model=ModelsResponse, summary="List configured models")
async def api_models() -> dict[str, list[dict[str, Any]]]:
    """
    Параметры:
    - отсутствуют.

    Что делает:
    - Возвращает снимок конфигурации всех моделей из каталога liteProxy.

    Выходные данные:
    - Словарь вида `{ "models": [...] }`.
    """
    models: list[dict[str, Any]] = list(get_models_snapshot())  # type: ignore[assignment]
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
