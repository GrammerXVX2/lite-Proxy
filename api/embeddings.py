from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request

from constants import EMBEDDING_PATH_CANDIDATES, ERR_NO_EMBEDDING_MODELS
from schemas import EmbedResponseModel
from services.model_catalog import resolve_target
from services.request_parser import read_request_body_as_dict
from services.upstream import post_json_to

from api.common import ns

router = APIRouter(tags=["embeddings"])


async def _post_embeddings_with_fallback(base_url: str, model_id: str, input_data: Any) -> Dict[str, Any]:
    """
    Параметры:
    - base_url: базовый URL upstream-сервиса embeddings.
    - model_id: backend-модель для upstream.
    - input_data: вход для embedding.

    Что делает:
    - Последовательно пробует несколько embedding-endpoint'ов (OpenAI/TEI варианты).
    - Возвращает первый успешный ответ или поднимает ошибку.

    Выходные данные:
    - JSON-ответ upstream embeddings endpoint.
    """
    attempts = [
        (EMBEDDING_PATH_CANDIDATES[0], {"model": model_id, "input": input_data}),
        (EMBEDDING_PATH_CANDIDATES[1], {"model": model_id, "input": input_data}),
        (EMBEDDING_PATH_CANDIDATES[2], {"inputs": input_data}),
        (EMBEDDING_PATH_CANDIDATES[3], {"inputs": input_data}),
    ]

    last_exc: HTTPException | None = None
    for path, payload in attempts:
        try:
            return await post_json_to(base_url, path, payload)
        except HTTPException as exc:
            last_exc = exc
            if exc.status_code in (404, 405, 422):
                continue
            raise

    if last_exc is not None:
        raise last_exc
    raise HTTPException(status_code=502, detail="embedding upstream is unavailable")


def _extract_embeddings(data: Any) -> List[List[float]]:
    """
    Параметры:
    - data: ответ upstream embeddings endpoint.

    Что делает:
    - Нормализует разные форматы ответа (OpenAI/TEI/single-vector) в единый список векторов.

    Выходные данные:
    - Список embedding-векторов.
    """
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        return [item.get("embedding", []) for item in data.get("data", []) if isinstance(item, dict)]

    if isinstance(data, dict) and isinstance(data.get("embeddings"), list):
        emb = data.get("embeddings")
        if emb and isinstance(emb[0], list):
            return emb

    if isinstance(data, dict) and isinstance(data.get("embedding"), list):
        emb = data.get("embedding")
        if emb and isinstance(emb[0], (int, float)):
            return [emb]

    if isinstance(data, list) and data and isinstance(data[0], list):
        return data
    if isinstance(data, list) and data and isinstance(data[0], (int, float)):
        return [data]

    return []


@router.post("/api/embed", response_model=EmbedResponseModel, summary="Embed text")
async def api_embed(request: Request) -> Dict[str, Any]:
    """
    Параметры:
    - request: входящий HTTP-запрос.

    Что делает:
    - Разбирает входное тело запроса,
    - выбирает embeddings-модель из каталога,
    - отправляет запрос в upstream и возвращает Ollama-compatible embedding-ответ.

    Выходные данные:
    - JSON-ответ в формате `EmbedResponseModel`.
    """
    body_data = await read_request_body_as_dict(request)

    requested_model = body_data.get("model")
    try:
        target = resolve_target(requested_model, expected_types={"embeddings"})
    except HTTPException as exc:
        if exc.status_code == 503:
            raise HTTPException(status_code=503, detail=ERR_NO_EMBEDDING_MODELS)
        raise

    model = requested_model or target["model"]
    input_data = body_data.get("input")

    if input_data is None:
        input_data = body_data.get("prompt") or body_data.get("text")

    if input_data is None and isinstance(body_data.get("message"), dict):
        input_data = body_data.get("message", {}).get("content")

    if input_data is None and isinstance(body_data.get("messages"), list):
        merged = []
        for message in body_data.get("messages"):
            if isinstance(message, dict) and message.get("content") is not None:
                merged.append(str(message.get("content")))
        if merged:
            input_data = "\n".join(merged)

    if input_data is None:
        raise HTTPException(status_code=400, detail="input is required")

    start_ns = ns()
    data = await _post_embeddings_with_fallback(target["base_url"], target["model_vllm"], input_data)

    embeddings = _extract_embeddings(data)
    usage = data.get("usage") if isinstance(data, dict) else {}

    return {
        "model": model,
        "embedding": embeddings[0] if embeddings else [],
        "embeddings": embeddings,
        "total_duration": max(0, ns() - start_ns),
        "load_duration": 0,
        "prompt_eval_count": usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0,
    }
