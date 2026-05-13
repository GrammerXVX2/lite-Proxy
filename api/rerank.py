from typing import Any

from fastapi import APIRouter, HTTPException, Request

from api.common import inline_schema
from constants import ERR_NO_RERANK_MODELS, RERANK_V1_PATH, RERANK_V2_PATH, SCORE_V1_PATH
from schemas import RerankRequestModel, RerankResponse, ScoreRequestModel, ScoreResponse
from services.model_catalog import ModelEntry, resolve_target
from services.request_parser import read_request_body_as_dict
from services.upstream import post_json_to

router = APIRouter(tags=["rerank"])


def _ensure_rerank_temperature(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Параметры:
    - payload: входной rerank/score payload.

    Что делает:
    - Устанавливает дефолт `temperature=0`, если параметр не передан клиентом.

    Выходные данные:
    - Обновлённый payload с гарантированным значением temperature.
    """
    patched = dict(payload)
    if patched.get("temperature") is None:
        patched["temperature"] = 0
    return patched


def _normalize_reranker_base_url(base_url: str) -> str:
    """
    Параметры:
    - base_url: базовый URL модели reranker.

    Что делает:
    - Нормализует base URL для reranker v1/v2/score endpoint'ов.
    - Если URL оканчивается на `/v1`, убирает этот суффикс.

    Выходные данные:
    - Нормализованный base URL.
    """
    normalized = str(base_url or "").rstrip("/")
    if normalized.endswith("/v1"):
        return normalized[:-3]
    return normalized


def _validate_score_payload(payload: dict[str, Any]) -> None:
    """
    Параметры:
    - payload: тело score-запроса.

    Что делает:
    - Проверяет, что payload содержит одну из поддерживаемых пар полей для score.
    - Поднимает HTTP 400 при невалидном формате.

    Выходные данные:
    - отсутствуют.
    """
    has_text_pair = payload.get("text_1") is not None and payload.get("text_2") is not None
    has_queries_docs = payload.get("queries") is not None and payload.get("documents") is not None
    has_queries_items = payload.get("queries") is not None and payload.get("items") is not None
    has_data_pair = payload.get("data_1") is not None and payload.get("data_2") is not None
    if has_text_pair or has_queries_docs or has_queries_items or has_data_pair:
        return
    raise HTTPException(
        status_code=400,
        detail=(
            "score request must include one of: "
            "(text_1,text_2), (queries,documents), (queries,items), or (data_1,data_2)"
        ),
    )


async def _resolve_reranker_target(requested_model: str | None) -> ModelEntry:
    """
    Параметры:
    - requested_model: публичное имя модели из запроса.

    Что делает:
    - Резолвит модель из каталога для rerank/score маршрутов.
    - Преобразует ошибку отсутствия моделей к специализированному сообщению.

    Выходные данные:
    - Конфигурация целевой модели.
    """
    try:
        return resolve_target(requested_model, expected_types={"reranker", "chat"})
    except HTTPException as exc:
        if exc.status_code == 503:
            raise HTTPException(status_code=503, detail=ERR_NO_RERANK_MODELS)
        raise


async def _rerank_impl(request: Request, payload: dict[str, Any], path: str) -> dict[str, Any]:
    """
    Параметры:
    - request: входящий HTTP-запрос.
    - payload: тело rerank-запроса.
    - path: upstream endpoint (`/v1/rerank` или `/v2/rerank`).

    Что делает:
    - Выполняет общую логику rerank-маршрутов:
        резолвит модель, валидирует обязательные поля, вызывает upstream.

    Выходные данные:
    - JSON-ответ reranker upstream endpoint.
    """
    payload = _ensure_rerank_temperature(payload)
    target = await _resolve_reranker_target(payload.get("model"))
    model = str(payload.get("model") or target.get("model"))

    if payload.get("query") is None or payload.get("documents") is None:
        raise HTTPException(status_code=400, detail="rerank request requires query and documents")

    upstream_payload = dict(payload)
    upstream_payload["model"] = target["model_vllm"]
    request.state.model = model
    request.state.upstream = target.get("base_url", "")

    upstream_base = _normalize_reranker_base_url(target["base_url"])
    return await post_json_to(upstream_base, path, upstream_payload, model_type=target["type"])


@router.post(
    "/api/reranker/rerank/v1",
    summary="Rerank (v1)",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "schema": inline_schema(RerankRequestModel.model_json_schema()),
                    "examples": {
                        "Без logprobs": {
                            "summary": "Обычный запрос",
                            "value": {
                                "model": "qwen3-reranker-4b",
                                "query": "Что такое квантовая запутанность?",
                                "documents": [
                                    "Квантовая запутанность — явление, при котором состояния двух частиц взаимозависимы.",
                                    "Фотосинтез — процесс преобразования световой энергии в химическую у растений.",
                                    "Принцип суперпозиции описывает нахождение квантовой системы в нескольких состояниях одновременно.",
                                ],
                                "top_n": 2,
                            },
                        },
                        "С logprobs": {
                            "summary": "С логитами (logprobs)",
                            "value": {
                                "model": "qwen3-reranker-4b",
                                "query": "Что такое квантовая запутанность?",
                                "documents": [
                                    "Квантовая запутанность — явление, при котором состояния двух частиц взаимозависимы.",
                                    "Фотосинтез — процесс преобразования световой энергии в химическую у растений.",
                                ],
                                "top_n": 2,
                                "logprobs": True,
                                "top_logprobs": 2,
                            },
                        },
                    },
                }
            },
            "required": True,
        },
        "responses": {
            "200": {
                "description": "Successful Response",
                "content": {"application/json": {"schema": inline_schema(RerankResponse.model_json_schema())}},
            }
        },
    },
)
async def api_rerank_v1(request: Request) -> dict[str, Any]:
    """
    Параметры:
    - request: входящий HTTP-запрос.
    - payload: валидированное тело rerank-запроса.

    Что делает:
    - Проксирует запрос на upstream `/v1/rerank`.

    Выходные данные:
    - JSON-ответ reranker.
    """
    raw = await read_request_body_as_dict(request)
    payload = RerankRequestModel.model_validate(raw)
    return await _rerank_impl(request, payload.model_dump(exclude_none=True), RERANK_V1_PATH)


@router.post(
    "/api/reranker/rerank/v2",
    summary="Rerank (v2)",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "schema": inline_schema(RerankRequestModel.model_json_schema()),
                    "examples": {
                        "Без logprobs": {
                            "summary": "Обычный запрос",
                            "value": {
                                "model": "qwen3-reranker-4b",
                                "query": "Что такое квантовая запутанность?",
                                "documents": [
                                    "Квантовая запутанность — явление, при котором состояния двух частиц взаимозависимы.",
                                    "Фотосинтез — процесс преобразования световой энергии в химическую у растений.",
                                    "Принцип суперпозиции описывает нахождение квантовой системы в нескольких состояниях одновременно.",
                                ],
                                "top_n": 2,
                            },
                        },
                        "С logprobs": {
                            "summary": "С логитами (logprobs)",
                            "value": {
                                "model": "qwen3-reranker-4b",
                                "query": "Что такое квантовая запутанность?",
                                "documents": [
                                    "Квантовая запутанность — явление, при котором состояния двух частиц взаимозависимы.",
                                    "Фотосинтез — процесс преобразования световой энергии в химическую у растений.",
                                ],
                                "top_n": 2,
                                "logprobs": True,
                                "top_logprobs": 2,
                            },
                        },
                    },
                }
            },
            "required": True,
        },
        "responses": {
            "200": {
                "description": "Successful Response",
                "content": {"application/json": {"schema": inline_schema(RerankResponse.model_json_schema())}},
            }
        },
    },
)
async def api_rerank_v2(request: Request) -> dict[str, Any]:
    """
    Параметры:
    - request: входящий HTTP-запрос.
    - payload: валидированное тело rerank-запроса.

    Что делает:
    - Проксирует запрос на upstream `/v2/rerank`.

    Выходные данные:
    - JSON-ответ reranker.
    """
    raw = await read_request_body_as_dict(request)
    payload = RerankRequestModel.model_validate(raw)
    return await _rerank_impl(request, payload.model_dump(exclude_none=True), RERANK_V2_PATH)


@router.post(
    "/api/reranker/score",
    summary="Score (v1)",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "schema": inline_schema(ScoreRequestModel.model_json_schema()),
                    "examples": {
                        "Без logprobs": {
                            "summary": "Обычный запрос",
                            "value": {
                                "model": "qwen3-reranker-4b",
                                "text_1": "Что такое квантовая запутанность?",
                                "text_2": "Квантовая запутанность — явление, при котором состояния двух частиц взаимозависимы.",
                            },
                        },
                        "С logprobs": {
                            "summary": "С логитами (logprobs)",
                            "value": {
                                "model": "qwen3-reranker-4b",
                                "text_1": "Что такое квантовая запутанность?",
                                "text_2": "Квантовая запутанность — явление, при котором состояния двух частиц взаимозависимы.",
                                "logprobs": True,
                                "top_logprobs": 2,
                            },
                        },
                    },
                }
            },
            "required": True,
        },
        "responses": {
            "200": {
                "description": "Successful Response",
                "content": {"application/json": {"schema": inline_schema(ScoreResponse.model_json_schema())}},
            }
        },
    },
)
async def api_reranker_score(request: Request) -> dict[str, Any]:
    """
    Параметры:
    - request: входящий HTTP-запрос.
    - payload: валидированное тело score-запроса.

    Что делает:
    - Валидирует формат score-запроса и проксирует его на upstream `/v1/score`.

    Выходные данные:
    - JSON-ответ score endpoint.
    """
    raw = await read_request_body_as_dict(request)
    payload = ScoreRequestModel.model_validate(raw)
    body_data = _ensure_rerank_temperature(payload.model_dump(exclude_none=True))
    _validate_score_payload(body_data)

    target = await _resolve_reranker_target(body_data.get("model"))
    model = str(body_data.get("model") or target.get("model"))
    upstream_payload = dict(body_data)
    upstream_payload["model"] = target["model_vllm"]

    request.state.model = model
    request.state.upstream = target.get("base_url", "")
    upstream_base = _normalize_reranker_base_url(target["base_url"])
    return await post_json_to(upstream_base, SCORE_V1_PATH, upstream_payload)
