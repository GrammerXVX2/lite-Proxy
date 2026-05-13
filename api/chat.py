from typing import Any

from fastapi import APIRouter, HTTPException, Request

from api.common import inline_schema
from constants import ERR_NO_CHAT_MODELS, ERR_STREAM_DISABLED
from handlers.chat import handle_chat, handle_generate
from schemas import ChatRequest, GenerateRequest, OllamaTextResponseModel
from services.model_catalog import resolve_target
from services.request_parser import read_request_body_as_dict

router = APIRouter(tags=["chat"])


@router.post(
    "/api/chat",
    summary="Chat completion (stream and reasoning disabled)",
    response_model=OllamaTextResponseModel,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "schema": inline_schema(ChatRequest.model_json_schema()),
                    "examples": {
                        "Без logprobs": {
                            "summary": "Обычный запрос",
                            "value": {
                                "model": "qwen35-122b-a10b-fp8",
                                "temperature": 0.5,
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": "Опиши картинку одним предложением.",
                                        "images": [
                                            "https://images.unsplash.com/photo-1518717758536-85ae29035b6d?w=640"
                                        ],
                                    }
                                ],
                            },
                        },
                        "С logprobs": {
                            "summary": "С логитами (logprobs)",
                            "value": {
                                "model": "qwen35-122b-a10b-fp8",
                                "temperature": 0.5,
                                "logprobs": True,
                                "top_logprobs": 2,
                                "messages": [
                                    {"role": "user", "content": "Что такое квантовая механика?"}
                                ],
                            },
                        },
                    },
                }
            },
            "required": True,
        }
    },
)
async def api_chat(request: Request) -> dict[str, Any]:
    raw = await read_request_body_as_dict(request)
    body = ChatRequest.model_validate(raw)

    if body.stream:
        raise HTTPException(status_code=400, detail=ERR_STREAM_DISABLED)

    try:
        target = resolve_target(body.model, expected_types={"chat"})
    except HTTPException as exc:
        if exc.status_code == 503:
            raise HTTPException(status_code=503, detail=ERR_NO_CHAT_MODELS)
        raise

    request.state.model = body.model or target["model"]
    request.state.upstream = target.get("base_url", "")

    return await handle_chat(body, target)


@router.post(
    "/api/generate",
    summary="Prompt completion (stream and reasoning disabled)",
    response_model=OllamaTextResponseModel,
    openapi_extra={
        "requestBody": {
            "content": {"application/json": {"schema": inline_schema(GenerateRequest.model_json_schema())}},
            "required": True,
        }
    },
)
async def api_generate(request: Request) -> dict[str, Any]:
    raw = await read_request_body_as_dict(request)
    body = GenerateRequest.model_validate(raw)

    if body.stream:
        raise HTTPException(status_code=400, detail=ERR_STREAM_DISABLED)

    try:
        target = resolve_target(body.model, expected_types={"chat"})
    except HTTPException as exc:
        if exc.status_code == 503:
            raise HTTPException(status_code=503, detail=ERR_NO_CHAT_MODELS)
        raise

    request.state.model = body.model or target["model"]
    request.state.upstream = target.get("base_url", "")

    return await handle_generate(body, target)

