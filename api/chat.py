from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from api.common import inline_schema
from constants import ERR_NO_CHAT_MODELS
from handlers.chat import handle_chat, handle_chat_stream, handle_generate, handle_generate_stream
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
async def api_chat(request: Request) -> Any:
    raw = await read_request_body_as_dict(request)
    body = ChatRequest.model_validate(raw)

    try:
        target = resolve_target(body.model, expected_types={"chat"})
    except HTTPException as exc:
        if exc.status_code == 503:
            raise HTTPException(status_code=503, detail=ERR_NO_CHAT_MODELS)
        raise

    request.state.model = body.model or target["model"]
    request.state.upstream = target.get("base_url", "")

    if body.stream:
        return StreamingResponse(
            handle_chat_stream(body, target),
            media_type="application/x-ndjson",
        )
    return await handle_chat(body, target)


@router.post(
    "/api/generate",
    summary="Prompt completion (stream and reasoning disabled)",
    response_model=OllamaTextResponseModel,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "schema": inline_schema(GenerateRequest.model_json_schema()),
                    "examples": {
                        "Без logprobs": {
                            "summary": "Обычный запрос",
                            "value": {
                                "model": "qwen35-122b-a10b-fp8",
                                "temperature": 0.5,
                                "prompt": "Что такое квантовая механика кратко?",
                            },
                        },
                        "С logprobs": {
                            "summary": "С логитами (logprobs)",
                            "value": {
                                "model": "qwen35-122b-a10b-fp8",
                                "temperature": 0.5,
                                "logprobs": True,
                                "top_logprobs": 2,
                                "prompt": "Что такое квантовая механика кратко?",
                            },
                        },
                        "Raw + logprobs (реранкер)": {
                            "summary": "Raw-режим — промпт как есть, возвращает logprobs",
                            "value": {
                                "model": "qwen35-122b-a10b-fp8",
                                "prompt": "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query. Note that the answer can only be 'yes' or 'no'.<|im_end|>\n<|im_start|>user\n<Query>: Как приготовить кофе?\n<Document>: Для приготовления кофе нужны зёрна и вода.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                                "raw": True,
                                "logprobs": True,
                                "top_logprobs": 5,
                                "options": {
                                    "num_predict": 1,
                                    "temperature": 0,
                                },
                            },
                        },
                    },
                }
            },
            "required": True,
        }
    },
)
async def api_generate(request: Request) -> Any:
    raw = await read_request_body_as_dict(request)
    body = GenerateRequest.model_validate(raw)

    try:
        target = resolve_target(body.model, expected_types={"chat"})
    except HTTPException as exc:
        if exc.status_code == 503:
            raise HTTPException(status_code=503, detail=ERR_NO_CHAT_MODELS)
        raise

    request.state.model = body.model or target["model"]
    request.state.upstream = target.get("base_url", "")

    if body.stream:
        return StreamingResponse(
            handle_generate_stream(body, target),
            media_type="application/x-ndjson",
        )
    return await handle_generate(body, target)

