from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request

from constants import ERR_NO_CHAT_MODELS, ERR_STREAM_DISABLED, OPENAI_CHAT_COMPLETIONS_PATH
from schemas import OllamaTextResponseModel
from services.model_catalog import resolve_target
from services.request_parser import read_request_body_as_dict
from services.upstream import post_json_to

from api.common import (
    analyze_max_tokens_budget,
    estimate_chat_input_tokens,
    estimate_input_tokens_from_text,
    extract_chat_text,
    extract_finish_reason,
    ns,
    ollama_response,
    strip_reasoning_artifacts,
)

router = APIRouter(tags=["chat"])

CHAT_DEFAULT_SAMPLING: Dict[str, Any] = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
}


def _resolve_sampling_param(body_data: Dict[str, Any], key: str) -> Any:
    """
    Параметры:
    - body_data: тело входного запроса.
    - key: имя sampling-параметра.

    Что делает:
    - Берёт значение параметра из верхнего уровня тела запроса.
    - Если отсутствует, пытается взять из `options`.
    - Если всё ещё отсутствует, подставляет дефолт из `CHAT_DEFAULT_SAMPLING`.

    Выходные данные:
    - Значение sampling-параметра.
    """
    value = body_data.get(key)
    if value is not None:
        return value
    options = body_data.get("options") if isinstance(body_data.get("options"), dict) else {}
    value = options.get(key)
    if value is not None:
        return value
    return CHAT_DEFAULT_SAMPLING[key]


def _resolve_optional_param(body_data: Dict[str, Any], key: str) -> Any:
    """
    Параметры:
    - body_data: тело входного запроса.
    - key: имя опционального параметра.

    Что делает:
    - Ищет параметр сначала в корне запроса, затем в `options`.

    Выходные данные:
    - Найденное значение или `None`.
    """
    value = body_data.get(key)
    if value is not None:
        return value
    options = body_data.get("options") if isinstance(body_data.get("options"), dict) else {}
    return options.get(key)


def _build_sampling_payload(body_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Параметры:
    - body_data: тело входного запроса.

    Что делает:
    - Формирует полный набор sampling-параметров для upstream-вызова.

    Выходные данные:
    - Словарь sampling-параметров.
    """
    payload = {
        key: _resolve_sampling_param(body_data, key)
        for key in CHAT_DEFAULT_SAMPLING
    }

    # Optional deterministic seed (Ollama-style `options.seed`).
    seed_raw = _resolve_optional_param(body_data, "seed")
    if seed_raw is not None:
        try:
            payload["seed"] = int(seed_raw)
        except (TypeError, ValueError):
            pass

    return payload


def _extract_messages(body_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Параметры:
    - body_data: тело входного запроса.

    Что делает:
    - Возвращает нормализованный список сообщений для chat-вызова.
    - Если `messages` отсутствует, строит fallback из `prompt/input/text/query/message.content`.

    Выходные данные:
    - Список сообщений в формате `{role, content}`.
    """
    messages = body_data.get("messages")
    if isinstance(messages, list) and messages:
        return messages

    fallback_text = (
        body_data.get("prompt")
        or body_data.get("input")
        or body_data.get("text")
        or body_data.get("query")
    )
    if fallback_text is not None:
        return [{"role": "user", "content": str(fallback_text)}]

    message_obj = body_data.get("message")
    if isinstance(message_obj, dict) and message_obj.get("content") is not None:
        return [{"role": str(message_obj.get("role") or "user"), "content": str(message_obj.get("content"))}]

    return [{"role": "user", "content": ""}]


def _extract_prompt(body_data: Dict[str, Any]) -> str:
    """
    Параметры:
    - body_data: тело входного запроса.

    Что делает:
    - Извлекает строковый prompt для `/api/generate` из поддерживаемых полей.

    Выходные данные:
    - Строка prompt.
    """
    prompt = body_data.get("prompt")
    if prompt is not None:
        return str(prompt)

    for key in ("input", "text", "query"):
        if body_data.get(key) is not None:
            return str(body_data.get(key))

    message_obj = body_data.get("message")
    if isinstance(message_obj, dict) and message_obj.get("content") is not None:
        return str(message_obj.get("content"))

    messages = body_data.get("messages")
    if isinstance(messages, list):
        parts: List[str] = []
        for message in messages:
            if isinstance(message, dict) and message.get("content") is not None:
                parts.append(str(message.get("content")))
        if parts:
            return "\n".join(parts)

    return ""


@router.post(
    "/api/chat",
    summary="Chat completion (stream and reasoning disabled)",
    response_model=OllamaTextResponseModel,
)
async def api_chat(request: Request) -> Dict[str, Any]:
    """
    Параметры:
    - request: входящий HTTP-запрос.

    Что делает:
    - Обрабатывает Ollama-style chat-запрос.
    - Принудительно отключает stream/reasoning, рассчитывает token budget,
            отправляет запрос в upstream и возвращает ответ в Ollama-style формате.

    Выходные данные:
    - JSON-ответ chat completion в формате `OllamaTextResponseModel`.
    """
    body_data = await read_request_body_as_dict(request)

    if bool(body_data.get("stream", False)):
        raise HTTPException(status_code=400, detail=ERR_STREAM_DISABLED)

    requested_model = body_data.get("model")
    try:
        target = resolve_target(requested_model, expected_types={"chat"})
    except HTTPException as exc:
        if exc.status_code == 503:
            raise HTTPException(status_code=503, detail=ERR_NO_CHAT_MODELS)
        raise

    model = requested_model or target["model"]
    messages = _extract_messages(body_data)

    estimated_input_tokens = estimate_chat_input_tokens(messages)
    token_budget = analyze_max_tokens_budget(
        body_data,
        estimated_input_tokens=estimated_input_tokens,
        max_context_tokens=target.get("max_context_tokens"),
        min_context_headroom=target.get("min_context_headroom"),
        default_max_tokens=target.get("default_max_tokens"),
    )

    payload = {
        "model": target["model_vllm"],
        "messages": messages,
        "max_tokens": int(token_budget["resolved_max_tokens"]),
        "stream": False,
        # Lite contract: reasoning is always disabled.
        "chat_template_kwargs": {"enable_thinking": False},
        **_build_sampling_payload(body_data),
    }

    request.state.model = model
    request.state.upstream = target.get("base_url", "")

    start_ns = ns()
    data = await post_json_to(target["base_url"], OPENAI_CHAT_COMPLETIONS_PATH, payload)

    content = strip_reasoning_artifacts(extract_chat_text(data))
    done_reason = extract_finish_reason(data)
    usage = data.get("usage") if isinstance(data, dict) else None
    return ollama_response(model, content, start_ns, done_reason=done_reason, usage=usage)


@router.post(
    "/api/generate",
    summary="Prompt completion (stream and reasoning disabled)",
    response_model=OllamaTextResponseModel,
)
async def api_generate(request: Request) -> Dict[str, Any]:
    """
    Параметры:
    - request: входящий HTTP-запрос.

    Что делает:
    - Обрабатывает Ollama-style generate-запрос.
    - Преобразует prompt в chat/completions payload для vLLM,
            применяет budget и возвращает ответ в Ollama-style формате.

    Выходные данные:
    - JSON-ответ prompt completion в формате `OllamaTextResponseModel`.
    """
    body_data = await read_request_body_as_dict(request)

    if bool(body_data.get("stream", False)):
        raise HTTPException(status_code=400, detail=ERR_STREAM_DISABLED)

    requested_model = body_data.get("model")
    try:
        target = resolve_target(requested_model, expected_types={"chat"})
    except HTTPException as exc:
        if exc.status_code == 503:
            raise HTTPException(status_code=503, detail=ERR_NO_CHAT_MODELS)
        raise

    model = requested_model or target["model"]
    prompt = _extract_prompt(body_data)

    estimated_input_tokens = estimate_input_tokens_from_text(prompt) + 16
    token_budget = analyze_max_tokens_budget(
        body_data,
        estimated_input_tokens=estimated_input_tokens,
        max_context_tokens=target.get("max_context_tokens"),
        min_context_headroom=target.get("min_context_headroom"),
        default_max_tokens=target.get("default_max_tokens"),
    )

    payload = {
        "model": target["model_vllm"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(token_budget["resolved_max_tokens"]),
        "stream": False,
        # Lite contract: reasoning is always disabled.
        "chat_template_kwargs": {"enable_thinking": False},
        **_build_sampling_payload(body_data),
    }

    request.state.model = model
    request.state.upstream = target.get("base_url", "")

    start_ns = ns()
    data = await post_json_to(target["base_url"], OPENAI_CHAT_COMPLETIONS_PATH, payload)

    content = strip_reasoning_artifacts(extract_chat_text(data))
    done_reason = extract_finish_reason(data)
    usage = data.get("usage") if isinstance(data, dict) else None
    return ollama_response(model, content, start_ns, done_reason=done_reason, usage=usage)
