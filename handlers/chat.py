from typing import Any

from constants import OPENAI_CHAT_COMPLETIONS_PATH
from schemas import ChatRequest, GenerateRequest
from services.model_catalog import ModelEntry
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

_SAMPLING_DEFAULTS: dict[str, Any] = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
}


def _pick(body_value: Any, options_value: Any, default: Any) -> Any:
    """Return the first non-None value from body → options → default."""
    if body_value is not None:
        return body_value
    if options_value is not None:
        return options_value
    return default


def build_sampling_payload(body: ChatRequest | GenerateRequest) -> dict[str, Any]:
    opts = body.options or {}
    opts_dict = opts.model_dump(exclude_none=True) if hasattr(opts, "model_dump") else {}

    payload: dict[str, Any] = {
        key: _pick(getattr(body, key, None), opts_dict.get(key), default)
        for key, default in _SAMPLING_DEFAULTS.items()
    }

    seed_raw = getattr(body, "seed", None) or opts_dict.get("seed")
    if seed_raw is not None:
        try:
            payload["seed"] = int(seed_raw)
        except (TypeError, ValueError):
            pass

    return payload


def _normalize_vision_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Ollama-style ``images`` lists to OpenAI vision content parts."""
    result: list[dict[str, Any]] = []
    for msg in messages:
        images: list[str] | None = msg.get("images")
        if not images:
            result.append({k: v for k, v in msg.items() if k != "images"})
            continue

        content_parts: list[dict[str, Any]] = []

        text = msg.get("content")
        if isinstance(text, str) and text:
            content_parts.append({"type": "text", "text": text})
        elif isinstance(text, list):
            content_parts.extend(text)

        for image in images:
            if not isinstance(image, str):
                continue
            if image.startswith(("http://", "https://")) or image.startswith("data:"):
                url = image
            else:
                url = f"data:image/jpeg;base64,{image}"
            content_parts.append({"type": "image_url", "image_url": {"url": url}})

        normalized = {k: v for k, v in msg.items() if k not in ("content", "images")}
        normalized["content"] = content_parts
        result.append(normalized)

    return result


def resolve_chat_messages(body: ChatRequest) -> list[dict[str, Any]]:
    if isinstance(body.messages, list) and body.messages:
        raw = [
            m.model_dump(exclude_none=True) if hasattr(m, "model_dump") else m
            for m in body.messages
        ]
        return _normalize_vision_messages(raw)

    for field in (body.prompt, body.input, body.text, body.query):
        if field is not None:
            return [{"role": "user", "content": str(field)}]

    if isinstance(body.message, dict) and body.message.get("content") is not None:
        return [
            {
                "role": str(body.message.get("role") or "user"),
                "content": str(body.message["content"]),
            }
        ]

    return [{"role": "user", "content": ""}]


def resolve_generate_prompt(body: GenerateRequest) -> str:
    if body.prompt is not None:
        return str(body.prompt)

    for field in (body.input, body.text, body.query):
        if field is not None:
            return str(field)

    if isinstance(body.message, dict) and body.message.get("content") is not None:
        return str(body.message["content"])

    if isinstance(body.messages, list):
        parts = [
            str(msg["content"])
            for msg in body.messages
            if isinstance(msg, dict) and msg.get("content") is not None
        ]
        if parts:
            return "\n".join(parts)

    return ""


async def handle_chat(body: ChatRequest, target: ModelEntry) -> dict[str, Any]:
    messages = resolve_chat_messages(body)
    estimated_input = estimate_chat_input_tokens(messages)
    token_budget = analyze_max_tokens_budget(
        body.model_dump(),
        estimated_input_tokens=estimated_input,
        max_context_tokens=target.get("max_context_tokens"),
        min_context_headroom=target.get("min_context_headroom"),
        default_max_tokens=target.get("default_max_tokens"),
    )

    payload = {
        "model": target["model_vllm"],
        "messages": messages,
        "max_tokens": int(token_budget["resolved_max_tokens"]),
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        **build_sampling_payload(body),
    }

    if body.logprobs:
        payload["logprobs"] = True
        if body.top_logprobs is not None:
            payload["top_logprobs"] = body.top_logprobs

    start_ns = ns()
    data = await post_json_to(
        target["base_url"], OPENAI_CHAT_COMPLETIONS_PATH, payload, model_type=target["type"]
    )

    model_name = body.model or target["model"]
    content = strip_reasoning_artifacts(extract_chat_text(data))
    done_reason = extract_finish_reason(data)
    usage = data.get("usage") if isinstance(data, dict) else None

    logprobs = None
    if body.logprobs and isinstance(data, dict):
        choices = data.get("choices", [])
        if choices:
            logprobs = choices[0].get("logprobs")

    return ollama_response(model_name, content, start_ns, done_reason=done_reason, usage=usage, logprobs=logprobs)


async def handle_generate(body: GenerateRequest, target: ModelEntry) -> dict[str, Any]:
    prompt = resolve_generate_prompt(body)
    estimated_input = estimate_input_tokens_from_text(prompt) + 16
    token_budget = analyze_max_tokens_budget(
        body.model_dump(),
        estimated_input_tokens=estimated_input,
        max_context_tokens=target.get("max_context_tokens"),
        min_context_headroom=target.get("min_context_headroom"),
        default_max_tokens=target.get("default_max_tokens"),
    )

    payload = {
        "model": target["model_vllm"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(token_budget["resolved_max_tokens"]),
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        **build_sampling_payload(body),
    }

    start_ns = ns()
    data = await post_json_to(
        target["base_url"], OPENAI_CHAT_COMPLETIONS_PATH, payload, model_type=target["type"]
    )

    model_name = body.model or target["model"]
    content = strip_reasoning_artifacts(extract_chat_text(data))
    done_reason = extract_finish_reason(data)
    usage = data.get("usage") if isinstance(data, dict) else None
    return ollama_response(model_name, content, start_ns, done_reason=done_reason, usage=usage)
