import json
from collections.abc import AsyncGenerator
from typing import Any

from constants import OPENAI_CHAT_COMPLETIONS_PATH, OPENAI_COMPLETIONS_PATH
from schemas import ChatRequest, GenerateRequest
from services.model_catalog import ModelEntry
from services.upstream import post_json_to, stream_response_from

from api.common import (
    analyze_max_tokens_budget,
    estimate_chat_input_tokens,
    estimate_input_tokens_from_text,
    extract_chat_text,
    extract_finish_reason,
    now_iso,
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
            raw_lp = choices[0].get("logprobs")
            if raw_lp:
                logprobs = _convert_chat_logprobs(raw_lp)

    return ollama_response(model_name, content, start_ns, done_reason=done_reason, usage=usage, logprobs=logprobs)


async def handle_generate(body: GenerateRequest, target: ModelEntry) -> dict[str, Any]:
    prompt = resolve_generate_prompt(body)
    start_ns = ns()
    model_name = body.model or target["model"]

    if body.raw:
        return await _handle_generate_raw(body, target, prompt, model_name, start_ns)

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

    if body.logprobs:
        payload["logprobs"] = True
        if body.top_logprobs is not None:
            payload["top_logprobs"] = body.top_logprobs

    data = await post_json_to(
        target["base_url"], OPENAI_CHAT_COMPLETIONS_PATH, payload, model_type=target["type"]
    )

    content = strip_reasoning_artifacts(extract_chat_text(data))
    done_reason = extract_finish_reason(data)
    usage = data.get("usage") if isinstance(data, dict) else None

    logprobs = None
    if body.logprobs and isinstance(data, dict):
        choices = data.get("choices", [])
        if choices:
            raw_lp = choices[0].get("logprobs")
            if raw_lp:
                logprobs = _convert_chat_logprobs(raw_lp)

    return ollama_response(model_name, content, start_ns, done_reason=done_reason, usage=usage, logprobs=logprobs)


def _convert_completions_logprobs(raw_lp: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert vLLM /v1/completions logprobs to Ollama list format."""
    tokens: list[str] = raw_lp.get("tokens") or []
    token_logprobs: list[float] = raw_lp.get("token_logprobs") or []
    top_logprobs_list: list[dict[str, float]] = raw_lp.get("top_logprobs") or []

    result: list[dict[str, Any]] = []
    for i, token in enumerate(tokens):
        entry: dict[str, Any] = {
            "token": token,
            "logprob": token_logprobs[i] if i < len(token_logprobs) else 0.0,
        }
        if i < len(top_logprobs_list) and top_logprobs_list[i]:
            entry["top_logprobs"] = top_logprobs_list[i]
        result.append(entry)
    return result


def _convert_chat_logprobs(raw_lp: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert vLLM /chat/completions logprobs to Ollama list format.

    vLLM chat format:
      {"content": [{"token": "yes", "logprob": -0.4, "top_logprobs": [{"token": "yes", "logprob": -0.4}, ...]}]}

    Ollama target format:
      [{"token": "yes", "logprob": -0.4, "top_logprobs": {"yes": -0.4, "no": -3.5}}]
    """
    content: list[dict[str, Any]] = raw_lp.get("content") or []
    result: list[dict[str, Any]] = []
    for item in content:
        entry: dict[str, Any] = {
            "token": item.get("token", ""),
            "logprob": item.get("logprob", 0.0),
        }
        top_raw: list[dict[str, Any]] = item.get("top_logprobs") or []
        if top_raw:
            entry["top_logprobs"] = {t["token"]: t["logprob"] for t in top_raw if "token" in t}
        result.append(entry)
    return result


async def _handle_generate_raw(
    body: GenerateRequest,
    target: ModelEntry,
    prompt: str,
    model_name: str,
    start_ns: int,
) -> dict[str, Any]:
    """Handle raw-mode generate: call /v1/completions, bypass chat template."""
    estimated_input = estimate_input_tokens_from_text(prompt) + 16
    token_budget = analyze_max_tokens_budget(
        body.model_dump(),
        estimated_input_tokens=estimated_input,
        max_context_tokens=target.get("max_context_tokens"),
        min_context_headroom=target.get("min_context_headroom"),
        default_max_tokens=target.get("default_max_tokens"),
    )

    payload: dict[str, Any] = {
        "model": target["model_vllm"],
        "prompt": prompt,
        "max_tokens": int(token_budget["resolved_max_tokens"]),
        "stream": False,
        **build_sampling_payload(body),
    }

    if body.logprobs:
        # /v1/completions uses logprobs=N (integer) for top-N logprobs
        payload["logprobs"] = body.top_logprobs if body.top_logprobs is not None else 1

    data = await post_json_to(
        target["base_url"], OPENAI_COMPLETIONS_PATH, payload, model_type=target["type"]
    )

    choices = data.get("choices") or [] if isinstance(data, dict) else []
    choice = choices[0] if choices else {}
    content = choice.get("text", "")
    done_reason = choice.get("finish_reason", "stop")
    usage = data.get("usage") if isinstance(data, dict) else None

    ollama_logprobs = None
    if body.logprobs:
        raw_lp = choice.get("logprobs")
        if raw_lp:
            ollama_logprobs = _convert_completions_logprobs(raw_lp)

    return ollama_response(model_name, content, start_ns, done_reason=done_reason, usage=usage, logprobs=ollama_logprobs)


def _parse_sse_chunk(line: str) -> dict[str, Any] | None:
    """Parse a single SSE data line into a dict, or return None to skip."""
    if not line.startswith("data: "):
        return None
    data = line[6:].strip()
    if data == "[DONE]":
        return None
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


def _ollama_stream_chunk(model_name: str, content: str, done: bool, done_reason: str | None = None) -> bytes:
    chunk: dict[str, Any] = {
        "model": model_name,
        "created_at": now_iso(),
        "message": {"role": "assistant", "content": content},
        "done": done,
    }
    if done and done_reason:
        chunk["done_reason"] = done_reason
    return (json.dumps(chunk, ensure_ascii=False) + "\n").encode()


async def handle_chat_stream(body: ChatRequest, target: ModelEntry) -> AsyncGenerator[bytes, None]:
    messages = resolve_chat_messages(body)
    estimated_input = estimate_chat_input_tokens(messages)
    token_budget = analyze_max_tokens_budget(
        body.model_dump(),
        estimated_input_tokens=estimated_input,
        max_context_tokens=target.get("max_context_tokens"),
        min_context_headroom=target.get("min_context_headroom"),
        default_max_tokens=target.get("default_max_tokens"),
    )

    payload: dict[str, Any] = {
        "model": target["model_vllm"],
        "messages": messages,
        "max_tokens": int(token_budget["resolved_max_tokens"]),
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
        **build_sampling_payload(body),
    }

    model_name = body.model or target["model"]

    async for line in stream_response_from(
        target["base_url"], OPENAI_CHAT_COMPLETIONS_PATH, payload, model_type=target["type"]
    ):
        chunk = _parse_sse_chunk(line)
        if chunk is None:
            continue
        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        delta_content = (choice.get("delta") or {}).get("content") or ""
        finish_reason: str | None = choice.get("finish_reason")
        done = finish_reason is not None
        delta_content = strip_reasoning_artifacts(delta_content)
        yield _ollama_stream_chunk(model_name, delta_content, done, finish_reason)


async def handle_generate_stream(body: GenerateRequest, target: ModelEntry) -> AsyncGenerator[bytes, None]:
    prompt = resolve_generate_prompt(body)
    estimated_input = estimate_input_tokens_from_text(prompt) + 16
    token_budget = analyze_max_tokens_budget(
        body.model_dump(),
        estimated_input_tokens=estimated_input,
        max_context_tokens=target.get("max_context_tokens"),
        min_context_headroom=target.get("min_context_headroom"),
        default_max_tokens=target.get("default_max_tokens"),
    )

    payload: dict[str, Any] = {
        "model": target["model_vllm"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(token_budget["resolved_max_tokens"]),
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
        **build_sampling_payload(body),
    }

    model_name = body.model or target["model"]

    async for line in stream_response_from(
        target["base_url"], OPENAI_CHAT_COMPLETIONS_PATH, payload, model_type=target["type"]
    ):
        chunk = _parse_sse_chunk(line)
        if chunk is None:
            continue
        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        delta_content = (choice.get("delta") or {}).get("content") or ""
        finish_reason: str | None = choice.get("finish_reason")
        done = finish_reason is not None
        delta_content = strip_reasoning_artifacts(delta_content)
        yield _ollama_stream_chunk(model_name, delta_content, done, finish_reason)
