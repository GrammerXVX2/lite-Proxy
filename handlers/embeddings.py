from typing import Any

from fastapi import HTTPException

from constants import EMBEDDING_PATH_CANDIDATES
from schemas import EmbedRequest
from services.model_catalog import ModelEntry, ModelType
from services.upstream import post_json_to

from api.common import ns


async def _post_with_fallback(
    base_url: str, model_id: str, input_data: Any, model_type: ModelType | None = None
) -> dict[str, Any]:
    attempts = [
        (EMBEDDING_PATH_CANDIDATES[0], {"model": model_id, "input": input_data}),
        (EMBEDDING_PATH_CANDIDATES[1], {"model": model_id, "input": input_data}),
        (EMBEDDING_PATH_CANDIDATES[2], {"inputs": input_data}),
        (EMBEDDING_PATH_CANDIDATES[3], {"inputs": input_data}),
    ]

    last_exc: HTTPException | None = None
    for path, payload in attempts:
        try:
            return await post_json_to(base_url, path, payload, model_type=model_type)
        except HTTPException as exc:
            last_exc = exc
            if exc.status_code in (404, 405, 422):
                continue
            raise

    if last_exc is not None:
        raise last_exc
    raise HTTPException(status_code=502, detail="embedding upstream is unavailable")


def _extract_embeddings(data: Any) -> list[list[float]]:
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        return [item.get("embedding", []) for item in data["data"] if isinstance(item, dict)]

    if isinstance(data, dict) and isinstance(data.get("embeddings"), list):
        emb = data["embeddings"]
        if emb and isinstance(emb[0], list):
            return emb

    if isinstance(data, dict) and isinstance(data.get("embedding"), list):
        emb = data["embedding"]
        if emb and isinstance(emb[0], (int, float)):
            return [emb]

    if isinstance(data, list) and data and isinstance(data[0], list):
        return data
    if isinstance(data, list) and data and isinstance(data[0], (int, float)):
        return [data]

    return []


def resolve_embed_input(body: EmbedRequest) -> Any:
    if body.input is not None:
        return body.input

    for field in (body.prompt, body.text):
        if field is not None:
            return field

    if isinstance(body.message, dict) and body.message.get("content") is not None:
        return body.message["content"]

    if isinstance(body.messages, list):
        parts = [
            str(msg["content"])
            for msg in body.messages
            if isinstance(msg, dict) and msg.get("content") is not None
        ]
        if parts:
            return "\n".join(parts)

    return None


async def handle_embed(body: EmbedRequest, target: ModelEntry) -> dict[str, Any]:
    input_data = resolve_embed_input(body)
    if input_data is None:
        raise HTTPException(status_code=400, detail="input is required")

    start_ns = ns()
    data = await _post_with_fallback(target["base_url"], target["model_vllm"], input_data, model_type=target["type"])

    embeddings = _extract_embeddings(data)
    usage = data.get("usage") if isinstance(data, dict) else {}

    return {
        "model": body.model or target["model"],
        "embedding": embeddings[0] if embeddings else [],
        "embeddings": embeddings,
        "total_duration": max(0, ns() - start_ns),
        "load_duration": 0,
        "prompt_eval_count": usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0,
    }
