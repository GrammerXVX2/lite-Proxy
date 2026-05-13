from typing import Any

from fastapi import APIRouter, HTTPException, Request

from api.common import inline_schema
from constants import ERR_NO_EMBEDDING_MODELS
from handlers.embeddings import handle_embed
from schemas import EmbedRequest, EmbedResponseModel
from services.model_catalog import resolve_target
from services.request_parser import read_request_body_as_dict

router = APIRouter(tags=["embeddings"])


@router.post(
    "/api/embed",
    response_model=EmbedResponseModel,
    summary="Embed text",
    openapi_extra={
        "requestBody": {
            "content": {"application/json": {"schema": inline_schema(EmbedRequest.model_json_schema())}},
            "required": True,
        }
    },
)
async def api_embed(request: Request) -> dict[str, Any]:
    raw = await read_request_body_as_dict(request)
    body = EmbedRequest.model_validate(raw)

    try:
        target = resolve_target(body.model, expected_types={"embeddings"})
    except HTTPException as exc:
        if exc.status_code == 503:
            raise HTTPException(status_code=503, detail=ERR_NO_EMBEDDING_MODELS)
        raise

    request.state.model = body.model or target["model"]
    request.state.upstream = target.get("base_url", "")

    return await handle_embed(body, target)

