import asyncio
import random
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from fastapi import HTTPException

from services.limiter import upstream_slot
from services.model_catalog import ModelType
from settings import settings


_SHARED_HTTP_CLIENT: httpx.AsyncClient | None = None


def _new_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=settings.http_timeout, limits=settings.http_limits)


async def startup_http_client() -> None:
    global _SHARED_HTTP_CLIENT
    if _SHARED_HTTP_CLIENT is None:
        _SHARED_HTTP_CLIENT = _new_http_client()


async def shutdown_http_client() -> None:
    global _SHARED_HTTP_CLIENT
    if _SHARED_HTTP_CLIENT is not None:
        await _SHARED_HTTP_CLIENT.aclose()
        _SHARED_HTTP_CLIENT = None


async def get_http_client() -> httpx.AsyncClient:
    global _SHARED_HTTP_CLIENT
    if _SHARED_HTTP_CLIENT is None:
        _SHARED_HTTP_CLIENT = _new_http_client()
    return _SHARED_HTTP_CLIENT


def _retry_delay(attempt_index: int) -> float:
    base = max(0.0, settings.upstream_retry_base_delay_seconds)
    jitter = max(0.0, settings.upstream_retry_jitter_seconds)
    return base * (2 ** attempt_index) + random.uniform(0.0, jitter)


def _is_retryable_request_error(exc: httpx.RequestError) -> bool:
    retryable_types = (
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.WriteTimeout,
        httpx.ReadError,
        httpx.WriteError,
        httpx.RemoteProtocolError,
        httpx.PoolTimeout,
    )
    return isinstance(exc, retryable_types)


async def post_json_to(
    base_url: str,
    path: str,
    payload: dict[str, Any],
    *,
    model_type: ModelType | None = None,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    attempts = max(1, settings.upstream_retry_attempts)
    retry_codes = settings.retry_status_codes
    client = await get_http_client()

    async with upstream_slot(model_type):
        for attempt in range(attempts):
            try:
                response = await client.post(url, json=payload)
            except httpx.RequestError as exc:
                if attempt + 1 < attempts and _is_retryable_request_error(exc):
                    await asyncio.sleep(_retry_delay(attempt))
                    continue
                raise HTTPException(
                    status_code=502,
                    detail=f"upstream connection error: {str(exc) or exc.__class__.__name__}",
                )

            if response.status_code >= 400:
                if attempt + 1 < attempts and response.status_code in retry_codes:
                    await asyncio.sleep(_retry_delay(attempt))
                    continue
                raise HTTPException(status_code=response.status_code, detail=response.text)

            try:
                return response.json()
            except Exception:
                raise HTTPException(status_code=502, detail="upstream returned invalid json")

        raise HTTPException(status_code=502, detail="upstream retry exhausted")


async def stream_response_from(
    base_url: str,
    path: str,
    payload: dict[str, Any],
    *,
    model_type: ModelType | None = None,
) -> AsyncGenerator[str, None]:
    """Yield raw SSE lines from upstream while holding the concurrency slot."""
    url = f"{base_url.rstrip('/')}{path}"
    client = await get_http_client()

    async with upstream_slot(model_type):
        try:
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=body.decode(errors="replace"),
                    )
                async for line in response.aiter_lines():
                    yield line
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"upstream connection error: {str(exc) or exc.__class__.__name__}",
            )


async def ping_model(base_url: str) -> tuple[bool, str]:
    """GET /v1/models on the upstream and return (ok, error_detail)."""
    url = f"{base_url.rstrip('/')}/models"
    client = await get_http_client()
    try:
        response = await client.get(url, timeout=5.0)
        if response.status_code < 400:
            return True, ""
        return False, f"HTTP {response.status_code}"
    except httpx.RequestError as exc:
        return False, str(exc) or exc.__class__.__name__

