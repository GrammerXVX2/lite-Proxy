import asyncio
import random
from typing import Any, Dict

import httpx
from fastapi import HTTPException

from settings import (
    UPSTREAM_HTTP_LIMITS,
    UPSTREAM_HTTP_TIMEOUT,
    UPSTREAM_RETRY_ATTEMPTS,
    UPSTREAM_RETRY_BASE_DELAY_SECONDS,
    UPSTREAM_RETRY_JITTER_SECONDS,
    UPSTREAM_RETRY_STATUS_CODES,
)


_SHARED_HTTP_CLIENT: httpx.AsyncClient | None = None


def _new_http_client() -> httpx.AsyncClient:
    """
    Параметры:
    - отсутствуют.

    Что делает:
    - Создаёт новый `httpx.AsyncClient` с таймаутами и лимитами из настроек.

    Выходные данные:
    - Экземпляр `httpx.AsyncClient`.
    """
    return httpx.AsyncClient(timeout=UPSTREAM_HTTP_TIMEOUT, limits=UPSTREAM_HTTP_LIMITS)


async def startup_http_client() -> None:
    """
    Параметры:
    - отсутствуют.

    Что делает:
    - Инициализирует shared HTTP-клиент при старте приложения (лениво, один раз).

    Выходные данные:
    - отсутствуют.
    """
    global _SHARED_HTTP_CLIENT
    if _SHARED_HTTP_CLIENT is None:
        _SHARED_HTTP_CLIENT = _new_http_client()


async def shutdown_http_client() -> None:
    """
    Параметры:
    - отсутствуют.

    Что делает:
    - Закрывает shared HTTP-клиент при остановке приложения.

    Выходные данные:
    - отсутствуют.
    """
    global _SHARED_HTTP_CLIENT
    if _SHARED_HTTP_CLIENT is not None:
        await _SHARED_HTTP_CLIENT.aclose()
        _SHARED_HTTP_CLIENT = None


async def get_http_client() -> httpx.AsyncClient:
    """
    Параметры:
    - отсутствуют.

    Что делает:
    - Возвращает shared HTTP-клиент, создавая его при необходимости.

    Выходные данные:
    - Экземпляр `httpx.AsyncClient`.
    """
    global _SHARED_HTTP_CLIENT
    if _SHARED_HTTP_CLIENT is None:
        _SHARED_HTTP_CLIENT = _new_http_client()
    return _SHARED_HTTP_CLIENT


def _retry_delay(attempt_index: int) -> float:
    """
    Параметры:
    - attempt_index: номер попытки (с 0).

    Что делает:
    - Рассчитывает задержку перед повтором с exponential backoff и jitter.

    Выходные данные:
    - Задержка в секундах.
    """
    base = max(0.0, UPSTREAM_RETRY_BASE_DELAY_SECONDS)
    jitter = max(0.0, UPSTREAM_RETRY_JITTER_SECONDS)
    return base * (2 ** attempt_index) + random.uniform(0.0, jitter)


def _is_retryable_request_error(exc: httpx.RequestError) -> bool:
    """
    Параметры:
    - exc: исключение `httpx.RequestError`.

    Что делает:
    - Проверяет, относится ли ошибка к типам, для которых допустим retry.

    Выходные данные:
    - `True`, если ошибку можно ретраить, иначе `False`.
    """
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


async def post_json_to(base_url: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Параметры:
    - base_url: базовый URL upstream-сервиса.
    - path: путь endpoint'а.
    - payload: JSON-тело POST-запроса.

    Что делает:
    - Выполняет POST-запрос в upstream с retry-механизмом.
    - Поднимает HTTPException при сетевых/HTTP/JSON-ошибках.

    Выходные данные:
    - Распарсенный JSON-ответ upstream.
    """
    url = f"{base_url.rstrip('/')}{path}"
    attempts = max(1, UPSTREAM_RETRY_ATTEMPTS)
    client = await get_http_client()

    for attempt in range(attempts):
        try:
            response = await client.post(url, json=payload)
        except httpx.RequestError as exc:
            if attempt + 1 < attempts and _is_retryable_request_error(exc):
                await asyncio.sleep(_retry_delay(attempt))
                continue
            raise HTTPException(status_code=502, detail=f"upstream connection error: {str(exc) or exc.__class__.__name__}")

        if response.status_code >= 400:
            if attempt + 1 < attempts and response.status_code in UPSTREAM_RETRY_STATUS_CODES:
                await asyncio.sleep(_retry_delay(attempt))
                continue
            raise HTTPException(status_code=response.status_code, detail=response.text)

        try:
            return response.json()
        except Exception:
            raise HTTPException(status_code=502, detail="upstream returned invalid json")

    raise HTTPException(status_code=502, detail="upstream retry exhausted")
