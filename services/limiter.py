"""Upstream concurrency limiter — per model-type.

Each model type (chat / embeddings / reranker) gets its own semaphore so
that a flood of embedding requests cannot block chat slots and vice-versa.
Excess requests queue in the asyncio event-loop (FIFO) instead of being
rejected — callers get their answer, just later.

Optionally syncs counters to Redis for cross-instance observability.
"""
import asyncio
import contextlib
from typing import Any, AsyncIterator

from settings import settings

# Import ModelType without triggering circular imports — model_catalog only
# depends on settings, not on limiter or upstream.
from services.model_catalog import ModelType

_ALL_TYPES: tuple[ModelType, ...] = ("chat", "embeddings", "reranker")

# ─── module-level state ───────────────────────────────────────────────────────

_semaphores: dict[ModelType, asyncio.Semaphore] = {}
_active: dict[ModelType, int] = {t: 0 for t in _ALL_TYPES}
_waiting: dict[ModelType, int] = {t: 0 for t in _ALL_TYPES}

_redis: Any = None  # redis.asyncio.Redis | None — imported lazily


def _redis_key(kind: str, model_type: ModelType) -> str:
    return f"liteproxy:{model_type}:{kind}"


# ─── lifecycle ────────────────────────────────────────────────────────────────


async def setup_limiter() -> None:
    global _semaphores, _active, _waiting, _redis
    cap = settings.max_concurrent_per_type
    _semaphores = {t: asyncio.Semaphore(cap) for t in _ALL_TYPES}
    _active = {t: 0 for t in _ALL_TYPES}
    _waiting = {t: 0 for t in _ALL_TYPES}

    if settings.redis_url:
        try:
            import redis.asyncio as aioredis  # noqa: PLC0415

            client = aioredis.from_url(settings.redis_url, decode_responses=True)
            await client.ping()
            for t in _ALL_TYPES:
                await client.set(_redis_key("active", t), 0)
                await client.set(_redis_key("waiting", t), 0)
            _redis = client
        except Exception:
            _redis = None


async def shutdown_limiter() -> None:
    global _redis
    if _redis is not None:
        with contextlib.suppress(Exception):
            await _redis.aclose()
        _redis = None


# ─── context manager ─────────────────────────────────────────────────────────


@contextlib.asynccontextmanager
async def upstream_slot(model_type: ModelType | None) -> AsyncIterator[None]:
    """Acquire one concurrency slot for *model_type*; release on exit.

    When all slots for that type are taken, the coroutine suspends here
    until a slot is freed.  Passing ``None`` skips limiting entirely.
    """
    sem = _semaphores.get(model_type) if model_type is not None else None  # type: ignore[arg-type]
    if sem is None:
        yield
        return

    mt: ModelType = model_type  # type: ignore[assignment]

    _waiting[mt] += 1
    _fire(_redis_key("waiting", mt), delta=1)
    try:
        await sem.acquire()
    finally:
        _waiting[mt] -= 1
        _fire(_redis_key("waiting", mt), delta=-1)

    _active[mt] += 1
    _fire(_redis_key("active", mt), delta=1)
    try:
        yield
    finally:
        sem.release()
        _active[mt] -= 1
        _fire(_redis_key("active", mt), delta=-1)


# ─── Redis fire-and-forget helpers ───────────────────────────────────────────


def _fire(key: str, delta: int) -> None:
    if _redis is not None:
        asyncio.ensure_future(_redis_adjust(key, delta))


async def _redis_adjust(key: str, delta: int) -> None:
    with contextlib.suppress(Exception):
        if delta > 0:
            await _redis.incr(key)
        else:
            await _redis.decr(key)


# ─── public status ────────────────────────────────────────────────────────────


def limiter_status() -> dict[str, dict[str, int]]:
    """Return per-type concurrency metrics (locally observed)."""
    cap = settings.max_concurrent_per_type
    return {
        t: {
            "max_concurrent": cap,
            "active": max(0, _active.get(t, 0)),
            "waiting": max(0, _waiting.get(t, 0)),
            "free_slots": max(0, cap - _active.get(t, 0)),
        }
        for t in _ALL_TYPES
    }
