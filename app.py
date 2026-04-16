from fastapi import FastAPI

from api.chat import router as chat_router
from api.embeddings import router as embeddings_router
from api.models import router as models_router
from api.rerank import router as rerank_router
from services.upstream import shutdown_http_client, startup_http_client

app = FastAPI(
    title="liteProxy",
    description=(
        "Lightweight Ollama-style proxy without DB registry and without proxy stats. "
        "Contains only chat/generate/embed/rerank/models/tags endpoints."
    ),
    version="1.0.0",
)

app.include_router(models_router)
app.include_router(chat_router)
app.include_router(embeddings_router)
app.include_router(rerank_router)


@app.on_event("startup")
async def _startup() -> None:
    """
    Параметры:
    - отсутствуют.

    Что делает:
    - Инициализирует общий HTTP-клиент для исходящих запросов к upstream-сервисам.

    Выходные данные:
    - отсутствуют.
    """
    await startup_http_client()


@app.on_event("shutdown")
async def _shutdown() -> None:
    """
    Параметры:
    - отсутствуют.

    Что делает:
    - Корректно закрывает общий HTTP-клиент перед остановкой приложения.

    Выходные данные:
    - отсутствуют.
    """
    await shutdown_http_client()
