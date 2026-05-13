from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.chat import router as chat_router
from api.embeddings import router as embeddings_router
from api.models import router as models_router
from api.rerank import router as rerank_router
from services.limiter import setup_limiter, shutdown_limiter
from services.upstream import shutdown_http_client, startup_http_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup_http_client()
    await setup_limiter()
    yield
    await shutdown_limiter()
    await shutdown_http_client()


app = FastAPI(
    title="liteProxy",
    description=(
        "Lightweight Ollama-style proxy without DB registry and without proxy stats. "
        "Contains only chat/generate/embed/rerank/models/tags endpoints."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(models_router)
app.include_router(chat_router)
app.include_router(embeddings_router)
app.include_router(rerank_router)

