from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class SamplingOptions(BaseModel):
    """Ollama-style `options` block; unknown keys are forwarded as-is."""

    model_config = ConfigDict(extra="allow")

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    seed: int | None = None
    num_ctx: int | None = None
    num_predict: int | None = None


class ChatMessage(BaseModel):
    """Single message in a chat conversation with optional image attachments."""

    model_config = ConfigDict(extra="allow")

    role: str = "user"
    content: str | list[dict[str, Any]] | None = None
    images: list[str] | None = Field(
        default=None,
        description=(
            "Image URLs or base64-encoded strings (Ollama-style). "
            "Converted to OpenAI vision content parts before forwarding to vLLM."
        ),
    )


class ChatRequest(BaseModel):
    """Body accepted by `/api/chat`."""

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {
                "model": "qwen35-122b-a10b-fp8",
                "temperature": 0.5,
                "messages": [
                    {
                        "role": "user",
                        "content": "Опиши картинку одним предложением.",
                        "images": [
                            "https://images.unsplash.com/photo-1518717758536-85ae29035b6d?w=640"
                        ],
                    }
                ],
            }
        },
    )

    model: str | None = None
    messages: list[ChatMessage] | None = None
    # Fallback single-turn fields (Ollama/OpenAI compat).
    prompt: str | None = None
    input: str | None = None
    text: str | None = None
    query: str | None = None
    message: dict[str, Any] | None = None
    # Control flags.
    stream: bool = False
    # Logprobs.
    logprobs: bool | None = None
    top_logprobs: int | None = Field(default=None, ge=0, le=20)
    # Token budget (top-level overrides `options`).
    max_tokens: int | None = None
    num_ctx: int | None = None
    # Sampling params (top-level overrides `options`).
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    seed: int | None = None
    options: SamplingOptions | None = None


class GenerateRequest(BaseModel):
    """Body accepted by `/api/generate`."""

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {
                "model": "qwen35-122b-a10b-fp8",
                "temperature": 0.5,
                "prompt": "Что такое квантовая механика кратко?",
            }
        },
    )

    model: str | None = None
    prompt: str | None = None
    input: str | None = None
    text: str | None = None
    query: str | None = None
    message: dict[str, Any] | None = None
    messages: list[dict[str, Any]] | None = None
    stream: bool = False
    max_tokens: int | None = None
    num_ctx: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    seed: int | None = None
    options: SamplingOptions | None = None


class EmbedRequest(BaseModel):
    """Body accepted by `/api/embed`."""

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {
                "model": "qwen3-embedding-4b",
                "input": "Что такое квантовая механика?",
            }
        },
    )

    model: str | None = None
    input: Any | None = None
    prompt: str | None = None
    text: str | None = None
    message: dict[str, Any] | None = None
    messages: list[dict[str, Any]] | None = None


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class OllamaTextResponseModel(BaseModel):
    """
    Параметры:
    - Поля модели ответа Ollama-style text completion.

    Что делает:
    - Описывает единый формат ответа для `/api/chat` и `/api/generate`.

    Выходные данные:
    - Экземпляр Pydantic-модели для валидации и OpenAPI.
    """
    model_config = ConfigDict(extra="allow")

    model: str
    created_at: str
    response: str
    done: bool
    done_reason: str
    total_duration: int
    load_duration: int
    prompt_eval_count: int
    prompt_eval_duration: int
    eval_count: int
    eval_duration: int
    logprobs: dict[str, Any] | None = None
    # OpenAI-compatible fields
    object: str | None = None
    choices: list[dict[str, Any]] | None = None
    usage: dict[str, Any] | None = None
    # Ollama /api/chat message field
    message: dict[str, Any] | None = None


class EmbedResponseModel(BaseModel):
    """
    Параметры:
    - Поля ответа embedding-запроса.

    Что делает:
    - Описывает формат ответа `/api/embed` с одиночным и пакетным embedding.

    Выходные данные:
    - Экземпляр Pydantic-модели для валидации и OpenAPI.
    """
    model: str
    embedding: list[float]
    embeddings: list[list[float]]
    total_duration: int
    load_duration: int
    prompt_eval_count: int


class ModelStatusItem(BaseModel):
    """
    Параметры:
    - Поля описания одной модели из каталога liteProxy.

    Что делает:
    - Представляет конфигурацию и capability-флаги модели для `/api/models`.

    Выходные данные:
    - Экземпляр Pydantic-модели для валидации и OpenAPI.
    """
    id: int = 0
    model: str
    model_vllm: str
    type: str
    modality: str = "llm"
    vision_supported: bool = False
    audio_supported: bool = False
    base_url: str
    max_context_tokens: int
    default_max_tokens: int
    min_context_headroom: int
    stream_supported: bool = False
    reasoning_supported: bool = False
    status: str = "available"
    detail: str = ""


# ---------------------------------------------------------------------------
# Rerank / Score response models
# ---------------------------------------------------------------------------


class TopLogprob(BaseModel):
    token: str
    logprob: float
    bytes: list[int] | None = None


class LogprobToken(BaseModel):
    token: str
    logprob: float
    bytes: list[int] | None = None
    top_logprobs: list[TopLogprob] | None = None


class RerankLogprobs(BaseModel):
    content: list[LogprobToken] | None = None


class RerankDocument(BaseModel):
    text: str | None = None


class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: RerankDocument | None = None
    logprobs: RerankLogprobs | None = None


class RerankUsage(BaseModel):
    total_tokens: int
    prompt_tokens: int | None = None


class RerankResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str | None = None
    model: str
    results: list[RerankResult]
    usage: RerankUsage | None = None


class ScoreResult(BaseModel):
    index: int
    score: float
    logprobs: RerankLogprobs | None = None


class ScoreResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    scores: list[ScoreResult]
    usage: RerankUsage | None = None


class RerankRequestModel(BaseModel):
    """
    Параметры:
    - Поля входного rerank-запроса (model, query, documents, top_n).

    Что делает:
    - Валидирует тело запросов rerank-endpoint'ов.

    Выходные данные:
    - Экземпляр Pydantic-модели с валидированными данными.
    """
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {
                "model": "qwen3-reranker-4b",
                "query": "Что такое квантовая запутанность?",
                "documents": [
                    "Квантовая запутанность — явление, при котором состояния двух частиц взаимозависимы.",
                    "Фотосинтез — процесс преобразования световой энергии в химическую у растений.",
                    "Принцип суперпозиции описывает нахождение квантовой системы в нескольких состояниях одновременно.",
                ],
                "top_n": 2,
            }
        },
    )
    model: str | None = None
    query: Any
    documents: list[Any] = Field(min_length=1)
    top_n: int | None = Field(default=None, ge=1)
    logprobs: bool | None = None
    top_logprobs: int | None = Field(default=None, ge=0, le=20)


class ScoreRequestModel(BaseModel):
    """
    Параметры:
    - Поля входного score-запроса в поддерживаемых форматах.

    Что делает:
    - Валидирует тело `/api/reranker/score` перед дополнительной бизнес-проверкой.

    Выходные данные:
    - Экземпляр Pydantic-модели с валидированными данными.
    """
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {
                "model": "qwen3-reranker-4b",
                "text_1": "Что такое квантовая запутанность?",
                "text_2": "Квантовая запутанность — явление, при котором состояния двух частиц взаимозависимы.",
            }
        },
    )
    model: str | None = None
    text_1: Any | None = None
    text_2: Any | None = None
    queries: Any | None = None
    documents: Any | None = None
    items: Any | None = None
    data_1: Any | None = None
    data_2: Any | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = Field(default=None, ge=0, le=20)


class ModelsResponse(BaseModel):
    """
    Параметры:
    - models: список `ModelStatusItem`.

    Что делает:
    - Описывает тело ответа `/api/models`.

    Выходные данные:
    - Экземпляр Pydantic-модели для валидации и OpenAPI.
    """
    models: list[ModelStatusItem]


class TagsResponse(BaseModel):
    """
    Параметры:
    - models: список моделей в Ollama-compatible tags формате.

    Что делает:
    - Описывает тело ответа `/api/tags`.

    Выходные данные:
    - Экземпляр Pydantic-модели для валидации и OpenAPI.
    """
    models: list[dict[str, Any]]
