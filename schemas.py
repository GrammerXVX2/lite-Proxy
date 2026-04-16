from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class OllamaTextResponseModel(BaseModel):
    """
    Параметры:
    - Поля модели ответа Ollama-style text completion.

    Что делает:
    - Описывает единый формат ответа для `/api/chat` и `/api/generate`.

    Выходные данные:
    - Экземпляр Pydantic-модели для валидации и OpenAPI.
    """
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
    embedding: List[float]
    embeddings: List[List[float]]
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


class RerankRequestModel(BaseModel):
    """
    Параметры:
    - Поля входного rerank-запроса (model, query, documents, top_n).

    Что делает:
    - Валидирует тело запросов rerank-endpoint'ов.

    Выходные данные:
    - Экземпляр Pydantic-модели с валидированными данными.
    """
    model_config = ConfigDict(extra="allow")
    model: Optional[str] = None
    query: Any
    documents: List[Any] = Field(min_length=1)
    top_n: Optional[int] = Field(default=None, ge=1)


class ScoreRequestModel(BaseModel):
    """
    Параметры:
    - Поля входного score-запроса в поддерживаемых форматах.

    Что делает:
    - Валидирует тело `/api/reranker/score` перед дополнительной бизнес-проверкой.

    Выходные данные:
    - Экземпляр Pydantic-модели с валидированными данными.
    """
    model_config = ConfigDict(extra="allow")
    model: Optional[str] = None
    text_1: Optional[Any] = None
    text_2: Optional[Any] = None
    queries: Optional[Any] = None
    documents: Optional[Any] = None
    items: Optional[Any] = None
    data_1: Optional[Any] = None
    data_2: Optional[Any] = None


class ModelsResponse(BaseModel):
    """
    Параметры:
    - models: список `ModelStatusItem`.

    Что делает:
    - Описывает тело ответа `/api/models`.

    Выходные данные:
    - Экземпляр Pydantic-модели для валидации и OpenAPI.
    """
    models: List[ModelStatusItem]


class TagsResponse(BaseModel):
    """
    Параметры:
    - models: список моделей в Ollama-compatible tags формате.

    Что делает:
    - Описывает тело ответа `/api/tags`.

    Выходные данные:
    - Экземпляр Pydantic-модели для валидации и OpenAPI.
    """
    models: List[Dict[str, Any]]
