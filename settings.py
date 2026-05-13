import httpx
from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Upstream vLLM base URL (trailing slash is stripped automatically).
    vllm_base_url: str = Field(default="http://localhost:8010/v1")

    # Model config source: JSON file path or inline JSON string.
    lite_model_config_file: str = Field(default="models.json")
    lite_model_config_json: str = Field(default="")

    # Public default model names; accept legacy env-var aliases.
    default_chat_model: str = Field(
        default="lite-chat",
        validation_alias=AliasChoices("DEFAULT_CHAT_MODEL", "MODEL_CHAT"),
    )
    default_embed_model: str = Field(
        default="lite-embed",
        validation_alias=AliasChoices("DEFAULT_EMBED_MODEL", "MODEL_EMBED"),
    )
    default_rerank_model: str = Field(
        default="lite-rerank",
        validation_alias=AliasChoices("DEFAULT_RERANK_MODEL", "MODEL_RERANK"),
    )

    # Token budget defaults.
    default_max_tokens: int = Field(default=1024)
    max_context_tokens: int = Field(default=8192)
    min_context_headroom: int = Field(default=128)

    # HTTP client tuning.
    upstream_timeout_seconds: float = Field(default=20.0)
    upstream_max_connections: int = Field(default=200)
    upstream_max_keepalive_connections: int = Field(default=50)
    upstream_keepalive_expiry_seconds: float = Field(default=30.0)

    # Retry policy.
    upstream_retry_attempts: int = Field(default=2)
    upstream_retry_base_delay_seconds: float = Field(default=0.2)
    upstream_retry_jitter_seconds: float = Field(default=0.1)
    # Stored as a comma-separated string; parsed by `retry_status_codes` property.
    upstream_retry_status_codes: str = Field(default="502,503,504")

    # Concurrency limiter — caps simultaneous in-flight upstream requests
    # per model type (chat / embeddings / reranker) independently.
    # Excess requests queue in the event-loop (FIFO) rather than being rejected.
    max_concurrent_per_type: int = Field(default=8)

    # Optional Redis URL for cross-instance queue metrics.
    # Leave empty to run limiter without Redis (local semaphore only).
    redis_url: str = Field(default="")

    @field_validator("vllm_base_url", mode="after")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.strip().rstrip("/")

    @property
    def retry_status_codes(self) -> set[int]:
        codes: set[int] = set()
        for part in self.upstream_retry_status_codes.split(","):
            part = part.strip()
            try:
                code = int(part)
                if 100 <= code <= 599:
                    codes.add(code)
            except ValueError:
                pass
        return codes or {502, 503, 504}

    @property
    def http_timeout(self) -> httpx.Timeout:
        return httpx.Timeout(timeout=self.upstream_timeout_seconds)

    @property
    def http_limits(self) -> httpx.Limits:
        return httpx.Limits(
            max_connections=self.upstream_max_connections,
            max_keepalive_connections=self.upstream_max_keepalive_connections,
            keepalive_expiry=self.upstream_keepalive_expiry_seconds,
        )


settings = Settings()

