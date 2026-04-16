import os

import httpx
from dotenv import load_dotenv


load_dotenv()


def _parse_retry_status_codes(raw: str) -> set[int]:
    """
    Параметры:
    - raw: строка со списком HTTP-кодов через запятую.

    Что делает:
    - Парсит строку в множество валидных HTTP-кодов (100..599).
    - Игнорирует пустые и нечисловые значения.
    - Возвращает дефолтный набор retry-кодов, если после парсинга список пуст.

    Выходные данные:
    - Множество HTTP-кодов для retry-логики.
    """
    codes: set[int] = set()
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            code = int(part)
        except ValueError:
            continue
        if 100 <= code <= 599:
            codes.add(code)
    return codes or {502, 503, 504}


VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1").strip().rstrip("/")

# Preferred source: JSON file with model routes (for example ./models.json).
LITE_MODEL_CONFIG_FILE = os.getenv("LITE_MODEL_CONFIG_FILE", "models.json").strip()

# Optional fallback: inline JSON config for model routing in lite mode.
LITE_MODEL_CONFIG_JSON = os.getenv("LITE_MODEL_CONFIG_JSON", "").strip()

DEFAULT_CHAT_MODEL = os.getenv("DEFAULT_CHAT_MODEL", "lite-chat")
DEFAULT_EMBED_MODEL = os.getenv("DEFAULT_EMBED_MODEL", "lite-embed")
DEFAULT_RERANK_MODEL = os.getenv("DEFAULT_RERANK_MODEL", "lite-rerank")

DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "1024"))
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "8192"))
MIN_CONTEXT_HEADROOM = int(os.getenv("MIN_CONTEXT_HEADROOM", "128"))

UPSTREAM_TIMEOUT_SECONDS = float(os.getenv("UPSTREAM_TIMEOUT_SECONDS", "20"))
UPSTREAM_MAX_CONNECTIONS = int(os.getenv("UPSTREAM_MAX_CONNECTIONS", "200"))
UPSTREAM_MAX_KEEPALIVE_CONNECTIONS = int(os.getenv("UPSTREAM_MAX_KEEPALIVE_CONNECTIONS", "50"))
UPSTREAM_KEEPALIVE_EXPIRY_SECONDS = float(os.getenv("UPSTREAM_KEEPALIVE_EXPIRY_SECONDS", "30"))
UPSTREAM_RETRY_ATTEMPTS = int(os.getenv("UPSTREAM_RETRY_ATTEMPTS", "2"))
UPSTREAM_RETRY_BASE_DELAY_SECONDS = float(os.getenv("UPSTREAM_RETRY_BASE_DELAY_SECONDS", "0.2"))
UPSTREAM_RETRY_JITTER_SECONDS = float(os.getenv("UPSTREAM_RETRY_JITTER_SECONDS", "0.1"))
UPSTREAM_RETRY_STATUS_CODES = _parse_retry_status_codes(
    os.getenv("UPSTREAM_RETRY_STATUS_CODES", "502,503,504")
)

UPSTREAM_HTTP_TIMEOUT = httpx.Timeout(timeout=UPSTREAM_TIMEOUT_SECONDS)
UPSTREAM_HTTP_LIMITS = httpx.Limits(
    max_connections=UPSTREAM_MAX_CONNECTIONS,
    max_keepalive_connections=UPSTREAM_MAX_KEEPALIVE_CONNECTIONS,
    keepalive_expiry=UPSTREAM_KEEPALIVE_EXPIRY_SECONDS,
)
