import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List

from fastapi import HTTPException

from settings import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_RERANK_MODEL,
    LITE_MODEL_CONFIG_FILE,
    LITE_MODEL_CONFIG_JSON,
    MAX_CONTEXT_TOKENS,
    MIN_CONTEXT_HEADROOM,
    VLLM_BASE_URL,
)

_ALLOWED_TYPES = {"chat", "embeddings", "reranker"}


def _now_iso() -> str:
    """
    Параметры:
    - отсутствуют.

    Что делает:
    - Возвращает текущее UTC-время в ISO-8601 формате.

    Выходные данные:
    - Строка времени.
    """
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _stable_digest(value: str) -> str:
    """
    Параметры:
    - value: исходная строка.

    Что делает:
    - Строит стабильный SHA-256 digest для идентификации модели в tags-ответе.

    Выходные данные:
    - Hex-строка SHA-256.
    """
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _guess_family(model_name: str) -> str:
    """
    Параметры:
    - model_name: имя модели.

    Что делает:
    - Определяет семейство модели по имени (qwen/mistral/llama/unknown).

    Выходные данные:
    - Строка семейства модели.
    """
    lower = model_name.lower()
    if "qwen" in lower:
        return "qwen"
    if "mistral" in lower or "ministral" in lower:
        return "mistral"
    if "llama" in lower:
        return "llama"
    return "unknown"


def _extract_quantization_level(model_name: str) -> str:
    """
    Параметры:
    - model_name: имя модели.

    Что делает:
    - Пытается извлечь признак квантования из имени модели.

    Выходные данные:
    - Строка уровня квантования или пустая строка.
    """
    upper = model_name.upper()
    if "Q4_K_M" in upper:
        return "Q4_K_M"
    if "Q6_K" in upper:
        return "Q6_K"
    if "Q8" in upper:
        return "Q8"
    return ""


def _normalize_aliases(raw: Dict[str, Any], model: str, backend_model: str) -> List[str]:
    """
    Параметры:
    - raw: сырая конфигурация модели.
    - model: публичное имя модели.
    - backend_model: backend-идентификатор модели.

    Что делает:
    - Формирует deduplicated список alias'ов из обязательных и дополнительных значений.

    Выходные данные:
    - Отсортированный список alias'ов.
    """
    aliases = {model, backend_model}
    extra_aliases = raw.get("aliases") or []
    if isinstance(extra_aliases, list):
        for alias in extra_aliases:
            alias_str = str(alias or "").strip()
            if alias_str:
                aliases.add(alias_str)
    return sorted(aliases)


def _coerce_int(raw: Any, fallback: int, min_value: int = 1) -> int:
    """
    Параметры:
    - raw: исходное значение.
    - fallback: значение по умолчанию.
    - min_value: минимально допустимое значение.

    Что делает:
    - Безопасно приводит значение к int и ограничивает снизу.

    Выходные данные:
    - Целое число.
    """
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        return fallback
    return max(min_value, parsed)


def _coerce_bool(raw: Any, fallback: bool = False) -> bool:
    """
    Параметры:
    - raw: исходное значение.
    - fallback: значение по умолчанию.

    Что делает:
    - Безопасно приводит значение к bool.

    Выходные данные:
    - Булево значение.
    """
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int):
        return raw != 0
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
    return fallback


def _normalize_entry(raw: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    Параметры:
    - raw: сырая запись модели из JSON-конфига.

    Что делает:
    - Валидирует и нормализует запись модели к внутреннему формату каталога.
    - Возвращает `None`, если запись некорректна для маршрутизации.

    Выходные данные:
    - Нормализованный словарь модели или `None`.
    """
    model = str(raw.get("model") or raw.get("public_model") or "").strip()
    if not model:
        return None

    backend_model = str(raw.get("backend_model") or raw.get("vllm_model") or model).strip()
    model_type = str(raw.get("type") or raw.get("model_type") or "chat").strip().lower()
    if model_type not in _ALLOWED_TYPES:
        return None

    base_url = str(raw.get("base_url") or VLLM_BASE_URL).strip().rstrip("/")
    modality = str(raw.get("modality") or "llm").strip().lower()
    vision_supported = _coerce_bool(raw.get("vision_supported"), fallback=(modality == "vl"))

    per_model_max_tokens = _coerce_int(
        raw.get("max_tokens", raw.get("default_max_tokens")),
        DEFAULT_MAX_TOKENS,
    )

    normalized = {
        "id": 0,
        "model": model,
        "model_vllm": backend_model,
        "type": model_type,
        "modality": "vl" if vision_supported else "llm",
        "vision_supported": vision_supported,
        "audio_supported": _coerce_bool(raw.get("audio_supported"), fallback=False),
        "base_url": base_url,
        "max_context_tokens": _coerce_int(raw.get("max_context_tokens"), MAX_CONTEXT_TOKENS),
        "default_max_tokens": per_model_max_tokens,
        "min_context_headroom": _coerce_int(raw.get("min_context_headroom"), MIN_CONTEXT_HEADROOM, min_value=0),
        "stream_supported": _coerce_bool(raw.get("stream_supported"), fallback=False),
        "reasoning_supported": _coerce_bool(raw.get("reasoning_supported"), fallback=False),
        "status": "available",
        "detail": "",
    }
    normalized["aliases"] = _normalize_aliases(raw, model, backend_model)
    return normalized


def _default_entry(model: str, model_type: str) -> Dict[str, Any]:
    """
    Параметры:
    - model: имя модели.
    - model_type: тип модели (`chat`, `embeddings`, `reranker`).

    Что делает:
    - Создаёт дефолтную запись для отсутствующего типа модели.

    Выходные данные:
    - Словарь конфигурации модели.
    """
    return {
        "id": 0,
        "model": model,
        "model_vllm": model,
        "type": model_type,
        "modality": "llm",
        "vision_supported": False,
        "audio_supported": False,
        "base_url": VLLM_BASE_URL,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "default_max_tokens": DEFAULT_MAX_TOKENS,
        "min_context_headroom": MIN_CONTEXT_HEADROOM,
        "stream_supported": False,
        "reasoning_supported": False,
        "aliases": [model],
        "status": "available",
        "detail": "",
    }


def _parse_models_json(raw_json: str, source_name: str) -> Iterable[Dict[str, Any]]:
    """
    Параметры:
    - raw_json: JSON-строка с моделями.
    - source_name: имя источника (для ошибок).

    Что делает:
    - Парсит конфиг моделей в формате массива или объекта с ключом `models`.

    Выходные данные:
    - Итерируемая коллекция словарей моделей.
    """
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{source_name} is not valid JSON: {str(exc)}")

    if isinstance(parsed, dict) and isinstance(parsed.get("models"), list):
        return [item for item in parsed["models"] if isinstance(item, dict)]
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    raise RuntimeError(f"{source_name} must be a JSON array or an object with key 'models'")


def _parse_raw_models() -> Iterable[Dict[str, Any]]:
    """
    Параметры:
    - отсутствуют.

    Что делает:
    - Загружает модели из `LITE_MODEL_CONFIG_FILE`.
    - Если файл недоступен, использует fallback `LITE_MODEL_CONFIG_JSON`.

    Выходные данные:
    - Итерируемая коллекция сырых записей моделей.
    """
    config_file = str(LITE_MODEL_CONFIG_FILE or "").strip()
    if config_file:
        cfg_path = Path(config_file)
        if not cfg_path.is_absolute():
            cfg_path = Path(__file__).resolve().parent.parent / cfg_path
        if cfg_path.exists():
            raw_json = cfg_path.read_text(encoding="utf-8")
            return _parse_models_json(raw_json, f"LITE_MODEL_CONFIG_FILE ({cfg_path})")

    if LITE_MODEL_CONFIG_JSON:
        return _parse_models_json(LITE_MODEL_CONFIG_JSON, "LITE_MODEL_CONFIG_JSON")

    return []


@lru_cache(maxsize=1)
def _load_models_cached() -> tuple[Dict[str, Any], ...]:
    """
    Параметры:
    - отсутствуют.

    Что делает:
    - Формирует и кеширует итоговый список моделей для маршрутизации.
    - Гарантирует наличие записей по типам chat/embeddings/reranker.

    Выходные данные:
    - Кортеж нормализованных конфигураций моделей.
    """
    models: List[Dict[str, Any]] = []

    for raw in _parse_raw_models():
        normalized = _normalize_entry(raw)
        if normalized is not None:
            models.append(normalized)

    existing_types = {item["type"] for item in models}
    if "chat" not in existing_types:
        models.append(_default_entry(DEFAULT_CHAT_MODEL, "chat"))
    if "embeddings" not in existing_types:
        models.append(_default_entry(DEFAULT_EMBED_MODEL, "embeddings"))
    if "reranker" not in existing_types:
        models.append(_default_entry(DEFAULT_RERANK_MODEL, "reranker"))

    for index, item in enumerate(models, start=1):
        item["id"] = index

    return tuple(models)


def get_models_snapshot() -> List[Dict[str, Any]]:
    """
    Параметры:
    - отсутствуют.

    Что делает:
    - Возвращает копию текущего списка моделей из кеша каталога.

    Выходные данные:
    - Список словарей моделей.
    """
    return [deepcopy(item) for item in _load_models_cached()]


def refresh_model_catalog() -> None:
    """
    Параметры:
    - отсутствуют.

    Что делает:
    - Сбрасывает кеш каталога моделей, чтобы при следующем запросе перечитать конфиг.

    Выходные данные:
    - отсутствуют.
    """
    _load_models_cached.cache_clear()


def resolve_target(requested_model: str | None, expected_types: set[str]) -> Dict[str, Any]:
    """
    Параметры:
    - requested_model: имя модели из запроса (может быть `None`).
    - expected_types: допустимые типы модели для конкретного endpoint.

    Что делает:
    - Выбирает целевую модель из каталога по имени/alias.
    - Проверяет тип модели и формирует понятные HTTP-ошибки для API.

    Выходные данные:
    - Словарь конфигурации выбранной модели.
    """
    models = get_models_snapshot()
    allowed = [item for item in models if item.get("type") in expected_types]
    if not allowed:
        allowed_label = ", ".join(sorted(expected_types))
        raise HTTPException(status_code=503, detail=f"no models configured for type: {allowed_label}")

    by_alias: Dict[str, Dict[str, Any]] = {}
    for item in allowed:
        by_alias[str(item.get("model", ""))] = item
        by_alias[str(item.get("model_vllm", ""))] = item
        for alias in item.get("aliases") or []:
            by_alias[str(alias)] = item

    resolved = None
    if requested_model is not None and str(requested_model).strip():
        resolved = by_alias.get(str(requested_model).strip())
        if resolved is None:
            available = sorted({str(item.get("model")) for item in allowed})
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "unknown_model",
                    "model": requested_model,
                    "available": available,
                },
            )

    if resolved is None:
        resolved = allowed[0]

    return resolved


def to_ollama_tag_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Параметры:
    - item: словарь внутренней конфигурации модели.

    Что делает:
    - Преобразует внутреннюю запись модели в Ollama-compatible tags-объект.

    Выходные данные:
    - Словарь модели в формате `/api/tags`.
    """
    model_name = str(item.get("model") or "").strip()
    backend_name = str(item.get("model_vllm") or model_name).strip()
    model_type = str(item.get("type") or "")

    quantization = _extract_quantization_level(backend_name or model_name)
    is_gguf = bool(quantization)
    modality = str(item.get("modality") or "llm")
    if modality == "vl":
        fmt = "vl"
    else:
        fmt = "gguf" if is_gguf else "vllm"

    return {
        "name": model_name,
        "model": model_name,
        "modified_at": _now_iso(),
        "size": 0,
        "digest": _stable_digest(f"{model_name}|{backend_name}|{model_type}"),
        "details": {
            "parent_model": backend_name,
            "format": fmt,
            "family": _guess_family(backend_name or model_name),
            "families": [_guess_family(backend_name or model_name)],
            "parameter_size": "",
            "quantization_level": quantization if is_gguf else "",
            "modality": modality,
            "vision_supported": bool(item.get("vision_supported", False)),
            "audio_supported": bool(item.get("audio_supported", False)),
        },
    }
