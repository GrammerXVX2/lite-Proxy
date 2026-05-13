import copy
import re
import time
from datetime import datetime, timezone
from typing import Any, TypedDict

from settings import settings


def inline_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve $defs $refs inline so Swagger's resolver can find them in openapi_extra."""
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def resolve(obj: Any) -> Any:
        if isinstance(obj, dict):
            if tuple(obj) == ("$ref",):
                ref: str = obj["$ref"]
                if ref.startswith("#/$defs/"):
                    name = ref[len("#/$defs/"):]
                    if name in defs:
                        return resolve(copy.deepcopy(defs[name]))
            return {k: resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [resolve(item) for item in obj]
        return obj

    return resolve(schema)


class TokenBudget(TypedDict):
    resolved_max_tokens: int


def now_iso() -> str:
    """
    Параметры:
    - отсутствуют.

    Что делает:
    - Возвращает текущее UTC-время в ISO-8601 формате с суффиксом `Z`.

    Выходные данные:
    - Строка времени в формате ISO.
    """
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ns() -> int:
    """
    Параметры:
    - отсутствуют.

    Что делает:
    - Возвращает монотонное время в наносекундах для измерения длительностей.

    Выходные данные:
    - Целое число наносекунд.
    """
    return time.perf_counter_ns()


def estimate_input_tokens_from_text(text: str) -> int:
    """
    Параметры:
    - text: входной текст.

    Что делает:
    - Оценивает число токенов по простой эвристике (примерно 1 токен на 4 символа).

    Выходные данные:
    - Оценка количества токенов (int).
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_chat_input_tokens(messages: list[dict[str, Any]]) -> int:
    """
    Параметры:
    - messages: список chat-сообщений в Ollama/OpenAI-like формате.

    Что делает:
    - Считает приблизительный размер входа в токенах с учётом текста и служебного оверхеда.

    Выходные данные:
    - Оценка количества входных токенов (int).
    """
    total = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            total += estimate_input_tokens_from_text(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text" and part.get("text") is not None:
                    total += estimate_input_tokens_from_text(str(part.get("text")))
        total += 8
    return total + 16


def analyze_max_tokens_budget(
    body: dict[str, Any],
    estimated_input_tokens: int = 0,
    max_context_tokens: int | None = None,
    min_context_headroom: int | None = None,
    default_max_tokens: int | None = None,
) -> TokenBudget:
    """
    Параметры:
    - body: тело запроса клиента.
    - estimated_input_tokens: оценка входных токенов.
    - max_context_tokens: лимит контекста модели.
    - min_context_headroom: резерв токенов, оставляемый свободным.
    - default_max_tokens: дефолтный размер генерации для модели.

        Что делает:
        - Вычисляет финальный `max_tokens` с учётом запроса клиента и доступного бюджета контекста.
        - Поддерживает Ollama-style `num_ctx` (в корне или в `options`) как желаемый контекст
            в пределах лимита модели.

    Выходные данные:
    - Словарь с ключом `resolved_max_tokens`.
    """
    _opts = body.get("options")
    options: dict[str, Any] = _opts if isinstance(_opts, dict) else {}

    requested_ctx_raw = body.get("num_ctx")
    if requested_ctx_raw is None:
        requested_ctx_raw = options.get("num_ctx")

    requested_context_value = None
    if requested_ctx_raw is not None:
        try:
            requested_context_value = max(1, int(requested_ctx_raw))
        except (TypeError, ValueError):
            requested_context_value = None

    requested_raw = body.get("max_tokens")
    if requested_raw is None:
        requested_raw = options.get("num_predict")

    requested_value = None
    if requested_raw is not None:
        try:
            requested_value = max(1, int(requested_raw))
        except (TypeError, ValueError):
            requested_value = None

    model_context_limit = max(1, int(max_context_tokens or settings.max_context_tokens))
    if requested_context_value is None:
        resolved_context = model_context_limit
    else:
        resolved_context = min(requested_context_value, model_context_limit)

    resolved_headroom = max(0, int(min_context_headroom or settings.min_context_headroom))
    resolved_default = max(1, int(default_max_tokens or settings.default_max_tokens))

    available_output_tokens = max(1, resolved_context - max(0, estimated_input_tokens) - resolved_headroom)
    hard_cap = available_output_tokens

    if requested_value is None:
        resolved = min(resolved_default, hard_cap)
    else:
        resolved = min(requested_value, hard_cap)

    return {
        "resolved_max_tokens": int(max(1, resolved)),
    }


def extract_chat_text(data: dict[str, Any]) -> str:
    """
    Параметры:
    - data: JSON-ответ upstream chat/completions.

    Что делает:
    - Извлекает текст ответа из `message.content`.
    - Если контент пуст, пробует fallback в `message.reasoning`.

    Выходные данные:
    - Строка с текстом ответа (или пустая строка).
    """
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content")
    if isinstance(content, str) and content:
        return content
    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        return reasoning
    return ""


def extract_finish_reason(data: dict[str, Any]) -> str:
    """
    Параметры:
    - data: JSON-ответ upstream chat/completions.

    Что делает:
    - Извлекает `finish_reason` из первого choice.
    - Использует `stop`, если причина завершения отсутствует.

    Выходные данные:
    - Строка причины завершения.
    """
    choice = (data.get("choices") or [{}])[0]
    reason = choice.get("finish_reason")
    if isinstance(reason, str) and reason:
        return reason
    return "stop"


def strip_reasoning_artifacts(text: str) -> str:
    """
    Параметры:
    - text: исходный текст ответа модели.

    Что делает:
    - Удаляет блоки `<think>...</think>` и лишние пробелы по краям.

    Выходные данные:
    - Очищенная строка.
    """
    if not text:
        return text
    return re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()


def _coerce_non_negative_int(value: Any, default: int = 0) -> int:
    """
    Параметры:
    - value: значение для преобразования к int.
    - default: значение по умолчанию при ошибке парсинга.

    Что делает:
    - Безопасно приводит значение к неотрицательному целому числу.

    Выходные данные:
    - Неотрицательное целое число.
    """
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, parsed)


def ollama_response(
    model: str,
    content: str,
    start_ns: int,
    done_reason: str = "stop",
    usage: dict[str, Any] | None = None,
    logprobs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Параметры:
    - model: публичное имя модели.
    - content: текст ответа.
    - start_ns: timestamp старта обработки в наносекундах.
    - done_reason: причина завершения генерации.
    - usage: usage-объект upstream (`prompt_tokens`, `completion_tokens`).
    - logprobs: logprobs-объект из choices[0].logprobs (если запрошены).

    Что делает:
    - Формирует ответ в Ollama-style формате.
    - Маппит usage из vLLM в `prompt_eval_count` и `eval_count`.
    - Включает logprobs в ответ, если они были запрошены.

    Выходные данные:
    - Словарь ответа в формате Ollama.
    """
    total_ns = max(0, ns() - start_ns)
    usage_obj = usage if isinstance(usage, dict) else {}
    prompt_eval_count = _coerce_non_negative_int(usage_obj.get("prompt_tokens"), default=0)
    eval_count = _coerce_non_negative_int(usage_obj.get("completion_tokens"), default=0)

    choice: dict[str, Any] = {
        "index": 0,
        "message": {"role": "assistant", "content": content},
        "finish_reason": done_reason,
    }
    if logprobs is not None:
        choice["logprobs"] = logprobs

    result: dict[str, Any] = {
        # OpenAI-compatible fields (for n8n and other OpenAI-style clients)
        "object": "chat.completion",
        "choices": [choice],
        "usage": {
            "prompt_tokens": prompt_eval_count,
            "completion_tokens": eval_count,
            "total_tokens": prompt_eval_count + eval_count,
        },
        # Ollama-compatible fields
        "model": model,
        "created_at": now_iso(),
        "message": {"role": "assistant", "content": content},
        "response": content,
        "done": True,
        "done_reason": done_reason,
        "total_duration": total_ns,
        "load_duration": 0,
        "prompt_eval_count": prompt_eval_count,
        "prompt_eval_duration": 0,
        "eval_count": eval_count,
        "eval_duration": 0,
    }
    if logprobs is not None:
        result["logprobs"] = logprobs
    return result
