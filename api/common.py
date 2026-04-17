import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from settings import DEFAULT_MAX_TOKENS, MAX_CONTEXT_TOKENS, MIN_CONTEXT_HEADROOM


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


def estimate_chat_input_tokens(messages: List[Dict[str, Any]]) -> int:
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
    body: Dict[str, Any],
    estimated_input_tokens: int = 0,
    max_context_tokens: int | None = None,
    min_context_headroom: int | None = None,
    default_max_tokens: int | None = None,
) -> Dict[str, Any]:
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
    options = body.get("options") if isinstance(body.get("options"), dict) else {}

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

    model_context_limit = max(1, int(max_context_tokens or MAX_CONTEXT_TOKENS))
    if requested_context_value is None:
        resolved_context = model_context_limit
    else:
        resolved_context = min(requested_context_value, model_context_limit)

    resolved_headroom = max(0, int(min_context_headroom or MIN_CONTEXT_HEADROOM))
    resolved_default = max(1, int(default_max_tokens or DEFAULT_MAX_TOKENS))

    available_output_tokens = max(1, resolved_context - max(0, estimated_input_tokens) - resolved_headroom)
    hard_cap = available_output_tokens

    if requested_value is None:
        resolved = min(resolved_default, hard_cap)
    else:
        resolved = min(requested_value, hard_cap)

    return {
        "resolved_max_tokens": int(max(1, resolved)),
    }


def extract_chat_text(data: Dict[str, Any]) -> str:
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


def extract_finish_reason(data: Dict[str, Any]) -> str:
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
    usage: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Параметры:
    - model: публичное имя модели.
    - content: текст ответа.
    - start_ns: timestamp старта обработки в наносекундах.
    - done_reason: причина завершения генерации.
    - usage: usage-объект upstream (`prompt_tokens`, `completion_tokens`).

    Что делает:
    - Формирует ответ в Ollama-style формате.
    - Маппит usage из vLLM в `prompt_eval_count` и `eval_count`.

    Выходные данные:
    - Словарь ответа в формате Ollama.
    """
    total_ns = max(0, ns() - start_ns)
    usage_obj = usage if isinstance(usage, dict) else {}
    prompt_eval_count = _coerce_non_negative_int(usage_obj.get("prompt_tokens"), default=0)
    eval_count = _coerce_non_negative_int(usage_obj.get("completion_tokens"), default=0)

    return {
        "model": model,
        "created_at": now_iso(),
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
