import json
from typing import Any, Dict
from urllib.parse import parse_qs

from fastapi import Request


async def read_request_body_as_dict(request: Request) -> Dict[str, Any]:
    """
    Параметры:
    - request: входящий HTTP-запрос FastAPI.

    Что делает:
    - Толерантно парсит тело запроса для разных content-type:
      `application/json`, `x-www-form-urlencoded`, `multipart/form-data`, raw-text.
    - Нормализует результат к словарю, чтобы endpoint'ы работали с единым форматом.

    Выходные данные:
    - Словарь с распарсенными данными запроса.
    """
    content_type = (request.headers.get("content-type") or "").lower()

    if "application/json" in content_type:
        try:
            parsed = await request.json()
        except Exception:
            parsed = {}
    elif "application/x-www-form-urlencoded" in content_type:
        raw_bytes = await request.body()
        raw = raw_bytes.decode("utf-8", errors="ignore")
        raw_stripped = raw.strip()

        if raw_stripped.startswith("{") or raw_stripped.startswith("["):
            try:
                parsed = json.loads(raw_stripped)
            except Exception:
                parsed = {}
        else:
            form_qs = parse_qs(raw, keep_blank_values=True)
            parsed = {
                key: (values[0] if isinstance(values, list) and len(values) == 1 else values)
                for key, values in form_qs.items()
            }
    elif "multipart/form-data" in content_type:
        try:
            parsed = dict(await request.form())
        except Exception:
            parsed = {}
    else:
        raw_bytes = await request.body()
        raw = raw_bytes.decode("utf-8", errors="ignore").strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {"prompt": raw, "input": raw}

    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, str):
        return {"prompt": parsed, "input": parsed}
    if isinstance(parsed, list):
        return {"input": parsed}
    return {}
