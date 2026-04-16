# liteProxy - Подробный RU гайд (для junior)

Этот документ объясняет, как запустить и использовать liteProxy простыми шагами.

## 1. Что такое liteProxy
`liteProxy` это упрощенный прокси между вашим клиентом и vLLM.

Он принимает запросы в Ollama-style формате и отправляет их в локальные vLLM сервисы.

Что важно:
- нет базы данных
- нет сервиса статистики
- нет `/api/chat-ui`
- `/api/chat` работает только без stream и без reasoning

## 2. Какие endpoint доступны
Основные:
- `POST /api/chat`
- `POST /api/generate`
- `POST /api/embed`
- `GET /api/models`
- `GET /api/tags`

Совместимость с reranker API:
- `POST /api/reranker/rerank/v1`
- `POST /api/reranker/rerank/v2`
- `POST /api/reranker/score`

## 3. Как прокси выбирает модель
Прокси читает модели из файла `models.json`.

Для каждой модели там указан тип:
- `chat`
- `embeddings`
- `reranker`

Если в запросе `model` не передан, используется дефолт из `.env`:
- `DEFAULT_CHAT_MODEL`
- `DEFAULT_EMBED_MODEL`
- `DEFAULT_RERANK_MODEL`

## 4. Что означают основные параметры модели
В `models.json`:
- `max_context_tokens`: максимум контекста модели (вход + выход)
- `max_tokens`: дефолтный размер выхода, если пользователь не указал `max_tokens` в запросе
- `min_context_headroom`: запас токенов, который прокси оставляет свободным

Итоговый лимит ответа вычисляется через бюджет контекста.

## 5. Дефолтные параметры генерации
Если клиент НЕ передал параметры, прокси подставит:

Для `chat` и `generate`:
- `temperature=0.7`
- `top_p=0.8`
- `top_k=20`
- `min_p=0.0`
- `presence_penalty=1.5`
- `repetition_penalty=1.0`

Для `rerank` и `score`:
- `temperature=0`

Для `embed`:
- температура не используется

## 6. Текущая локальная конфигурация моделей
Сейчас настроено так:
- Embeddings: `lainlives/Qwen3-Embedding-4B-bnb-4bit` -> `http://127.0.0.1:8011/v1`
- Reranker: `Qwen/Qwen3-Reranker-0.6B` -> `http://127.0.0.1:8012/v1`
- Chat: `cyankiwi/Qwen3.5-9B-AWQ-4bit` -> `http://127.0.0.1:8013/v1`

Порт прокси:
- `11435`

## 7. Как запустить локально (пошагово)
1. Перейдите в папку проекта `liteProxy`.
2. Установите зависимости.
3. Запустите сервис.

Команды:
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 11435
```

После запуска откройте:
- `http://127.0.0.1:11435/docs`

## 8. Быстрая проверка, что все работает
### 8.1 Проверка списка моделей
```bash
curl -s http://127.0.0.1:11435/api/models | jq
```

### 8.2 Проверка chat
```bash
curl -s http://127.0.0.1:11435/api/chat \
  -H 'content-type: application/json' \
  -d '{
    "model": "cyankiwi/Qwen3.5-9B-AWQ-4bit",
    "messages": [{"role": "user", "content": "Привет! Коротко представься."}]
  }' | jq
```

### 8.3 Проверка embed
```bash
curl -s http://127.0.0.1:11435/api/embed \
  -H 'content-type: application/json' \
  -d '{
    "model": "lainlives/Qwen3-Embedding-4B-bnb-4bit",
    "input": "Что такое OAuth2?"
  }' | jq
```

### 8.4 Проверка rerank
```bash
curl -s http://127.0.0.1:11435/api/reranker/rerank/v1 \
  -H 'content-type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-Reranker-0.6B",
    "query": "как запустить docker",
    "documents": [
      "Установите docker engine и запустите docker compose up",
      "Погода сегодня солнечная"
    ],
    "top_n": 1
  }' | jq
```

## 9. Частые проблемы и как решать
1. `connection error` или 502:
- значит прокси не достучался до vLLM
- проверьте, что сервисы на портах 8011/8012/8013 реально запущены

2. Пустой список или неправильная модель:
- проверьте `models.json`
- проверьте, что `LITE_MODEL_CONFIG_FILE=models.json` в `.env`

3. Ответ слишком короткий:
- увеличьте `max_tokens` в запросе
- или увеличьте `max_tokens` для модели в `models.json`

4. Ошибка по лимиту контекста:
- сократите входной текст
- проверьте `max_context_tokens` и `min_context_headroom`

## 10. Что важно помнить
- Этот сервис специально упрощен для локального использования.
- Не добавляйте сюда тяжелую логику статистики/БД, если цель остаться в lite-режиме.
- Сначала проверяйте `/api/models`, потом уже chat/embed/rerank.
