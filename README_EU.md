# liteProxy - EU/EN Guide

A lightweight Ollama-style proxy for local vLLM services.

## What This Proxy Does
- Routes requests to local vLLM endpoints by model type.
- Keeps API surface small and practical.
- Uses file-based model config (`models.json`).
- Guards vLLM from burst overload via per-type concurrency limiting.

## What Is Intentionally Removed
- No database.
- No proxy statistics service.
- No `/api/chat-ui` endpoint.
- `/api/chat` is always non-stream and non-thinking.

## Endpoints
- `POST /api/chat`
- `POST /api/generate`
- `POST /api/embed`
- `POST /api/reranker/rerank/v1`
- `POST /api/reranker/rerank/v2`
- `POST /api/reranker/score`
- `GET /api/models`
- `GET /api/tags`
- `GET /api/queue/status`

## Runtime Defaults
If request does not provide sampling params:
- Chat/Generate (instruct, non-thinking):
  - `temperature=0.7`
  - `top_p=0.8`
  - `top_k=20`
  - `min_p=0.0`
  - `presence_penalty=1.5`
  - `repetition_penalty=1.0`
- Rerank/Score:
  - `temperature=0`
- Embeddings:
  - no temperature parameter

## Configuration
Main files:
- `.env`
- `models.json`

Important env vars:
| Variable | Default | Description |
|---|---|---|
| `VLLM_BASE_URL` | `http://localhost:8010/v1` | Fallback upstream base URL |
| `LITE_MODEL_CONFIG_FILE` | `models.json` | Path to model routing config |
| `DEFAULT_CHAT_MODEL` | `lite-chat` | Default model when `model` not in request |
| `DEFAULT_EMBED_MODEL` | `lite-embed` | — |
| `DEFAULT_RERANK_MODEL` | `lite-rerank` | — |
| `MAX_CONCURRENT_PER_TYPE` | `8` | Max simultaneous upstream requests per model type |
| `REDIS_URL` | _(empty)_ | Optional Redis for cross-instance queue metrics |

Model list is loaded from `models.json`.
Each model supports per-model:
- `max_context_tokens`
- `max_tokens`
- `min_context_headroom`

## Concurrency Limiter
Each model type (`chat`, `embeddings`, `reranker`) has an independent semaphore.

- `MAX_CONCURRENT_PER_TYPE=8` → 8 simultaneous chat requests, 8 embed, 8 rerank.
- Requests beyond the cap **queue in FIFO order** — callers get their response, just later.
- Monitor live queue depth: `GET /api/queue/status`

```json
{
  "chat":       { "max_concurrent": 8, "active": 3, "waiting": 1, "free_slots": 5 },
  "embeddings": { "max_concurrent": 8, "active": 0, "waiting": 0, "free_slots": 8 },
  "reranker":   { "max_concurrent": 8, "active": 1, "waiting": 0, "free_slots": 7 }
}
```

## Current Local Model Setup
- Embeddings: `lainlives/Qwen3-Embedding-4B-bnb-4bit` → `http://127.0.0.1:8011/v1`
- Reranker: `Qwen/Qwen3-Reranker-0.6B` → `http://127.0.0.1:8012/v1`
- Chat: `cyankiwi/Qwen3.5-9B-AWQ-4bit` → `http://127.0.0.1:8013/v1`

## Run Locally
```bash
# with uv (recommended)
uv sync
uvicorn app:app --host 0.0.0.0 --port 11435

# or plain pip
pip install .
uvicorn app:app --host 0.0.0.0 --port 11435
```

## Run with Docker Compose (includes Redis)
```bash
docker compose up -d --build
```

## Run Standalone Docker (no Redis)
```bash
docker build -t lite-proxy .
docker run --rm -p 11435:11435 -e VLLM_BASE_URL=http://host.docker.internal:8010/v1 lite-proxy
```

## Quick Smoke Checks
```bash
curl -s http://127.0.0.1:11435/api/models | jq
```

```bash
curl -s http://127.0.0.1:11435/api/queue/status | jq
```

```bash
curl -s http://127.0.0.1:11435/api/chat \
  -H 'content-type: application/json' \
  -d '{
    "model": "Qwen3.5-122B-A10B-FP8",
    "messages": [{"role": "user", "content": "Hello"}]
  }' | jq
```
