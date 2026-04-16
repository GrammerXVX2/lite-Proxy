# liteProxy - EU/EN Guide

A lightweight Ollama-style proxy for local vLLM services.

## What This Proxy Does
- Routes requests to local vLLM endpoints by model type.
- Keeps API surface small and practical.
- Uses file-based model config (`models.json`).

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
- `VLLM_BASE_URL` - fallback base URL
- `LITE_MODEL_CONFIG_FILE=models.json`
- `DEFAULT_CHAT_MODEL`, `DEFAULT_EMBED_MODEL`, `DEFAULT_RERANK_MODEL`

Model list is loaded from `models.json`.
Each model supports per-model:
- `max_context_tokens`
- `max_tokens`
- `min_context_headroom`

## Current Local Model Setup
- Embeddings: `lainlives/Qwen3-Embedding-4B-bnb-4bit` -> `http://127.0.0.1:8001/v1`
- Reranker: `Qwen/Qwen3-Reranker-0.6B` -> `http://127.0.0.1:8002/v1`
- Chat: `cyankiwi/Qwen3.5-9B-AWQ-4bit` -> `http://127.0.0.1:8003/v1`

## Run Locally
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 11435
```

## Run in Docker
```bash
docker build -t lite-proxy .
docker run --rm -p 11435:11435 -e VLLM_BASE_URL=http://host.docker.internal:8000/v1 lite-proxy
```

## Quick Smoke Checks
```bash
curl -s http://127.0.0.1:11435/api/models | jq
```

```bash
curl -s http://127.0.0.1:11435/api/chat \
  -H 'content-type: application/json' \
  -d '{
    "model": "cyankiwi/Qwen3.5-9B-AWQ-4bit",
    "messages": [{"role": "user", "content": "Hello"}]
  }' | jq
```
