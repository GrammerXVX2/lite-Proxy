liteProxy

Additional guides:
- `README_EU.md` - concise English (EU) guide
- `README_RU.md` - detailed Russian guide

Lightweight proxy version for local systems:
- no database
- no proxy statistics
- no `/api/chat-ui`
- `/api/chat` has streaming and reasoning disabled

Endpoints
- POST `/api/chat`
- POST `/api/generate`
- POST `/api/embed`
- GET `/api/models`
- GET `/api/tags`

Compatibility endpoints for reranker are also included:
- POST `/api/reranker/rerank/v1`
- POST `/api/reranker/rerank/v2`
- POST `/api/reranker/score`

Configuration
1. Base upstream endpoint:
   - `VLLM_BASE_URL=http://127.0.0.1:8010/v1`
2. Model routing file:
   - `LITE_MODEL_CONFIG_FILE=models.json`
   - `models.json` supports per-model `max_tokens` (token cap is not used).

If `LITE_MODEL_CONFIG_FILE` is missing (and inline config is not set), liteProxy auto-creates default routes:
- chat: `DEFAULT_CHAT_MODEL` (default `lite-chat`)
- embeddings: `DEFAULT_EMBED_MODEL` (default `lite-embed`)
- reranker: `DEFAULT_RERANK_MODEL` (default `lite-rerank`)

Default runtime params when request does not provide them:
- chat/generate (instruct, non-thinking): `temperature=0.7`, `top_p=0.8`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0`
- rerank/score: `temperature=0`
- embeddings: no temperature parameter

Run locally
- `pip install -r requirements.txt`
- `uvicorn app:app --host 0.0.0.0 --port 11435`

Run in Docker
- `docker build -t lite-proxy .`
- `docker run --rm -p 11435:11435 -e VLLM_BASE_URL=http://host.docker.internal:8010/v1 lite-proxy`
