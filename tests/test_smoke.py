"""
Smoke tests — fast, no LLM calls.
Verify: /, /api/models, /api/tags, /api/queue/status
"""
import pytest
import httpx
from conftest import CHAT_MODEL, EMBED_MODEL, RERANK_MODEL


class TestHealth:
    def test_root_ok(self, client: httpx.Client):
        r = client.get("/")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "docs" in body

    def test_docs_reachable(self, client: httpx.Client):
        r = client.get("/docs")
        assert r.status_code == 200

    def test_openapi_schema(self, client: httpx.Client):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        schema = r.json()
        paths = schema.get("paths", {})
        assert "/api/chat"            in paths, "missing /api/chat in schema"
        assert "/api/generate"        in paths, "missing /api/generate in schema"
        assert "/api/embed"           in paths, "missing /api/embed in schema"
        assert "/api/reranker/rerank/v1" in paths, "missing /api/reranker/rerank/v1 in schema"
        assert "/api/queue/status"    in paths, "missing /api/queue/status in schema"


class TestModelEndpoints:
    def test_api_models_structure(self, client: httpx.Client):
        r = client.get("/api/models")
        assert r.status_code == 200
        body = r.json()
        assert "models" in body
        models = body["models"]
        assert isinstance(models, list)
        assert len(models) >= 3, f"expected >=3 models, got {len(models)}"

    def test_api_models_types(self, client: httpx.Client):
        r = client.get("/api/models")
        models = r.json()["models"]
        types = {m["type"] for m in models}
        assert "chat"       in types, "no chat model configured"
        assert "embeddings" in types, "no embeddings model configured"
        assert "reranker"   in types, "no reranker model configured"

    def test_api_models_chat_has_vision(self, client: httpx.Client):
        r = client.get("/api/models")
        chat_models = [m for m in r.json()["models"] if m["type"] == "chat"]
        assert any(m.get("vision_supported") for m in chat_models), \
            "chat model should have vision_supported=true"

    def test_api_tags_structure(self, client: httpx.Client):
        r = client.get("/api/tags")
        assert r.status_code == 200
        body = r.json()
        assert "models" in body
        tags = body["models"]
        assert isinstance(tags, list)
        assert len(tags) >= 1

    def test_api_tags_fields(self, client: httpx.Client):
        r = client.get("/api/tags")
        for tag in r.json()["models"]:
            assert "name"       in tag, f"missing 'name' in tag {tag}"
            assert "model"      in tag, f"missing 'model' in tag {tag}"
            assert "modified_at" in tag

    def test_queue_status_structure(self, client: httpx.Client):
        r = client.get("/api/queue/status")
        assert r.status_code == 200
        body = r.json()
        for model_type in ("chat", "embeddings", "reranker"):
            assert model_type in body, f"missing key '{model_type}' in queue status"
            info = body[model_type]
            for field in ("max_concurrent", "active", "waiting", "free_slots"):
                assert field in info, f"missing '{field}' in {model_type} queue info"

    def test_queue_status_values(self, client: httpx.Client):
        r = client.get("/api/queue/status")
        body = r.json()
        for model_type, info in body.items():
            assert info["max_concurrent"] > 0
            assert info["active"]    >= 0
            assert info["waiting"]   >= 0
            assert info["free_slots"] >= 0


class TestBadRequests:
    def test_chat_unknown_model_404(self, client: httpx.Client):
        r = client.post("/api/chat", json={
            "model": "nonexistent-model-xyz",
            "messages": [{"role": "user", "content": "hi"}]
        })
        assert r.status_code in (400, 404, 503)

    def test_chat_stream_disabled(self, client: httpx.Client):
        r = client.post("/api/chat", json={
            "model": CHAT_MODEL,
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}]
        })
        assert r.status_code == 400

    def test_generate_stream_disabled(self, client: httpx.Client):
        r = client.post("/api/generate", json={
            "model": CHAT_MODEL,
            "stream": True,
            "prompt": "hi"
        })
        assert r.status_code == 400

    def test_embed_unknown_model(self, client: httpx.Client):
        r = client.post("/api/embed", json={
            "model": "nonexistent-embed-xyz",
            "input": "hello"
        })
        assert r.status_code in (400, 404, 503)
