"""
Functional (live) tests — actually call vLLM upstream.
Mark with: pytest -m live
"""
import pytest
import httpx
from conftest import CHAT_MODEL, EMBED_MODEL, RERANK_MODEL


pytestmark = pytest.mark.live


class TestChat:
    def test_basic_chat(self, client: httpx.Client):
        r = client.post("/api/chat", json={
            "model": CHAT_MODEL,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": "Ответь одним словом: столица Франции?"}]
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body.get("response"), "empty response from /api/chat"
        assert body.get("model")
        assert body.get("done") is True

    def test_chat_alias(self, client: httpx.Client):
        """Model alias should resolve to the same model."""
        r = client.post("/api/chat", json={
            "model": "chat",
            "temperature": 0.0,
            "messages": [{"role": "user", "content": "ping"}]
        })
        assert r.status_code == 200, r.text

    def test_chat_system_prompt(self, client: httpx.Client):
        r = client.post("/api/chat", json={
            "model": CHAT_MODEL,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": "Ты краткий ассистент. Отвечай одним словом."},
                {"role": "user",   "content": "Столица Германии?"}
            ]
        })
        assert r.status_code == 200, r.text
        assert r.json().get("response")

    def test_chat_with_image_url(self, client: httpx.Client):
        r = client.post("/api/chat", json={
            "model": CHAT_MODEL,
            "temperature": 0.0,
            "messages": [{
                "role": "user",
                "content": "Опиши что на картинке одним словом.",
                "images": ["https://images.unsplash.com/photo-1518717758536-85ae29035b6d?w=320"]
            }]
        })
        assert r.status_code == 200, r.text
        assert r.json().get("response")

    def test_chat_max_tokens(self, client: httpx.Client):
        r = client.post("/api/chat", json={
            "model": CHAT_MODEL,
            "temperature": 0.0,
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "Расскажи длинную историю"}]
        })
        assert r.status_code == 200, r.text


class TestGenerate:
    def test_basic_generate(self, client: httpx.Client):
        r = client.post("/api/generate", json={
            "model": CHAT_MODEL,
            "temperature": 0.0,
            "prompt": "Столица России — это"
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body.get("response")
        assert body.get("done") is True

    def test_generate_with_input_field(self, client: httpx.Client):
        """Fallback field 'input' should work like 'prompt'."""
        r = client.post("/api/generate", json={
            "model": CHAT_MODEL,
            "temperature": 0.0,
            "input": "2 + 2 ="
        })
        assert r.status_code == 200, r.text
        assert r.json().get("response")


class TestEmbed:
    def test_basic_embed_string(self, client: httpx.Client):
        r = client.post("/api/embed", json={
            "model": EMBED_MODEL,
            "input": "Квантовая механика"
        })
        assert r.status_code == 200, r.text
        body = r.json()
        embeddings = body.get("embeddings", [])
        assert len(embeddings) > 0, "no embeddings returned"
        assert isinstance(embeddings[0], list)
        assert len(embeddings[0]) > 0

    def test_embed_list_of_strings(self, client: httpx.Client):
        r = client.post("/api/embed", json={
            "model": EMBED_MODEL,
            "input": ["первый текст", "второй текст", "третий текст"]
        })
        assert r.status_code == 200, r.text
        embeddings = r.json().get("embeddings", [])
        assert len(embeddings) == 3

    def test_embed_alias(self, client: httpx.Client):
        r = client.post("/api/embed", json={
            "model": "embed",
            "input": "test"
        })
        assert r.status_code == 200, r.text

    def test_embed_prompt_field(self, client: httpx.Client):
        """Fallback 'prompt' field should work like 'input'."""
        r = client.post("/api/embed", json={
            "model": EMBED_MODEL,
            "prompt": "тест совместимости"
        })
        assert r.status_code == 200, r.text
        assert r.json().get("embeddings")


class TestRerank:
    def test_basic_rerank(self, client: httpx.Client):
        r = client.post("/api/reranker/rerank/v1", json={
            "model": RERANK_MODEL,
            "query": "Квантовая механика",
            "documents": [
                "Квантовая механика описывает поведение частиц на субатомном уровне.",
                "Классическая механика описывает движение макроскопических тел.",
                "Кулинария — это искусство приготовления пищи."
            ]
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert "results" in body
        results = body["results"]
        assert len(results) == 3
        for item in results:
            assert "index"           in item
            assert "relevance_score" in item

    def test_rerank_scores_ordered(self, client: httpx.Client):
        """First doc should score highest for matching query."""
        r = client.post("/api/reranker/rerank/v1", json={
            "model": RERANK_MODEL,
            "query": "Квантовая механика",
            "documents": [
                "Квантовая механика — раздел физики.",
                "Рецепт борща: свёкла, картошка, капуста."
            ]
        })
        assert r.status_code == 200, r.text
        results = sorted(r.json()["results"], key=lambda x: x["relevance_score"], reverse=True)
        assert results[0]["index"] == 0, "quantum doc should rank first"

    def test_rerank_alias(self, client: httpx.Client):
        r = client.post("/api/reranker/rerank/v1", json={
            "model": "reranker",
            "query": "test",
            "documents": ["doc one", "doc two"]
        })
        assert r.status_code == 200, r.text
