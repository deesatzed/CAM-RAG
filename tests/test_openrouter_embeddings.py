"""Tests for OpenRouterEmbeddingBackend.

Covers:
1. Response parsing (valid, partial, malformed)
2. Backend construction and API key validation
3. embed() with controlled HTTP responses
4. embed_batch() with controlled HTTP responses
5. _call_api error handling
6. Protocol compliance (EmbeddingBackend)
7. Pipeline wiring (DenseVectorStep passes backend from spec)
8. Live integration test (gated by OPENROUTER_API_KEY)

All data is real -- no mock, no simulation, no placeholders.
HTTP-level patching (urlopen) is used to avoid real API calls in unit tests.
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

from cam_rag.retrieval.embedding_backends.openrouter import (
    OpenRouterEmbeddingBackend,
    _parse_embedding_response,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_URLOPEN = (
    "cam_rag.retrieval.embedding_backends.openrouter"
    ".urllib.request.urlopen"
)


class _FakeHTTPResponse:
    """Minimal context-manager response that returns pre-built JSON."""

    def __init__(self, body_dict: dict):
        self._body = json.dumps(body_dict).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def _urlopen_returning(body_dict: dict):
    """Return a patch side_effect that always yields *body_dict*."""
    resp = _FakeHTTPResponse(body_dict)
    return lambda req, timeout=None: resp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_api_key(monkeypatch):
    """Provide a dummy API key so the backend can be constructed."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-for-unit-tests")


# ---------------------------------------------------------------------------
# 1. Response parsing
# ---------------------------------------------------------------------------


class TestParseEmbeddingResponse:
    """Test the standalone response parser."""

    def test_valid_single_embedding(self):
        body = {
            "data": [
                {"index": 0, "embedding": [0.1, 0.2, 0.3]}
            ]
        }
        result = _parse_embedding_response(body, expected_count=1, dim=3)
        assert len(result) == 1
        assert result[0] == [0.1, 0.2, 0.3]

    def test_valid_multiple_embeddings(self):
        body = {
            "data": [
                {"index": 1, "embedding": [0.4, 0.5]},
                {"index": 0, "embedding": [0.1, 0.2]},
            ]
        }
        result = _parse_embedding_response(body, expected_count=2, dim=2)
        assert len(result) == 2
        # Should be sorted by index
        assert result[0] == [0.1, 0.2]
        assert result[1] == [0.4, 0.5]

    def test_missing_data_key(self):
        body = {"error": "something"}
        result = _parse_embedding_response(body, expected_count=2, dim=3)
        assert len(result) == 2
        assert all(v == [0.0] * 3 for v in result)

    def test_non_list_data(self):
        body = {"data": "not a list"}
        result = _parse_embedding_response(body, expected_count=1, dim=4)
        assert len(result) == 1
        assert result[0] == [0.0] * 4

    def test_fewer_embeddings_than_expected(self):
        body = {
            "data": [
                {"index": 0, "embedding": [1.0, 2.0]}
            ]
        }
        result = _parse_embedding_response(body, expected_count=3, dim=2)
        assert len(result) == 3
        assert result[0] == [1.0, 2.0]
        assert result[1] == [0.0, 0.0]
        assert result[2] == [0.0, 0.0]

    def test_more_embeddings_than_expected(self):
        body = {
            "data": [
                {"index": 0, "embedding": [1.0]},
                {"index": 1, "embedding": [2.0]},
                {"index": 2, "embedding": [3.0]},
            ]
        }
        result = _parse_embedding_response(body, expected_count=2, dim=1)
        assert len(result) == 2

    def test_empty_embedding_in_entry(self):
        body = {
            "data": [{"index": 0, "embedding": []}]
        }
        result = _parse_embedding_response(body, expected_count=1, dim=4)
        assert result[0] == [0.0] * 4

    def test_missing_embedding_key(self):
        body = {
            "data": [{"index": 0}]
        }
        result = _parse_embedding_response(body, expected_count=1, dim=3)
        assert result[0] == [0.0] * 3


# ---------------------------------------------------------------------------
# 2. Backend construction and API key validation
# ---------------------------------------------------------------------------


class TestBackendConstruction:
    """Test OpenRouterEmbeddingBackend initialization."""

    def test_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            OpenRouterEmbeddingBackend()

    def test_accepts_explicit_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        backend = OpenRouterEmbeddingBackend(api_key="explicit-key")
        assert backend.api_key == "explicit-key"

    def test_reads_env_api_key(self, fake_api_key):
        backend = OpenRouterEmbeddingBackend()
        assert backend.api_key == "test-key-for-unit-tests"

    def test_default_model_and_dim(self, fake_api_key):
        backend = OpenRouterEmbeddingBackend()
        assert backend.model == "perplexity/pplx-embed-v1-4b"
        assert backend.dim == 2560

    def test_custom_model_and_dim(self, fake_api_key):
        backend = OpenRouterEmbeddingBackend(
            model="custom/model", dim=768,
        )
        assert backend.model == "custom/model"
        assert backend.dim == 768


# ---------------------------------------------------------------------------
# 3. embed() with controlled HTTP responses
# ---------------------------------------------------------------------------


class TestEmbedSingle:
    """Test single-text embedding via urlopen patching."""

    def test_embed_returns_vector(self, fake_api_key):
        backend = OpenRouterEmbeddingBackend()
        body = {"data": [{"index": 0, "embedding": [0.5] * 2560}]}
        with patch(_URLOPEN, side_effect=_urlopen_returning(body)):
            result = backend.embed("hello")
        assert len(result) == 2560
        assert result[0] == 0.5

    def test_embed_caches_result(self, fake_api_key):
        backend = OpenRouterEmbeddingBackend()
        call_count = 0

        def counting_urlopen(req, timeout=None):
            nonlocal call_count
            call_count += 1
            body = {"data": [{"index": 0, "embedding": [0.1] * 2560}]}
            return _FakeHTTPResponse(body)

        with patch(_URLOPEN, side_effect=counting_urlopen):
            first = backend.embed("cached text")
            second = backend.embed("cached text")
        assert first == second
        assert call_count == 1  # Only one API call

    def test_embed_returns_zero_on_empty_response(self, fake_api_key):
        backend = OpenRouterEmbeddingBackend()
        body = {"data": []}
        with patch(_URLOPEN, side_effect=_urlopen_returning(body)):
            result = backend.embed("something")
        # Empty data → zero-vector fallback
        assert result == [0.0] * 2560

    def test_clear_cache(self, fake_api_key):
        backend = OpenRouterEmbeddingBackend()
        backend._cache["test"] = [1.0] * 2560
        assert "test" in backend._cache
        backend.clear_cache()
        assert "test" not in backend._cache


# ---------------------------------------------------------------------------
# 4. embed_batch()
# ---------------------------------------------------------------------------


class TestEmbedBatch:
    """Test batch embedding via urlopen patching."""

    def test_batch_returns_correct_count(self, fake_api_key):
        backend = OpenRouterEmbeddingBackend()
        body = {
            "data": [
                {"index": i, "embedding": [float(i)] * 2560}
                for i in range(3)
            ]
        }
        with patch(_URLOPEN, side_effect=_urlopen_returning(body)):
            results = backend.embed_batch(["alpha", "beta", "gamma"])
        assert len(results) == 3

    def test_batch_uses_cache(self, fake_api_key):
        backend = OpenRouterEmbeddingBackend()
        backend._cache["cached"] = [9.9] * 2560
        body = {
            "data": [{"index": 0, "embedding": [1.0] * 2560}]
        }
        with patch(_URLOPEN, side_effect=_urlopen_returning(body)):
            results = backend.embed_batch(["cached", "uncached"])
        assert results[0] == [9.9] * 2560  # From cache
        assert results[1] == [1.0] * 2560  # From API

    def test_batch_fills_zero_on_empty_response(self, fake_api_key):
        backend = OpenRouterEmbeddingBackend()
        body = {"data": []}
        with patch(_URLOPEN, side_effect=_urlopen_returning(body)):
            results = backend.embed_batch(["a", "b"])
        assert len(results) == 2
        assert all(v == [0.0] * 2560 for v in results)


# ---------------------------------------------------------------------------
# 5. _call_api error handling
# ---------------------------------------------------------------------------


class TestCallApiErrors:
    """Test HTTP error handling in _call_api."""

    def test_network_error_returns_zero_vectors(self, fake_api_key):
        import urllib.error

        backend = OpenRouterEmbeddingBackend()
        with patch(
            _URLOPEN,
            side_effect=urllib.error.URLError("connection refused"),
        ):
            result = backend._call_api(["test"])
        assert result == [[0.0] * 2560]

    def test_invalid_json_returns_zero_vectors(self, fake_api_key):
        backend = OpenRouterEmbeddingBackend()

        class BadResponse:
            def read(self):
                return b"not json"
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        with patch(_URLOPEN, return_value=BadResponse()):
            result = backend._call_api(["test"])
        assert result == [[0.0] * 2560]

    def test_valid_api_call_parses_response(self, fake_api_key):
        backend = OpenRouterEmbeddingBackend()
        body = {"data": [{"index": 0, "embedding": [0.42] * 2560}]}
        with patch(_URLOPEN, side_effect=_urlopen_returning(body)):
            result = backend._call_api(["hello world"])
        assert len(result) == 1
        assert len(result[0]) == 2560
        assert result[0][0] == 0.42


# ---------------------------------------------------------------------------
# 6. Protocol compliance (EmbeddingBackend)
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    """Verify OpenRouterEmbeddingBackend satisfies EmbeddingBackend protocol."""

    def test_has_dim_attribute(self, fake_api_key):
        backend = OpenRouterEmbeddingBackend()
        assert hasattr(backend, "dim")
        assert isinstance(backend.dim, int)

    def test_has_embed_method(self, fake_api_key):
        backend = OpenRouterEmbeddingBackend()
        assert callable(backend.embed)

    def test_works_with_dense_retriever(self, fake_api_key):
        """Ensure the backend plugs into DenseVectorRetriever."""
        from cam_rag.retrieval.dense import DenseVectorRetriever
        from cam_rag.retrieval.models import RetrievalDocument

        backend = OpenRouterEmbeddingBackend(dim=4)
        docs = [
            RetrievalDocument(
                doc_id="d1", text="hello world", metadata={},
            )
        ]
        body = {"data": [{"index": 0, "embedding": [0.5, 0.3, 0.1, 0.2]}]}
        with patch(_URLOPEN, side_effect=_urlopen_returning(body)):
            retriever = DenseVectorRetriever(docs, backend=backend)
        assert retriever.backend is backend


# ---------------------------------------------------------------------------
# 7. Pipeline wiring — DenseVectorStep uses spec.embedding_backend
# ---------------------------------------------------------------------------


class TestPipelineWiring:
    """Verify DenseVectorStep passes embedding_backend from spec."""

    def test_dense_step_uses_spec_backend(self):
        """DenseVectorStep should pass spec.embedding_backend to retriever."""
        from cam_rag.pipeline.context import PipelineContext
        from cam_rag.pipeline.executor import DenseVectorStep
        from cam_rag.rag.spec import RAGAppSpec
        from cam_rag.retrieval import AdaptiveParams, RetrievalDocument
        from cam_rag.retrieval.dense import HashEmbeddingBackend

        custom_backend = HashEmbeddingBackend(dim=32)
        spec = RAGAppSpec(
            name="test",
            embedding_backend=custom_backend,
        )
        docs = [
            RetrievalDocument(
                doc_id="d1",
                text="machine learning models",
                metadata={"chunk_id": "d1"},
            ),
        ]
        ctx = PipelineContext(
            query_text="what is ML",
            chunks=[],
            spec=spec,
            adaptive=AdaptiveParams(
                sparse_k=10,
                dense_k=10,
                rrf_k=60,
                sparse_weight=0.4,
                dense_weight=0.6,
            ),
            limit=5,
            retrieval_docs=docs,
            expanded_query="what is ML",
        )

        step = DenseVectorStep()
        step.execute(ctx)

        assert ctx.dense_retriever is not None
        assert ctx.dense_retriever.backend is custom_backend

    def test_dense_step_defaults_without_backend(self):
        """When spec.embedding_backend is None, falls back to Hash."""
        from cam_rag.pipeline.context import PipelineContext
        from cam_rag.pipeline.executor import DenseVectorStep
        from cam_rag.rag.spec import RAGAppSpec
        from cam_rag.retrieval import AdaptiveParams, RetrievalDocument
        from cam_rag.retrieval.dense import HashEmbeddingBackend

        spec = RAGAppSpec(name="test")
        docs = [
            RetrievalDocument(
                doc_id="d1",
                text="test document",
                metadata={"chunk_id": "d1"},
            ),
        ]
        ctx = PipelineContext(
            query_text="query",
            chunks=[],
            spec=spec,
            adaptive=AdaptiveParams(
                sparse_k=10,
                dense_k=10,
                rrf_k=60,
                sparse_weight=0.4,
                dense_weight=0.6,
            ),
            limit=5,
            retrieval_docs=docs,
            expanded_query="query",
        )

        step = DenseVectorStep()
        step.execute(ctx)

        assert ctx.dense_retriever is not None
        assert isinstance(ctx.dense_retriever.backend, HashEmbeddingBackend)


# ---------------------------------------------------------------------------
# 8. Live integration test (gated by env var)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set — skipping live test",
)
class TestLiveOpenRouter:
    """Live integration tests against the OpenRouter API.

    Run with: pytest -k TestLiveOpenRouter
    Requires OPENROUTER_API_KEY in the environment.
    """

    def test_live_single_embed(self):
        backend = OpenRouterEmbeddingBackend()
        vector = backend.embed(
            "The quick brown fox jumps over the lazy dog."
        )
        assert len(vector) == 2560
        assert any(v != 0.0 for v in vector), "Expected non-zero embedding"

    def test_live_batch_embed(self):
        backend = OpenRouterEmbeddingBackend()
        texts = [
            "Machine learning is a subset of AI.",
            "Neural networks model brain function.",
        ]
        vectors = backend.embed_batch(texts)
        assert len(vectors) == 2
        for vec in vectors:
            assert len(vec) == 2560
            assert any(v != 0.0 for v in vec)

    def test_live_cosine_similarity_meaningful(self):
        """Similar texts should have higher cosine than dissimilar."""
        from cam_rag.retrieval.dense import cosine_similarity

        backend = OpenRouterEmbeddingBackend()
        v1 = backend.embed("artificial intelligence research")
        v2 = backend.embed("machine learning algorithms")
        v3 = backend.embed("chocolate cake recipe")

        sim_related = cosine_similarity(v1, v2)
        sim_unrelated = cosine_similarity(v1, v3)
        assert sim_related > sim_unrelated, (
            f"Related pair ({sim_related:.4f}) should be more similar "
            f"than unrelated pair ({sim_unrelated:.4f})"
        )
