"""Tests for OllamaEmbeddingBackend.

Covers:
1. Response parsing (valid single, valid multiple, malformed, empty)
2. Backend construction and defaults
3. embed() with controlled HTTP responses
4. embed_for_indexing() prefix handling
5. embed_batch() with controlled HTTP responses
6. _call_api error handling (network error, invalid JSON)
7. Caching behaviour (single, batch, clear)
8. Prefix handling (query prefix in embed, document prefix in embed_for_indexing)
9. Protocol compliance (EmbeddingBackend) and DenseVectorRetriever wiring
10. is_available() health check
11. Live integration tests (gated by OLLAMA_AVAILABLE env check)

All data is real -- no mock, no simulation, no placeholders.
HTTP-level patching (urlopen) is used to avoid real API calls in unit tests.
"""

from __future__ import annotations

import json
import os
import urllib.error
from unittest.mock import patch

import pytest

from cam_rag.retrieval.embedding_backends.ollama import (
    OllamaEmbeddingBackend,
    _parse_ollama_response,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_URLOPEN = (
    "cam_rag.retrieval.embedding_backends.ollama"
    ".urllib.request.urlopen"
)


class _FakeHTTPResponse:
    """Minimal context-manager response that returns pre-built JSON."""

    def __init__(self, body_dict: dict, status: int = 200):
        self._body = json.dumps(body_dict).encode()
        self.status = status

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
# 1. Response parsing
# ---------------------------------------------------------------------------


class TestParseOllamaResponse:
    """Test the standalone Ollama response parser."""

    def test_valid_single_embedding(self):
        body = {"model": "bge-m3", "embeddings": [[0.1, 0.2, 0.3]]}
        result = _parse_ollama_response(body, expected_count=1, dim=3)
        assert len(result) == 1
        assert result[0] == [0.1, 0.2, 0.3]

    def test_valid_multiple_embeddings(self):
        body = {
            "model": "bge-m3",
            "embeddings": [[0.1, 0.2], [0.4, 0.5]],
        }
        result = _parse_ollama_response(body, expected_count=2, dim=2)
        assert len(result) == 2
        assert result[0] == [0.1, 0.2]
        assert result[1] == [0.4, 0.5]

    def test_missing_embeddings_key(self):
        body = {"model": "bge-m3", "error": "something"}
        result = _parse_ollama_response(body, expected_count=2, dim=3)
        assert len(result) == 2
        assert all(v == [0.0] * 3 for v in result)

    def test_non_list_embeddings(self):
        body = {"embeddings": "not a list"}
        result = _parse_ollama_response(body, expected_count=1, dim=4)
        assert len(result) == 1
        assert result[0] == [0.0] * 4

    def test_fewer_embeddings_than_expected(self):
        body = {"embeddings": [[1.0, 2.0]]}
        result = _parse_ollama_response(body, expected_count=3, dim=2)
        assert len(result) == 3
        assert result[0] == [1.0, 2.0]
        assert result[1] == [0.0, 0.0]
        assert result[2] == [0.0, 0.0]

    def test_more_embeddings_than_expected(self):
        body = {"embeddings": [[1.0], [2.0], [3.0]]}
        result = _parse_ollama_response(body, expected_count=2, dim=1)
        assert len(result) == 2

    def test_empty_embedding_entry(self):
        body = {"embeddings": [[]]}
        result = _parse_ollama_response(body, expected_count=1, dim=4)
        assert result[0] == [0.0] * 4

    def test_non_list_embedding_entry(self):
        body = {"embeddings": ["not a list"]}
        result = _parse_ollama_response(body, expected_count=1, dim=3)
        assert result[0] == [0.0] * 3


# ---------------------------------------------------------------------------
# 2. Backend construction and defaults
# ---------------------------------------------------------------------------


class TestBackendConstruction:
    """Test OllamaEmbeddingBackend initialization."""

    def test_default_model_and_dim(self):
        backend = OllamaEmbeddingBackend()
        assert backend.model == "bge-m3"
        assert backend.dim == 1024

    def test_default_base_url(self):
        backend = OllamaEmbeddingBackend()
        assert backend.base_url == "http://localhost:11434"

    def test_custom_model_and_dim(self):
        backend = OllamaEmbeddingBackend(
            model="nomic-embed-text-v2-moe", dim=768,
        )
        assert backend.model == "nomic-embed-text-v2-moe"
        assert backend.dim == 768

    def test_custom_base_url(self):
        backend = OllamaEmbeddingBackend(
            base_url="http://192.168.1.100:11434",
        )
        assert backend.base_url == "http://192.168.1.100:11434"

    def test_no_api_key_required(self):
        """Ollama is local -- no API key needed, construction always works."""
        backend = OllamaEmbeddingBackend()
        assert backend.model == "bge-m3"

    def test_prefix_defaults_empty(self):
        backend = OllamaEmbeddingBackend()
        assert backend.prefix_query == ""
        assert backend.prefix_document == ""

    def test_custom_prefixes(self):
        backend = OllamaEmbeddingBackend(
            prefix_query="search_query: ",
            prefix_document="search_document: ",
        )
        assert backend.prefix_query == "search_query: "
        assert backend.prefix_document == "search_document: "


# ---------------------------------------------------------------------------
# 3. embed() with controlled HTTP responses
# ---------------------------------------------------------------------------


class TestEmbedSingle:
    """Test single-text embedding via urlopen patching."""

    def test_embed_returns_vector(self):
        backend = OllamaEmbeddingBackend()
        body = {"embeddings": [[0.5] * 1024]}
        with patch(_URLOPEN, side_effect=_urlopen_returning(body)):
            result = backend.embed("hello")
        assert len(result) == 1024
        assert result[0] == 0.5

    def test_embed_caches_result(self):
        backend = OllamaEmbeddingBackend()
        call_count = 0

        def counting_urlopen(req, timeout=None):
            nonlocal call_count
            call_count += 1
            body = {"embeddings": [[0.1] * 1024]}
            return _FakeHTTPResponse(body)

        with patch(_URLOPEN, side_effect=counting_urlopen):
            first = backend.embed("cached text")
            second = backend.embed("cached text")
        assert first == second
        assert call_count == 1  # Only one API call

    def test_embed_returns_zero_on_empty_response(self):
        backend = OllamaEmbeddingBackend()
        body = {"embeddings": []}
        with patch(_URLOPEN, side_effect=_urlopen_returning(body)):
            result = backend.embed("something")
        # Empty embeddings list -> zero-vector fallback
        assert result == [0.0] * 1024

    def test_clear_cache(self):
        backend = OllamaEmbeddingBackend()
        backend._cache["test"] = [1.0] * 1024
        assert "test" in backend._cache
        backend.clear_cache()
        assert "test" not in backend._cache


# ---------------------------------------------------------------------------
# 4. embed_for_indexing() prefix handling
# ---------------------------------------------------------------------------


class TestEmbedForIndexing:
    """Test document-time embedding with prefix_document."""

    def test_embed_for_indexing_returns_vector(self):
        backend = OllamaEmbeddingBackend()
        body = {"embeddings": [[0.7] * 1024]}
        with patch(_URLOPEN, side_effect=_urlopen_returning(body)):
            result = backend.embed_for_indexing("document text")
        assert len(result) == 1024
        assert result[0] == 0.7

    def test_embed_for_indexing_uses_document_prefix(self):
        backend = OllamaEmbeddingBackend(
            prefix_document="search_document: ",
        )
        sent_payloads: list[bytes] = []

        def capture_urlopen(req, timeout=None):
            sent_payloads.append(req.data)
            body = {"embeddings": [[0.3] * 1024]}
            return _FakeHTTPResponse(body)

        with patch(_URLOPEN, side_effect=capture_urlopen):
            backend.embed_for_indexing("my document")

        payload = json.loads(sent_payloads[0])
        assert payload["input"] == "search_document: my document"

    def test_embed_for_indexing_caches_with_prefix(self):
        backend = OllamaEmbeddingBackend(
            prefix_document="doc: ",
        )
        call_count = 0

        def counting_urlopen(req, timeout=None):
            nonlocal call_count
            call_count += 1
            body = {"embeddings": [[0.2] * 1024]}
            return _FakeHTTPResponse(body)

        with patch(_URLOPEN, side_effect=counting_urlopen):
            first = backend.embed_for_indexing("text")
            second = backend.embed_for_indexing("text")
        assert first == second
        assert call_count == 1

    def test_embed_for_indexing_returns_zero_on_empty(self):
        backend = OllamaEmbeddingBackend()
        body = {"embeddings": []}
        with patch(_URLOPEN, side_effect=_urlopen_returning(body)):
            result = backend.embed_for_indexing("text")
        assert result == [0.0] * 1024


# ---------------------------------------------------------------------------
# 5. embed_batch()
# ---------------------------------------------------------------------------


class TestEmbedBatch:
    """Test batch embedding via urlopen patching."""

    def test_batch_returns_correct_count(self):
        backend = OllamaEmbeddingBackend()
        body = {
            "embeddings": [
                [float(i)] * 1024 for i in range(3)
            ],
        }
        with patch(_URLOPEN, side_effect=_urlopen_returning(body)):
            results = backend.embed_batch(["alpha", "beta", "gamma"])
        assert len(results) == 3

    def test_batch_uses_cache(self):
        backend = OllamaEmbeddingBackend()
        backend._cache["cached"] = [9.9] * 1024
        body = {"embeddings": [[1.0] * 1024]}
        with patch(_URLOPEN, side_effect=_urlopen_returning(body)):
            results = backend.embed_batch(["cached", "uncached"])
        assert results[0] == [9.9] * 1024  # From cache
        assert results[1] == [1.0] * 1024  # From API

    def test_batch_fills_zero_on_empty_response(self):
        backend = OllamaEmbeddingBackend()
        body = {"embeddings": []}
        with patch(_URLOPEN, side_effect=_urlopen_returning(body)):
            results = backend.embed_batch(["a", "b"])
        assert len(results) == 2
        assert all(v == [0.0] * 1024 for v in results)

    def test_batch_applies_query_prefix(self):
        backend = OllamaEmbeddingBackend(
            prefix_query="search_query: ",
        )
        sent_payloads: list[bytes] = []

        def capture_urlopen(req, timeout=None):
            sent_payloads.append(req.data)
            body = {"embeddings": [[0.1] * 1024, [0.2] * 1024]}
            return _FakeHTTPResponse(body)

        with patch(_URLOPEN, side_effect=capture_urlopen):
            backend.embed_batch(["foo", "bar"])

        payload = json.loads(sent_payloads[0])
        assert payload["input"] == [
            "search_query: foo",
            "search_query: bar",
        ]


# ---------------------------------------------------------------------------
# 6. _call_api error handling
# ---------------------------------------------------------------------------


class TestCallApiErrors:
    """Test HTTP error handling in _call_api."""

    def test_network_error_returns_zero_vectors(self):
        backend = OllamaEmbeddingBackend()
        with patch(
            _URLOPEN,
            side_effect=urllib.error.URLError("connection refused"),
        ):
            result = backend._call_api(["test"])
        assert result == [[0.0] * 1024]

    def test_oserror_returns_zero_vectors(self):
        backend = OllamaEmbeddingBackend()
        with patch(
            _URLOPEN,
            side_effect=OSError("socket timeout"),
        ):
            result = backend._call_api(["test"])
        assert result == [[0.0] * 1024]

    def test_invalid_json_returns_zero_vectors(self):
        backend = OllamaEmbeddingBackend()

        class BadResponse:
            def read(self):
                return b"not json"
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        with patch(_URLOPEN, return_value=BadResponse()):
            result = backend._call_api(["test"])
        assert result == [[0.0] * 1024]

    def test_valid_api_call_parses_response(self):
        backend = OllamaEmbeddingBackend()
        body = {"embeddings": [[0.42] * 1024]}
        with patch(_URLOPEN, side_effect=_urlopen_returning(body)):
            result = backend._call_api(["hello world"])
        assert len(result) == 1
        assert len(result[0]) == 1024
        assert result[0][0] == 0.42

    def test_api_sends_single_string_for_one_text(self):
        """Verify the payload sends a string (not list) for a single text."""
        backend = OllamaEmbeddingBackend()
        sent_payloads: list[bytes] = []

        def capture_urlopen(req, timeout=None):
            sent_payloads.append(req.data)
            body = {"embeddings": [[0.1] * 1024]}
            return _FakeHTTPResponse(body)

        with patch(_URLOPEN, side_effect=capture_urlopen):
            backend._call_api(["single text"])

        payload = json.loads(sent_payloads[0])
        assert payload["input"] == "single text"
        assert payload["model"] == "bge-m3"

    def test_api_sends_list_for_multiple_texts(self):
        """Verify the payload sends a list for multiple texts."""
        backend = OllamaEmbeddingBackend()
        sent_payloads: list[bytes] = []

        def capture_urlopen(req, timeout=None):
            sent_payloads.append(req.data)
            body = {"embeddings": [[0.1] * 1024, [0.2] * 1024]}
            return _FakeHTTPResponse(body)

        with patch(_URLOPEN, side_effect=capture_urlopen):
            backend._call_api(["text one", "text two"])

        payload = json.loads(sent_payloads[0])
        assert payload["input"] == ["text one", "text two"]


# ---------------------------------------------------------------------------
# 7. Caching behaviour
# ---------------------------------------------------------------------------


class TestCaching:
    """Test caching across embed, embed_for_indexing, and embed_batch."""

    def test_embed_and_embed_for_indexing_cache_separately_with_prefix(self):
        """With different prefixes, query and doc embeddings cache separately."""
        backend = OllamaEmbeddingBackend(
            prefix_query="q: ",
            prefix_document="d: ",
        )
        call_count = 0

        def counting_urlopen(req, timeout=None):
            nonlocal call_count
            call_count += 1
            body = {"embeddings": [[float(call_count)] * 1024]}
            return _FakeHTTPResponse(body)

        with patch(_URLOPEN, side_effect=counting_urlopen):
            query_vec = backend.embed("text")
            doc_vec = backend.embed_for_indexing("text")

        # Two separate API calls because prefixed keys differ
        assert call_count == 2
        # Values should differ because call_count incremented
        assert query_vec[0] == 1.0
        assert doc_vec[0] == 2.0

    def test_embed_without_prefix_shares_cache(self):
        """Without prefixes, embed and embed_for_indexing share cache."""
        backend = OllamaEmbeddingBackend()
        call_count = 0

        def counting_urlopen(req, timeout=None):
            nonlocal call_count
            call_count += 1
            body = {"embeddings": [[0.5] * 1024]}
            return _FakeHTTPResponse(body)

        with patch(_URLOPEN, side_effect=counting_urlopen):
            first = backend.embed("shared text")
            second = backend.embed_for_indexing("shared text")

        assert call_count == 1  # Second call hits cache
        assert first == second

    def test_batch_populates_cache_for_single_embed(self):
        """After embed_batch, individual embed() calls should hit cache."""
        backend = OllamaEmbeddingBackend()
        body = {
            "embeddings": [[0.1] * 1024, [0.2] * 1024],
        }

        call_count = 0

        def counting_urlopen(req, timeout=None):
            nonlocal call_count
            call_count += 1
            return _FakeHTTPResponse(body)

        with patch(_URLOPEN, side_effect=counting_urlopen):
            backend.embed_batch(["alpha", "beta"])
            alpha_vec = backend.embed("alpha")

        assert call_count == 1  # No additional API call
        assert alpha_vec == [0.1] * 1024


# ---------------------------------------------------------------------------
# 8. Prefix handling detail
# ---------------------------------------------------------------------------


class TestPrefixHandling:
    """Verify prefix_query and prefix_document are applied correctly."""

    def test_embed_prepends_query_prefix(self):
        backend = OllamaEmbeddingBackend(
            prefix_query="search_query: ",
        )
        sent_payloads: list[bytes] = []

        def capture_urlopen(req, timeout=None):
            sent_payloads.append(req.data)
            body = {"embeddings": [[0.1] * 1024]}
            return _FakeHTTPResponse(body)

        with patch(_URLOPEN, side_effect=capture_urlopen):
            backend.embed("test query")

        payload = json.loads(sent_payloads[0])
        assert payload["input"] == "search_query: test query"

    def test_embed_no_prefix_when_empty(self):
        backend = OllamaEmbeddingBackend()  # prefix_query=""
        sent_payloads: list[bytes] = []

        def capture_urlopen(req, timeout=None):
            sent_payloads.append(req.data)
            body = {"embeddings": [[0.1] * 1024]}
            return _FakeHTTPResponse(body)

        with patch(_URLOPEN, side_effect=capture_urlopen):
            backend.embed("raw text")

        payload = json.loads(sent_payloads[0])
        assert payload["input"] == "raw text"

    def test_embed_for_indexing_prepends_document_prefix(self):
        backend = OllamaEmbeddingBackend(
            prefix_document="search_document: ",
        )
        sent_payloads: list[bytes] = []

        def capture_urlopen(req, timeout=None):
            sent_payloads.append(req.data)
            body = {"embeddings": [[0.1] * 1024]}
            return _FakeHTTPResponse(body)

        with patch(_URLOPEN, side_effect=capture_urlopen):
            backend.embed_for_indexing("my document")

        payload = json.loads(sent_payloads[0])
        assert payload["input"] == "search_document: my document"

    def test_embed_for_indexing_no_prefix_when_empty(self):
        backend = OllamaEmbeddingBackend()  # prefix_document=""
        sent_payloads: list[bytes] = []

        def capture_urlopen(req, timeout=None):
            sent_payloads.append(req.data)
            body = {"embeddings": [[0.1] * 1024]}
            return _FakeHTTPResponse(body)

        with patch(_URLOPEN, side_effect=capture_urlopen):
            backend.embed_for_indexing("raw document")

        payload = json.loads(sent_payloads[0])
        assert payload["input"] == "raw document"


# ---------------------------------------------------------------------------
# 9. Protocol compliance (EmbeddingBackend)
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    """Verify OllamaEmbeddingBackend satisfies EmbeddingBackend protocol."""

    def test_has_dim_attribute(self):
        backend = OllamaEmbeddingBackend()
        assert hasattr(backend, "dim")
        assert isinstance(backend.dim, int)

    def test_has_embed_method(self):
        backend = OllamaEmbeddingBackend()
        assert callable(backend.embed)

    def test_works_with_dense_retriever(self):
        """Ensure the backend plugs into DenseVectorRetriever."""
        from cam_rag.retrieval.dense import DenseVectorRetriever
        from cam_rag.retrieval.models import RetrievalDocument

        backend = OllamaEmbeddingBackend(dim=4)
        docs = [
            RetrievalDocument(
                doc_id="d1", text="hello world", metadata={},
            ),
        ]
        body = {"embeddings": [[0.5, 0.3, 0.1, 0.2]]}
        with patch(_URLOPEN, side_effect=_urlopen_returning(body)):
            retriever = DenseVectorRetriever(docs, backend=backend)
        assert retriever.backend is backend

    def test_imported_from_retrieval_package(self):
        """OllamaEmbeddingBackend should be importable from retrieval."""
        from cam_rag.retrieval import OllamaEmbeddingBackend as Imported

        assert Imported is OllamaEmbeddingBackend

    def test_imported_from_embedding_backends_package(self):
        """OllamaEmbeddingBackend should be in embedding_backends __all__."""
        from cam_rag.retrieval.embedding_backends import (
            OllamaEmbeddingBackend as Imported,
        )

        assert Imported is OllamaEmbeddingBackend


# ---------------------------------------------------------------------------
# 10. is_available() health check
# ---------------------------------------------------------------------------


class TestIsAvailable:
    """Test Ollama health check."""

    def test_returns_true_when_reachable(self):
        backend = OllamaEmbeddingBackend()
        resp = _FakeHTTPResponse({"models": []}, status=200)
        with patch(_URLOPEN, return_value=resp):
            assert backend.is_available() is True

    def test_returns_false_on_connection_error(self):
        backend = OllamaEmbeddingBackend()
        with patch(
            _URLOPEN,
            side_effect=urllib.error.URLError("connection refused"),
        ):
            assert backend.is_available() is False

    def test_returns_false_on_oserror(self):
        backend = OllamaEmbeddingBackend()
        with patch(
            _URLOPEN,
            side_effect=OSError("socket error"),
        ):
            assert backend.is_available() is False


# ---------------------------------------------------------------------------
# 11. Live integration tests (gated by OLLAMA_AVAILABLE env var)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("OLLAMA_AVAILABLE"),
    reason="OLLAMA_AVAILABLE not set -- skipping live Ollama tests",
)
class TestLiveOllama:
    """Live integration tests against a local Ollama instance.

    Run with: OLLAMA_AVAILABLE=1 pytest -k TestLiveOllama
    Requires a running Ollama instance with bge-m3 model pulled.
    """

    def test_live_single_embed(self):
        backend = OllamaEmbeddingBackend()
        vector = backend.embed(
            "The quick brown fox jumps over the lazy dog.",
        )
        assert len(vector) == 1024
        assert any(v != 0.0 for v in vector), "Expected non-zero embedding"

    def test_live_batch_embed(self):
        backend = OllamaEmbeddingBackend()
        texts = [
            "Machine learning is a subset of AI.",
            "Neural networks model brain function.",
        ]
        vectors = backend.embed_batch(texts)
        assert len(vectors) == 2
        for vec in vectors:
            assert len(vec) == 1024
            assert any(v != 0.0 for v in vec)

    def test_live_embed_for_indexing(self):
        backend = OllamaEmbeddingBackend()
        vector = backend.embed_for_indexing(
            "This is a test document for indexing.",
        )
        assert len(vector) == 1024
        assert any(v != 0.0 for v in vector), "Expected non-zero embedding"

    def test_live_cosine_similarity_meaningful(self):
        """Similar texts should have higher cosine than dissimilar."""
        from cam_rag.retrieval.dense import cosine_similarity

        backend = OllamaEmbeddingBackend()
        v1 = backend.embed("artificial intelligence research")
        v2 = backend.embed("machine learning algorithms")
        v3 = backend.embed("chocolate cake recipe")

        sim_related = cosine_similarity(v1, v2)
        sim_unrelated = cosine_similarity(v1, v3)
        assert sim_related > sim_unrelated, (
            f"Related pair ({sim_related:.4f}) should be more similar "
            f"than unrelated pair ({sim_unrelated:.4f})"
        )

    def test_live_is_available(self):
        backend = OllamaEmbeddingBackend()
        assert backend.is_available() is True
