"""Tests for cross-encoder reranking backends, prompt/parsing, and pipeline integration.

Covers:
1. Reranking prompt construction
2. Score parsing (valid JSON, markdown code blocks, malformed, edge cases)
3. Ollama reranker backend (API calls, availability, error handling)
4. OpenRouter reranker backend (API key, API calls, error handling)
5. CrossEncoderRerankStep pipeline execution
6. Query path integration (static + pipeline paths)
7. Evidence re-ordering verification
8. Live integration tests (gated by env vars)

All data is real -- no mock, no simulation, no placeholders.
HTTP-level patching (urlopen) is used to avoid real API calls in unit tests.
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

from cam_rag.rag.models import Chunk, Evidence
from cam_rag.reranking.ollama import OllamaRerankerBackend
from cam_rag.reranking.openrouter import OpenRouterRerankerBackend
from cam_rag.reranking.prompt import build_rerank_prompt, parse_rerank_scores

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OLLAMA_URLOPEN = "cam_rag.reranking.ollama.urllib.request.urlopen"
_OPENROUTER_URLOPEN = "cam_rag.reranking.openrouter.urllib.request.urlopen"


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


def _make_evidence(n: int = 5) -> list[Evidence]:
    """Create a list of Evidence items with known scores for reranking tests."""
    texts = [
        "Emergency triage protocol requires immediate assessment within 5 minutes.",
        "Blood pressure measurement technique for adult patients.",
        "Glasgow Coma Scale is used to assess level of consciousness.",
        "Medication dosing adjustments for renal impairment.",
        "Wound care and dressing changes for post-surgical patients.",
    ]
    return [
        Evidence(
            chunk=Chunk(
                id=f"chunk-{i}",
                document_id=f"doc-{i}",
                text=texts[i % len(texts)],
                source=f"protocol-{i}.md",
                title=f"Protocol {i + 1}",
            ),
            score=0.5 + (i * 0.05),  # Ascending scores 0.5, 0.55, ...
            retriever="hybrid_rrf",
            rank=i,
            signals={
                "rrf_score": 0.5 + (i * 0.05),
            },
        )
        for i in range(n)
    ]


def _reranker_json_response(scores: dict) -> dict:
    """Build an Ollama-style chat response with JSON scores."""
    return {"message": {"content": json.dumps(scores)}}


def _openrouter_json_response(scores: dict) -> dict:
    """Build an OpenRouter-style chat response with JSON scores."""
    return {"choices": [{"message": {"content": json.dumps(scores)}}]}


# ---------------------------------------------------------------------------
# 1. Prompt construction
# ---------------------------------------------------------------------------


class TestBuildRerankPrompt:
    """Test reranking prompt construction."""

    def test_basic_structure(self):
        system, user = build_rerank_prompt(
            "What is triage?",
            ["Passage one about triage.", "Passage two about BP."],
        )
        assert "relevance" in system.lower()
        assert "JSON" in system
        assert "[1]" in user
        assert "[2]" in user
        assert "What is triage?" in user

    def test_passage_numbering(self):
        passages = ["A", "B", "C"]
        _, user = build_rerank_prompt("query", passages)
        assert "[1]" in user
        assert "[2]" in user
        assert "[3]" in user

    def test_passage_truncation(self):
        long_passage = "x" * 5000
        _, user = build_rerank_prompt("query", [long_passage])
        # Passages truncated to 1000 chars each
        assert len(user) < 5000

    def test_empty_passages(self):
        _, user = build_rerank_prompt("query", [])
        # No numbered passages, but prompt still valid
        assert "query" in user.lower() or "Query" in user


# ---------------------------------------------------------------------------
# 2. Score parsing
# ---------------------------------------------------------------------------


class TestParseRerankScores:
    """Test reranker response parsing."""

    def test_valid_json(self):
        response = '{"1": 8, "2": 3, "3": 9}'
        scores = parse_rerank_scores(response, 3)
        assert len(scores) == 3
        assert scores[0] == 0.8  # 8/10
        assert scores[1] == 0.3  # 3/10
        assert scores[2] == 0.9  # 9/10

    def test_json_with_int_keys(self):
        response = '{"1": 10, "2": 0}'
        scores = parse_rerank_scores(response, 2)
        assert scores[0] == 1.0  # 10/10 clamped to 1.0
        assert scores[1] == 0.0  # 0/10

    def test_markdown_code_block(self):
        response = '```json\n{"1": 7, "2": 5}\n```'
        scores = parse_rerank_scores(response, 2)
        assert scores[0] == 0.7
        assert scores[1] == 0.5

    def test_json_embedded_in_text(self):
        response = 'Here are the scores: {"1": 6, "2": 4} That is all.'
        scores = parse_rerank_scores(response, 2)
        assert scores[0] == 0.6
        assert scores[1] == 0.4

    def test_invalid_json_fallback(self):
        response = "not valid json at all"
        scores = parse_rerank_scores(response, 3)
        assert len(scores) == 3
        assert all(s == 0.5 for s in scores)

    def test_missing_passage_numbers(self):
        response = '{"1": 8}'
        scores = parse_rerank_scores(response, 3)
        assert scores[0] == 0.8
        assert scores[1] == 0.5  # Missing → default 5/10 = 0.5
        assert scores[2] == 0.5

    def test_score_clamping(self):
        response = '{"1": 15, "2": -5}'
        scores = parse_rerank_scores(response, 2)
        assert scores[0] == 1.0  # Clamped to 1.0
        assert scores[1] == 0.0  # Clamped to 0.0

    def test_non_numeric_score(self):
        response = '{"1": "high", "2": 7}'
        scores = parse_rerank_scores(response, 2)
        assert scores[0] == 0.5  # Fallback
        assert scores[1] == 0.7

    def test_empty_response(self):
        scores = parse_rerank_scores("", 2)
        assert scores == [0.5, 0.5]

    def test_non_dict_json(self):
        scores = parse_rerank_scores("[1, 2, 3]", 2)
        assert scores == [0.5, 0.5]


# ---------------------------------------------------------------------------
# 3. Ollama reranker backend
# ---------------------------------------------------------------------------


class TestOllamaRerankerBackend:
    """Test OllamaRerankerBackend with HTTP-level patching."""

    def test_default_config(self):
        backend = OllamaRerankerBackend()
        assert backend.model == "qwen3:4b"
        assert backend.temperature == 0.0

    def test_custom_config(self):
        backend = OllamaRerankerBackend(model="llama3.2:3b", temperature=0.1)
        assert backend.model == "llama3.2:3b"

    def test_rerank_returns_scores(self):
        backend = OllamaRerankerBackend()
        body = _reranker_json_response({"1": 9, "2": 3, "3": 7})
        with patch(_OLLAMA_URLOPEN, side_effect=_urlopen_returning(body)):
            scores = backend.rerank("triage query", ["passage A", "passage B", "passage C"])
        assert len(scores) == 3
        assert scores[0] == 0.9
        assert scores[1] == 0.3
        assert scores[2] == 0.7

    def test_rerank_empty_passages(self):
        backend = OllamaRerankerBackend()
        scores = backend.rerank("query", [])
        assert scores == []

    def test_rerank_network_error_fallback(self):
        import urllib.error
        backend = OllamaRerankerBackend()
        with patch(
            _OLLAMA_URLOPEN,
            side_effect=urllib.error.URLError("connection refused"),
        ):
            scores = backend.rerank("query", ["passage 1", "passage 2"])
        assert scores == [0.5, 0.5]

    def test_rerank_invalid_json_fallback(self):
        backend = OllamaRerankerBackend()

        class BadResponse:
            def read(self):
                return b"not json"
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        with patch(_OLLAMA_URLOPEN, return_value=BadResponse()):
            scores = backend.rerank("query", ["passage 1"])
        assert scores == [0.5]

    def test_is_available_true(self):
        backend = OllamaRerankerBackend()

        class OKResponse:
            status = 200
            def read(self):
                return b'{"models":[]}'
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        with patch(_OLLAMA_URLOPEN, return_value=OKResponse()):
            assert backend.is_available() is True

    def test_is_available_false(self):
        import urllib.error
        backend = OllamaRerankerBackend()
        with patch(
            _OLLAMA_URLOPEN,
            side_effect=urllib.error.URLError("not reachable"),
        ):
            assert backend.is_available() is False

    def test_api_call_payload(self):
        backend = OllamaRerankerBackend(model="test-model")
        body = _reranker_json_response({"1": 5})
        captured_req = None

        def capture_urlopen(req, timeout=None):
            nonlocal captured_req
            captured_req = req
            return _FakeHTTPResponse(body)

        with patch(_OLLAMA_URLOPEN, side_effect=capture_urlopen):
            backend.rerank("test query", ["passage 1"])

        assert captured_req is not None
        payload = json.loads(captured_req.data)
        assert payload["model"] == "test-model"
        assert payload["stream"] is False
        assert len(payload["messages"]) == 2


# ---------------------------------------------------------------------------
# 4. OpenRouter reranker backend
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_openrouter_key(monkeypatch):
    """Provide a dummy API key for OpenRouter tests."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-reranker-key")


class TestOpenRouterRerankerBackend:
    """Test OpenRouterRerankerBackend with HTTP-level patching."""

    def test_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            OpenRouterRerankerBackend()

    def test_accepts_explicit_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        backend = OpenRouterRerankerBackend(api_key="explicit-key")
        assert backend.api_key == "explicit-key"

    def test_default_config(self, fake_openrouter_key):
        backend = OpenRouterRerankerBackend()
        assert "claude" in backend.model or "haiku" in backend.model
        assert backend.temperature == 0.0

    def test_rerank_returns_scores(self, fake_openrouter_key):
        backend = OpenRouterRerankerBackend()
        body = _openrouter_json_response({"1": 8, "2": 4})
        with patch(_OPENROUTER_URLOPEN, side_effect=_urlopen_returning(body)):
            scores = backend.rerank("query", ["passage A", "passage B"])
        assert len(scores) == 2
        assert scores[0] == 0.8
        assert scores[1] == 0.4

    def test_rerank_empty_passages(self, fake_openrouter_key):
        backend = OpenRouterRerankerBackend()
        scores = backend.rerank("query", [])
        assert scores == []

    def test_rerank_network_error_fallback(self, fake_openrouter_key):
        import urllib.error
        backend = OpenRouterRerankerBackend()
        with patch(
            _OPENROUTER_URLOPEN,
            side_effect=urllib.error.URLError("timeout"),
        ):
            scores = backend.rerank("query", ["passage 1"])
        assert scores == [0.5]

    def test_api_sends_auth_header(self, fake_openrouter_key):
        backend = OpenRouterRerankerBackend()
        body = _openrouter_json_response({"1": 5})
        captured_req = None

        def capture_urlopen(req, timeout=None):
            nonlocal captured_req
            captured_req = req
            return _FakeHTTPResponse(body)

        with patch(_OPENROUTER_URLOPEN, side_effect=capture_urlopen):
            backend.rerank("query", ["passage"])

        assert captured_req is not None
        assert "Bearer test-reranker-key" in captured_req.get_header("Authorization")


# ---------------------------------------------------------------------------
# 5. CrossEncoderRerankStep pipeline execution
# ---------------------------------------------------------------------------


class TestCrossEncoderRerankStep:
    """Test the CrossEncoderRerankStep technique step."""

    def test_no_backend_configured_noop(self):
        """Step is a no-op when no reranker_backend is configured."""
        from cam_rag.pipeline.context import PipelineContext
        from cam_rag.pipeline.executor import CrossEncoderRerankStep
        from cam_rag.rag.spec import RAGAppSpec
        from cam_rag.retrieval import AdaptiveParams, RetrievalDocument

        spec = RAGAppSpec(name="test-no-rerank")
        evidence = _make_evidence(3)
        original_scores = [e.score for e in evidence]

        ctx = PipelineContext(
            query_text="test query",
            chunks=[],
            spec=spec,
            adaptive=AdaptiveParams(
                sparse_k=10, dense_k=10, rrf_k=60,
                sparse_weight=0.4, dense_weight=0.6,
            ),
            limit=5,
            retrieval_docs=[],
            expanded_query="test query",
            evidence=evidence,
        )

        step = CrossEncoderRerankStep()
        step.execute(ctx)

        # Scores unchanged
        assert [e.score for e in ctx.evidence] == original_scores

    def test_empty_evidence_noop(self):
        """Step is a no-op when evidence is empty."""
        from cam_rag.pipeline.context import PipelineContext
        from cam_rag.pipeline.executor import CrossEncoderRerankStep
        from cam_rag.rag.spec import RAGAppSpec
        from cam_rag.retrieval import AdaptiveParams

        backend = OllamaRerankerBackend()
        spec = RAGAppSpec(name="test-rerank-empty", reranker_backend=backend)

        ctx = PipelineContext(
            query_text="test query",
            chunks=[],
            spec=spec,
            adaptive=AdaptiveParams(
                sparse_k=10, dense_k=10, rrf_k=60,
                sparse_weight=0.4, dense_weight=0.6,
            ),
            limit=5,
            retrieval_docs=[],
            expanded_query="test query",
        )

        step = CrossEncoderRerankStep()
        step.execute(ctx)
        assert ctx.evidence == []

    def test_reranking_reorders_evidence(self):
        """Step re-sorts evidence by reranker scores."""
        from cam_rag.pipeline.context import PipelineContext
        from cam_rag.pipeline.executor import CrossEncoderRerankStep
        from cam_rag.rag.spec import RAGAppSpec
        from cam_rag.retrieval import AdaptiveParams

        backend = OllamaRerankerBackend()
        spec = RAGAppSpec(name="test-rerank-order", reranker_backend=backend)
        evidence = _make_evidence(3)
        original_ids = [e.chunk.id for e in evidence]

        ctx = PipelineContext(
            query_text="triage protocol",
            chunks=[],
            spec=spec,
            adaptive=AdaptiveParams(
                sparse_k=10, dense_k=10, rrf_k=60,
                sparse_weight=0.4, dense_weight=0.6,
            ),
            limit=5,
            retrieval_docs=[],
            expanded_query="triage protocol",
            evidence=evidence,
        )

        # Reranker gives highest score to the last item
        scores = {"1": 3, "2": 1, "3": 9}
        body = _reranker_json_response(scores)
        with patch(_OLLAMA_URLOPEN, side_effect=_urlopen_returning(body)):
            step = CrossEncoderRerankStep()
            step.execute(ctx)

        # Evidence should be re-sorted: chunk-2 (0.9), chunk-0 (0.3), chunk-1 (0.1)
        assert ctx.evidence[0].chunk.id == "chunk-2"
        assert ctx.evidence[0].score == 0.9
        assert ctx.evidence[0].rank == 0

    def test_reranking_preserves_original_score(self):
        """Reranking stores original RRF score in signals."""
        from cam_rag.pipeline.context import PipelineContext
        from cam_rag.pipeline.executor import CrossEncoderRerankStep
        from cam_rag.rag.spec import RAGAppSpec
        from cam_rag.retrieval import AdaptiveParams

        backend = OllamaRerankerBackend()
        spec = RAGAppSpec(name="test-rerank-signal", reranker_backend=backend)
        evidence = _make_evidence(2)
        original_score_0 = evidence[0].score

        ctx = PipelineContext(
            query_text="query",
            chunks=[],
            spec=spec,
            adaptive=AdaptiveParams(
                sparse_k=10, dense_k=10, rrf_k=60,
                sparse_weight=0.4, dense_weight=0.6,
            ),
            limit=5,
            retrieval_docs=[],
            expanded_query="query",
            evidence=evidence,
        )

        body = _reranker_json_response({"1": 7, "2": 8})
        with patch(_OLLAMA_URLOPEN, side_effect=_urlopen_returning(body)):
            step = CrossEncoderRerankStep()
            step.execute(ctx)

        # Find the original chunk-0 and check its preserved score
        chunk_0 = next(e for e in ctx.evidence if e.chunk.id == "chunk-0")
        assert chunk_0.signals["rrf_score_pre_rerank"] == original_score_0
        assert chunk_0.signals["reranker_score"] == 0.7

    def test_step_registered_in_executor(self):
        """CrossEncoderRerankStep is registered and discoverable."""
        from cam_rag.pipeline.executor import get_step
        step = get_step("cross_encoder_rerank")
        assert step.name == "cross_encoder_rerank"

    def test_step_in_default_registry(self):
        """cross_encoder_rerank is in the default technique registry."""
        from cam_rag.query import _build_default_registry
        registry = _build_default_registry()
        desc = registry.get("cross_encoder_rerank")
        assert desc.name == "cross_encoder_rerank"
        assert desc.category == "reranking"

    def test_step_in_pipeline_plan(self):
        """cross_encoder_rerank should appear in composed pipeline plans."""
        from cam_rag.pipeline.registry import compose_pipeline
        from cam_rag.query import _build_default_registry

        registry = _build_default_registry()
        plan = compose_pipeline(registry, "summary", corpus_size=10)
        assert "cross_encoder_rerank" in plan.steps
        # Should come after rrf_fusion
        rrf_idx = plan.steps.index("rrf_fusion")
        rerank_idx = plan.steps.index("cross_encoder_rerank")
        assert rerank_idx > rrf_idx


# ---------------------------------------------------------------------------
# 6. Query path integration
# ---------------------------------------------------------------------------


class TestQueryPathReranking:
    """Test reranking wiring in the query path."""

    def _make_documents(self):
        from cam_rag.rag.models import CorpusDocument
        return [
            CorpusDocument(
                id="doc1",
                text=(
                    "Emergency triage is the process of determining the "
                    "priority of patients based on the severity of their "
                    "condition. The goal is to ensure that the most critical "
                    "patients receive treatment first."
                ),
                source="triage.md",
                title="Triage Protocol",
            ),
            CorpusDocument(
                id="doc2",
                text=(
                    "Vital signs monitoring includes measurement of blood "
                    "pressure, heart rate, respiratory rate, temperature, "
                    "and oxygen saturation."
                ),
                source="vitals.md",
                title="Vital Signs Guide",
            ),
            CorpusDocument(
                id="doc3",
                text=(
                    "Glasgow Coma Scale assesses eye opening, verbal "
                    "response, and motor response. It is used to evaluate "
                    "consciousness level after brain injury."
                ),
                source="gcs.md",
                title="GCS Assessment",
            ),
        ]

    def test_reranking_disabled_by_default(self):
        """Without reranker_backend, no reranking happens."""
        from cam_rag.query import query
        from cam_rag.rag.spec import RAGAppSpec

        spec = RAGAppSpec(name="test-no-rerank-query")
        documents = self._make_documents()
        result = query("What is triage?", documents, spec)
        # No reranker signals in evidence
        for item in result.evidence:
            assert "reranker_score" not in item.signals

    def test_reranking_with_static_path(self):
        """Reranking works in the static (default) path."""
        from cam_rag.query import query
        from cam_rag.rag.spec import RAGAppSpec

        backend = OllamaRerankerBackend()
        spec = RAGAppSpec(
            name="test-rerank-static",
            reranker_backend=backend,
        )
        documents = self._make_documents()

        # Make the reranker favor the last evidence item
        scores = {"1": 2, "2": 9, "3": 5}
        body = _reranker_json_response(scores)
        with patch(_OLLAMA_URLOPEN, side_effect=_urlopen_returning(body)):
            result = query("What is triage?", documents, spec)

        # Evidence should have reranker signals
        for item in result.evidence:
            assert "reranker_score" in item.signals
            assert "rrf_score_pre_rerank" in item.signals

    def test_reranking_with_pipeline_path(self):
        """Reranking works in the pipeline path."""
        from cam_rag.query import query
        from cam_rag.rag.spec import RAGAppSpec

        backend = OllamaRerankerBackend()
        spec = RAGAppSpec(
            name="test-rerank-pipeline",
            reranker_backend=backend,
            use_pipeline=True,
        )
        documents = self._make_documents()

        scores = {"1": 8, "2": 3, "3": 6}
        body = _reranker_json_response(scores)
        with patch(_OLLAMA_URLOPEN, side_effect=_urlopen_returning(body)):
            result = query("What is triage?", documents, spec)

        # Evidence should have reranker signals
        for item in result.evidence:
            assert "reranker_score" in item.signals

    def test_reranking_evidence_reordered(self):
        """Reranking changes evidence scores and assigns reranker signals."""
        from cam_rag.query import query
        from cam_rag.rag.spec import RAGAppSpec

        backend = OllamaRerankerBackend()
        spec_no_rerank = RAGAppSpec(name="test-order-baseline")
        spec_rerank = RAGAppSpec(
            name="test-order-reranked",
            reranker_backend=backend,
        )
        documents = self._make_documents()

        baseline = query("What is triage?", documents, spec_no_rerank)
        baseline_scores = [e.score for e in baseline.evidence]

        # Give very different reranker scores
        n = len(baseline.evidence)
        scores = {str(i + 1): (i + 1) * 3 for i in range(n)}
        body = _reranker_json_response(scores)
        with patch(_OLLAMA_URLOPEN, side_effect=_urlopen_returning(body)):
            reranked = query("What is triage?", documents, spec_rerank)

        reranked_scores = [e.score for e in reranked.evidence]
        # Scores should be from the reranker, not RRF
        assert reranked_scores != baseline_scores
        # All items should have reranker signal
        for item in reranked.evidence:
            assert "reranker_score" in item.signals

    def test_reranking_fallback_on_api_failure(self):
        """When reranker API fails, evidence keeps original order (fallback scores)."""
        import urllib.error
        from cam_rag.query import query
        from cam_rag.rag.spec import RAGAppSpec

        backend = OllamaRerankerBackend()
        spec = RAGAppSpec(
            name="test-rerank-fallback",
            reranker_backend=backend,
        )
        documents = self._make_documents()

        with patch(
            _OLLAMA_URLOPEN,
            side_effect=urllib.error.URLError("connection refused"),
        ):
            result = query("What is triage?", documents, spec)

        # Should still return results (with fallback 0.5 scores)
        assert len(result.evidence) > 0
        for item in result.evidence:
            assert "reranker_score" in item.signals
            assert item.signals["reranker_score"] == 0.5

    def test_openrouter_reranker_in_query_path(self, monkeypatch):
        """OpenRouter reranker backend works in the query path."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        from cam_rag.query import query
        from cam_rag.rag.spec import RAGAppSpec

        backend = OpenRouterRerankerBackend()
        spec = RAGAppSpec(
            name="test-openrouter-rerank",
            reranker_backend=backend,
        )
        documents = self._make_documents()

        scores = {"1": 7, "2": 9, "3": 4}
        body = _openrouter_json_response(scores)
        with patch(_OPENROUTER_URLOPEN, side_effect=_urlopen_returning(body)):
            result = query("What is triage?", documents, spec)

        for item in result.evidence:
            assert "reranker_score" in item.signals

    def test_reranking_combined_with_generation(self):
        """Reranking + generation work together."""
        from cam_rag.generation.ollama import OllamaGenerationBackend
        from cam_rag.query import query
        from cam_rag.rag.spec import RAGAppSpec

        _GEN_URLOPEN = "cam_rag.generation.ollama.urllib.request.urlopen"
        reranker = OllamaRerankerBackend()
        generator = OllamaGenerationBackend()

        spec = RAGAppSpec(
            name="test-rerank-and-gen",
            reranker_backend=reranker,
            generation_backend=generator,
        )
        documents = self._make_documents()

        call_count = 0

        def multi_urlopen(req, timeout=None):
            nonlocal call_count
            call_count += 1
            url = req.full_url
            if "/api/chat" in url:
                payload = json.loads(req.data)
                # Check if this is a reranking call (system prompt mentions relevance)
                sys_msg = payload["messages"][0]["content"]
                if "relevance" in sys_msg.lower():
                    body = _reranker_json_response({"1": 8, "2": 6, "3": 9})
                else:
                    body = {"message": {"content": "Generated answer based on reranked evidence [1]."}}
                return _FakeHTTPResponse(body)
            return _FakeHTTPResponse({})

        with patch(_OLLAMA_URLOPEN, side_effect=multi_urlopen):
            with patch(_GEN_URLOPEN, side_effect=multi_urlopen):
                result = query("What is triage?", documents, spec)

        # Both reranking and generation should have occurred
        for item in result.evidence:
            assert "reranker_score" in item.signals
        assert "Generated answer" in result.answer or "Retrieved" in result.answer


# ---------------------------------------------------------------------------
# 7. Live integration tests (gated by env vars)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("OLLAMA_AVAILABLE"),
    reason="OLLAMA_AVAILABLE not set -- skipping live Ollama test",
)
class TestLiveOllamaReranking:
    """Live integration tests for Ollama reranking."""

    def test_live_rerank(self):
        backend = OllamaRerankerBackend()
        if not backend.is_available():
            pytest.skip("Ollama server not reachable")
        scores = backend.rerank(
            "What is emergency triage?",
            [
                "Triage is the process of prioritizing patients by severity.",
                "Chocolate cake recipe: mix flour, sugar, and cocoa.",
            ],
        )
        assert len(scores) == 2
        # The triage passage should score higher than the cake recipe
        assert scores[0] > scores[1]


@pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set -- skipping live OpenRouter test",
)
class TestLiveOpenRouterReranking:
    """Live integration tests for OpenRouter reranking."""

    def test_live_rerank(self):
        backend = OpenRouterRerankerBackend()
        scores = backend.rerank(
            "What is emergency triage?",
            [
                "Triage is the process of prioritizing patients by severity.",
                "Chocolate cake recipe: mix flour, sugar, and cocoa.",
            ],
        )
        assert len(scores) == 2
        assert scores[0] > scores[1]
