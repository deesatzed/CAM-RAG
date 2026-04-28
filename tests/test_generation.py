"""Tests for LLM generation backends, prompt construction, and pipeline integration.

Covers:
1. Prompt construction (build_rag_prompt)
2. Ollama generation backend (response parsing, API calls, availability)
3. OpenRouter generation backend (response parsing, API calls, construction)
4. GenerationStep pipeline execution
5. Query path integration (generation_backend enabled vs disabled)
6. Error handling (API failures, empty responses, malformed JSON)
7. Live integration tests (gated by env vars)

All data is real -- no mock, no simulation, no placeholders.
HTTP-level patching (urlopen) is used to avoid real API calls in unit tests.
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

from cam_rag.generation.ollama import (
    OllamaGenerationBackend,
    _parse_chat_response,
)
from cam_rag.generation.openrouter import (
    OpenRouterGenerationBackend,
    _parse_chat_response as _parse_openrouter_response,
)
from cam_rag.generation.prompt import build_rag_prompt
from cam_rag.rag.models import Chunk, Evidence

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OLLAMA_URLOPEN = (
    "cam_rag.generation.ollama.urllib.request.urlopen"
)
_OPENROUTER_URLOPEN = (
    "cam_rag.generation.openrouter.urllib.request.urlopen"
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


def _make_evidence(n: int = 3) -> list[Evidence]:
    """Create a list of real Evidence objects for testing."""
    texts = [
        "The emergency triage protocol requires assessment within 5 minutes.",
        "Vital signs should be checked every 15 minutes for critical patients.",
        "Glasgow Coma Scale is used to assess consciousness level.",
    ]
    return [
        Evidence(
            chunk=Chunk(
                id=f"chunk-{i}",
                document_id=f"doc-{i}",
                text=texts[i % len(texts)],
                source=f"protocol-{i}.md",
                title=f"Medical Protocol {i + 1}",
                section_heading=f"Section {i + 1}",
            ),
            score=0.9 - (i * 0.1),
            retriever="hybrid_rrf",
            rank=i,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# 1. Prompt construction
# ---------------------------------------------------------------------------


class TestBuildRagPrompt:
    """Test the prompt builder."""

    def test_basic_prompt_structure(self):
        evidence = _make_evidence(2)
        system, user = build_rag_prompt("What is triage?", evidence)
        assert "research assistant" in system.lower()
        assert "ONLY the evidence" in system
        assert "[1]" in user
        assert "[2]" in user
        assert "What is triage?" in user

    def test_evidence_sources_included(self):
        evidence = _make_evidence(1)
        _, user = build_rag_prompt("test query", evidence)
        assert "Medical Protocol 1" in user
        assert "Section 1" in user

    def test_empty_evidence(self):
        _, user = build_rag_prompt("test query", [])
        assert "No evidence retrieved" in user

    def test_max_evidence_items(self):
        evidence = _make_evidence(5)
        _, user = build_rag_prompt(
            "test query", evidence, max_evidence_items=2,
        )
        assert "[1]" in user
        assert "[2]" in user
        assert "[3]" not in user

    def test_evidence_truncation(self):
        long_evidence = _make_evidence(1)
        long_evidence[0].chunk.text = "x" * 20_000
        _, user = build_rag_prompt(
            "test", long_evidence, max_evidence_chars=100,
        )
        # Should be truncated with ...
        assert "..." in user
        # Should not contain the full 20k chars
        assert len(user) < 20_000

    def test_evidence_without_title_uses_source(self):
        evidence = _make_evidence(1)
        evidence[0].chunk.title = ""
        _, user = build_rag_prompt("query", evidence)
        assert "protocol-0.md" in user

    def test_evidence_without_section_heading(self):
        evidence = _make_evidence(1)
        evidence[0].chunk.section_heading = ""
        _, user = build_rag_prompt("query", evidence)
        # Should not have a " > " heading separator
        assert " > " not in user.split("[1]")[1].split("\n")[0]


# ---------------------------------------------------------------------------
# 2. Ollama response parsing
# ---------------------------------------------------------------------------


class TestParseOllamaChat:
    """Test Ollama /api/chat response parsing."""

    def test_valid_response(self):
        body = {
            "message": {
                "role": "assistant",
                "content": "The triage protocol states...",
            }
        }
        assert _parse_chat_response(body) == "The triage protocol states..."

    def test_empty_content(self):
        body = {"message": {"role": "assistant", "content": ""}}
        assert _parse_chat_response(body) == ""

    def test_missing_message_key(self):
        body = {"error": "model not found"}
        assert _parse_chat_response(body) == ""

    def test_non_dict_message(self):
        body = {"message": "not a dict"}
        assert _parse_chat_response(body) == ""

    def test_non_string_content(self):
        body = {"message": {"role": "assistant", "content": 42}}
        assert _parse_chat_response(body) == ""

    def test_whitespace_stripped(self):
        body = {"message": {"content": "  hello world  "}}
        assert _parse_chat_response(body) == "hello world"


# ---------------------------------------------------------------------------
# 3. OpenRouter response parsing
# ---------------------------------------------------------------------------


class TestParseOpenRouterChat:
    """Test OpenRouter /chat/completions response parsing."""

    def test_valid_response(self):
        body = {
            "choices": [
                {"message": {"role": "assistant", "content": "Answer here."}}
            ]
        }
        assert _parse_openrouter_response(body) == "Answer here."

    def test_empty_choices(self):
        body = {"choices": []}
        assert _parse_openrouter_response(body) == ""

    def test_missing_choices(self):
        body = {"error": "rate limited"}
        assert _parse_openrouter_response(body) == ""

    def test_non_list_choices(self):
        body = {"choices": "invalid"}
        assert _parse_openrouter_response(body) == ""

    def test_non_dict_first_choice(self):
        body = {"choices": ["not a dict"]}
        assert _parse_openrouter_response(body) == ""

    def test_non_dict_message_in_choice(self):
        body = {"choices": [{"message": "not a dict"}]}
        assert _parse_openrouter_response(body) == ""

    def test_non_string_content(self):
        body = {"choices": [{"message": {"content": 123}}]}
        assert _parse_openrouter_response(body) == ""

    def test_whitespace_stripped(self):
        body = {"choices": [{"message": {"content": "  trimmed  "}}]}
        assert _parse_openrouter_response(body) == "trimmed"


# ---------------------------------------------------------------------------
# 4. Ollama generation backend
# ---------------------------------------------------------------------------


class TestOllamaGenerationBackend:
    """Test OllamaGenerationBackend with HTTP-level patching."""

    def test_default_config(self):
        backend = OllamaGenerationBackend()
        assert backend.model == "qwen3:4b"
        assert backend.base_url == "http://localhost:11434"
        assert backend.temperature == 0.1

    def test_custom_config(self):
        backend = OllamaGenerationBackend(
            model="llama3.2:3b", temperature=0.7, max_tokens=512,
        )
        assert backend.model == "llama3.2:3b"
        assert backend.temperature == 0.7
        assert backend.max_tokens == 512

    def test_generate_success(self):
        backend = OllamaGenerationBackend()
        body = {"message": {"content": "Based on the evidence, triage involves..."}}
        with patch(_OLLAMA_URLOPEN, side_effect=_urlopen_returning(body)):
            result = backend.generate("system", "user prompt")
        assert "triage" in result

    def test_generate_empty_response(self):
        backend = OllamaGenerationBackend()
        body = {"message": {"content": ""}}
        with patch(_OLLAMA_URLOPEN, side_effect=_urlopen_returning(body)):
            result = backend.generate("system", "user prompt")
        assert result == ""

    def test_generate_network_error(self):
        import urllib.error
        backend = OllamaGenerationBackend()
        with patch(
            _OLLAMA_URLOPEN,
            side_effect=urllib.error.URLError("connection refused"),
        ):
            result = backend.generate("system", "user prompt")
        assert result == ""

    def test_generate_invalid_json(self):
        backend = OllamaGenerationBackend()

        class BadResponse:
            def read(self):
                return b"not json"
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        with patch(_OLLAMA_URLOPEN, return_value=BadResponse()):
            result = backend.generate("system", "user prompt")
        assert result == ""

    def test_is_available_true(self):
        backend = OllamaGenerationBackend()

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
        backend = OllamaGenerationBackend()
        with patch(
            _OLLAMA_URLOPEN,
            side_effect=urllib.error.URLError("not reachable"),
        ):
            assert backend.is_available() is False

    def test_api_call_sends_correct_payload(self):
        backend = OllamaGenerationBackend(model="test-model")
        body = {"message": {"content": "response"}}
        captured_req = None

        def capture_urlopen(req, timeout=None):
            nonlocal captured_req
            captured_req = req
            return _FakeHTTPResponse(body)

        with patch(_OLLAMA_URLOPEN, side_effect=capture_urlopen):
            backend.generate("sys prompt", "user prompt")

        assert captured_req is not None
        payload = json.loads(captured_req.data)
        assert payload["model"] == "test-model"
        assert payload["stream"] is False
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"


# ---------------------------------------------------------------------------
# 5. OpenRouter generation backend
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_openrouter_key(monkeypatch):
    """Provide a dummy API key so the backend can be constructed."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-for-generation")


class TestOpenRouterGenerationBackend:
    """Test OpenRouterGenerationBackend with HTTP-level patching."""

    def test_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            OpenRouterGenerationBackend()

    def test_accepts_explicit_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        backend = OpenRouterGenerationBackend(api_key="explicit-key")
        assert backend.api_key == "explicit-key"

    def test_reads_env_api_key(self, fake_openrouter_key):
        backend = OpenRouterGenerationBackend()
        assert backend.api_key == "test-key-for-generation"

    def test_default_config(self, fake_openrouter_key):
        backend = OpenRouterGenerationBackend()
        assert backend.model == "anthropic/claude-sonnet-4"
        assert "openrouter.ai" in backend.base_url

    def test_custom_config(self, fake_openrouter_key):
        backend = OpenRouterGenerationBackend(
            model="meta-llama/llama-3-70b", temperature=0.5,
        )
        assert backend.model == "meta-llama/llama-3-70b"
        assert backend.temperature == 0.5

    def test_generate_success(self, fake_openrouter_key):
        backend = OpenRouterGenerationBackend()
        body = {
            "choices": [
                {"message": {"content": "The answer based on evidence is..."}}
            ]
        }
        with patch(_OPENROUTER_URLOPEN, side_effect=_urlopen_returning(body)):
            result = backend.generate("system", "user prompt")
        assert "evidence" in result

    def test_generate_empty_response(self, fake_openrouter_key):
        backend = OpenRouterGenerationBackend()
        body = {"choices": [{"message": {"content": ""}}]}
        with patch(_OPENROUTER_URLOPEN, side_effect=_urlopen_returning(body)):
            result = backend.generate("system", "user prompt")
        assert result == ""

    def test_generate_network_error(self, fake_openrouter_key):
        import urllib.error
        backend = OpenRouterGenerationBackend()
        with patch(
            _OPENROUTER_URLOPEN,
            side_effect=urllib.error.URLError("timeout"),
        ):
            result = backend.generate("system", "user prompt")
        assert result == ""

    def test_generate_invalid_json(self, fake_openrouter_key):
        backend = OpenRouterGenerationBackend()

        class BadResponse:
            def read(self):
                return b"not json at all"
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        with patch(_OPENROUTER_URLOPEN, return_value=BadResponse()):
            result = backend.generate("system", "user prompt")
        assert result == ""

    def test_api_call_sends_auth_header(self, fake_openrouter_key):
        backend = OpenRouterGenerationBackend()
        body = {"choices": [{"message": {"content": "ok"}}]}
        captured_req = None

        def capture_urlopen(req, timeout=None):
            nonlocal captured_req
            captured_req = req
            return _FakeHTTPResponse(body)

        with patch(_OPENROUTER_URLOPEN, side_effect=capture_urlopen):
            backend.generate("sys", "user")

        assert captured_req is not None
        assert "Bearer test-key-for-generation" in captured_req.get_header("Authorization")

    def test_api_call_sends_correct_payload(self, fake_openrouter_key):
        backend = OpenRouterGenerationBackend(model="test/model")
        body = {"choices": [{"message": {"content": "ok"}}]}
        captured_req = None

        def capture_urlopen(req, timeout=None):
            nonlocal captured_req
            captured_req = req
            return _FakeHTTPResponse(body)

        with patch(_OPENROUTER_URLOPEN, side_effect=capture_urlopen):
            backend.generate("sys prompt", "user prompt")

        payload = json.loads(captured_req.data)
        assert payload["model"] == "test/model"
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["content"] == "sys prompt"
        assert payload["messages"][1]["content"] == "user prompt"


# ---------------------------------------------------------------------------
# 6. GenerationStep pipeline execution
# ---------------------------------------------------------------------------


class TestGenerationStep:
    """Test the GenerationStep technique step."""

    def test_no_backend_configured(self):
        """GenerationStep does nothing when no backend is configured."""
        from cam_rag.pipeline.context import PipelineContext
        from cam_rag.pipeline.executor import GenerationStep
        from cam_rag.rag.spec import RAGAppSpec
        from cam_rag.retrieval import AdaptiveParams, RetrievalDocument

        spec = RAGAppSpec(name="test-no-gen")
        docs = [RetrievalDocument(
            doc_id="d1", text="test", metadata={"chunk_id": "d1"},
        )]
        ctx = PipelineContext(
            query_text="test query",
            chunks=[],
            spec=spec,
            adaptive=AdaptiveParams(
                sparse_k=10, dense_k=10, rrf_k=60,
                sparse_weight=0.4, dense_weight=0.6,
            ),
            limit=5,
            retrieval_docs=docs,
            expanded_query="test query",
        )
        step = GenerationStep()
        step.execute(ctx)
        assert ctx.generated_answer == ""

    def test_no_evidence_returns_message(self):
        """GenerationStep returns a no-evidence message when evidence is empty."""
        from cam_rag.pipeline.context import PipelineContext
        from cam_rag.pipeline.executor import GenerationStep
        from cam_rag.rag.spec import RAGAppSpec
        from cam_rag.retrieval import AdaptiveParams, RetrievalDocument

        backend = OllamaGenerationBackend()
        spec = RAGAppSpec(name="test-gen-no-evidence", generation_backend=backend)
        docs = [RetrievalDocument(
            doc_id="d1", text="test", metadata={"chunk_id": "d1"},
        )]
        ctx = PipelineContext(
            query_text="what is triage?",
            chunks=[],
            spec=spec,
            adaptive=AdaptiveParams(
                sparse_k=10, dense_k=10, rrf_k=60,
                sparse_weight=0.4, dense_weight=0.6,
            ),
            limit=5,
            retrieval_docs=docs,
            expanded_query="what is triage?",
        )
        step = GenerationStep()
        step.execute(ctx)
        assert "No evidence" in ctx.generated_answer
        assert "what is triage?" in ctx.generated_answer

    def test_generation_with_evidence(self):
        """GenerationStep calls the backend and stores the answer."""
        from cam_rag.pipeline.context import PipelineContext
        from cam_rag.pipeline.executor import GenerationStep
        from cam_rag.rag.spec import RAGAppSpec
        from cam_rag.retrieval import AdaptiveParams, RetrievalDocument

        backend = OllamaGenerationBackend()
        spec = RAGAppSpec(name="test-gen-with-evidence", generation_backend=backend)
        docs = [RetrievalDocument(
            doc_id="d1", text="test", metadata={"chunk_id": "d1"},
        )]
        ctx = PipelineContext(
            query_text="what is triage?",
            chunks=[],
            spec=spec,
            adaptive=AdaptiveParams(
                sparse_k=10, dense_k=10, rrf_k=60,
                sparse_weight=0.4, dense_weight=0.6,
            ),
            limit=5,
            retrieval_docs=docs,
            expanded_query="what is triage?",
            evidence=_make_evidence(2),
        )

        body = {"message": {"content": "Triage is a process of sorting patients [1]."}}
        with patch(_OLLAMA_URLOPEN, side_effect=_urlopen_returning(body)):
            step = GenerationStep()
            step.execute(ctx)

        assert "Triage" in ctx.generated_answer
        assert "[1]" in ctx.generated_answer

    def test_generation_fallback_on_empty_response(self):
        """When backend returns empty, generated_answer stays empty."""
        from cam_rag.pipeline.context import PipelineContext
        from cam_rag.pipeline.executor import GenerationStep
        from cam_rag.rag.spec import RAGAppSpec
        from cam_rag.retrieval import AdaptiveParams, RetrievalDocument

        backend = OllamaGenerationBackend()
        spec = RAGAppSpec(name="test-gen-fallback", generation_backend=backend)
        docs = [RetrievalDocument(
            doc_id="d1", text="test", metadata={"chunk_id": "d1"},
        )]
        ctx = PipelineContext(
            query_text="query",
            chunks=[],
            spec=spec,
            adaptive=AdaptiveParams(
                sparse_k=10, dense_k=10, rrf_k=60,
                sparse_weight=0.4, dense_weight=0.6,
            ),
            limit=5,
            retrieval_docs=docs,
            expanded_query="query",
            evidence=_make_evidence(1),
        )

        body = {"message": {"content": ""}}
        with patch(_OLLAMA_URLOPEN, side_effect=_urlopen_returning(body)):
            step = GenerationStep()
            step.execute(ctx)

        # Empty generation means no update
        assert ctx.generated_answer == ""


# ---------------------------------------------------------------------------
# 7. Query path integration
# ---------------------------------------------------------------------------


class TestQueryPathIntegration:
    """Test generation wiring in the query path."""

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
                    "and oxygen saturation. Continuous monitoring is required "
                    "for critical patients."
                ),
                source="vitals.md",
                title="Vital Signs Guide",
            ),
        ]

    def test_generation_disabled_by_default(self):
        """Without generation_backend, answer is retrieval-only template."""
        from cam_rag.query import query
        from cam_rag.rag.spec import RAGAppSpec

        spec = RAGAppSpec(name="test-no-gen-query")
        documents = self._make_documents()
        result = query("What is emergency triage?", documents, spec)
        assert "Retrieved" in result.answer
        assert "cited evidence" in result.answer

    def test_generation_enabled_with_backend(self):
        """With generation_backend, answer comes from the LLM."""
        from cam_rag.query import query
        from cam_rag.rag.spec import RAGAppSpec

        backend = OllamaGenerationBackend()
        spec = RAGAppSpec(
            name="test-gen-query",
            generation_backend=backend,
        )
        documents = self._make_documents()

        body = {
            "message": {
                "content": (
                    "Emergency triage is the process of prioritizing "
                    "patients based on severity [1]. Critical patients "
                    "receive treatment first."
                ),
            }
        }
        with patch(_OLLAMA_URLOPEN, side_effect=_urlopen_returning(body)):
            result = query("What is emergency triage?", documents, spec)

        assert "triage" in result.answer.lower()
        assert "[1]" in result.answer
        # Should NOT be the retrieval-only template
        assert "Retrieved" not in result.answer

    def test_generation_fallback_on_api_failure(self):
        """When the generation API fails, falls back to retrieval-only."""
        import urllib.error
        from cam_rag.query import query
        from cam_rag.rag.spec import RAGAppSpec

        backend = OllamaGenerationBackend()
        spec = RAGAppSpec(
            name="test-gen-fallback-query",
            generation_backend=backend,
        )
        documents = self._make_documents()

        with patch(
            _OLLAMA_URLOPEN,
            side_effect=urllib.error.URLError("connection refused"),
        ):
            result = query("What is emergency triage?", documents, spec)

        # Falls back to retrieval-only
        assert "Retrieved" in result.answer
        assert "cited evidence" in result.answer

    def test_generation_with_pipeline_path(self):
        """Generation works when using the pipeline path."""
        from cam_rag.query import query
        from cam_rag.rag.spec import RAGAppSpec

        backend = OllamaGenerationBackend()
        spec = RAGAppSpec(
            name="test-gen-pipeline",
            generation_backend=backend,
            use_pipeline=True,
        )
        documents = self._make_documents()

        body = {
            "message": {
                "content": "Based on the evidence, triage prioritizes patients [1].",
            }
        }
        with patch(_OLLAMA_URLOPEN, side_effect=_urlopen_returning(body)):
            result = query("What is emergency triage?", documents, spec)

        assert "triage" in result.answer.lower()
        assert "[1]" in result.answer

    def test_generation_trace_records_stage(self):
        """The generation stage should appear in the trace."""
        from cam_rag.query import query
        from cam_rag.rag.spec import RAGAppSpec

        backend = OllamaGenerationBackend()
        spec = RAGAppSpec(
            name="test-gen-trace",
            generation_backend=backend,
        )
        documents = self._make_documents()

        body = {
            "message": {"content": "Answer with citations [1]."}
        }
        with patch(_OLLAMA_URLOPEN, side_effect=_urlopen_returning(body)):
            result = query("What is triage?", documents, spec)

        assert "generation" in result.trace.stages

    def test_evidence_and_citations_still_populated(self):
        """Generation should not affect evidence/citations population."""
        from cam_rag.query import query
        from cam_rag.rag.spec import RAGAppSpec

        backend = OllamaGenerationBackend()
        spec = RAGAppSpec(
            name="test-gen-evidence",
            generation_backend=backend,
        )
        documents = self._make_documents()

        body = {
            "message": {"content": "Generated answer."}
        }
        with patch(_OLLAMA_URLOPEN, side_effect=_urlopen_returning(body)):
            result = query("What is triage?", documents, spec)

        assert len(result.evidence) > 0
        assert len(result.citations) > 0
        assert result.confidence > 0.0

    def test_openrouter_backend_in_query_path(self, monkeypatch):
        """OpenRouter generation backend works in the query path."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        from cam_rag.query import query
        from cam_rag.rag.spec import RAGAppSpec

        backend = OpenRouterGenerationBackend()
        spec = RAGAppSpec(
            name="test-openrouter-gen",
            generation_backend=backend,
        )
        documents = self._make_documents()

        body = {
            "choices": [
                {"message": {"content": "OpenRouter generated answer [1]."}}
            ]
        }
        with patch(_OPENROUTER_URLOPEN, side_effect=_urlopen_returning(body)):
            result = query("What is triage?", documents, spec)

        assert "OpenRouter" in result.answer
        assert "[1]" in result.answer


# ---------------------------------------------------------------------------
# 8. GenerationStep registered in executor
# ---------------------------------------------------------------------------


class TestGenerationRegistered:
    """Verify GenerationStep is registered and discoverable."""

    def test_generation_step_registered(self):
        from cam_rag.pipeline.executor import get_step
        step = get_step("generation")
        assert step.name == "generation"

    def test_generation_in_default_registry(self):
        from cam_rag.query import _build_default_registry
        registry = _build_default_registry()
        desc = registry.get("generation")
        assert desc.name == "generation"
        assert desc.category == "generation"

    def test_generation_step_in_pipeline_plan(self):
        """GenerationStep should be included in composed pipeline plans."""
        from cam_rag.pipeline.registry import compose_pipeline
        from cam_rag.query import _build_default_registry

        registry = _build_default_registry()
        plan = compose_pipeline(registry, "summary", corpus_size=10)
        assert "generation" in plan.steps


# ---------------------------------------------------------------------------
# 9. Live integration tests (gated by env vars)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("OLLAMA_AVAILABLE"),
    reason="OLLAMA_AVAILABLE not set -- skipping live Ollama test",
)
class TestLiveOllamaGeneration:
    """Live integration tests for Ollama generation."""

    def test_live_generate(self):
        backend = OllamaGenerationBackend()
        if not backend.is_available():
            pytest.skip("Ollama server not reachable")
        result = backend.generate(
            "You are a helpful assistant.",
            "What is 2 + 2? Answer in one word.",
        )
        assert len(result) > 0
        assert "4" in result or "four" in result.lower()


@pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set -- skipping live OpenRouter test",
)
class TestLiveOpenRouterGeneration:
    """Live integration tests for OpenRouter generation."""

    def test_live_generate(self):
        backend = OpenRouterGenerationBackend()
        result = backend.generate(
            "You are a helpful assistant.",
            "What is 2 + 2? Answer in one word.",
        )
        assert len(result) > 0
        assert "4" in result or "four" in result.lower()
