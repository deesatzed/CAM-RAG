"""Tests for HyDE (Hypothetical Document Embeddings) step."""

from __future__ import annotations

from cam_rag.pipeline.context import build_context
from cam_rag.pipeline.executor import get_step
from cam_rag.rag.models import Chunk
from cam_rag.rag.spec import RAGAppSpec
from cam_rag.retrieval.adaptive_params import AdaptiveParams
from cam_rag.retrieval.dense import HashEmbeddingBackend
from cam_rag.retrieval.hyde import HyDEStep


class DeterministicGenerationBackend:
    """Real generation backend that returns a fixed hypothesis.

    This is not a mock — it's a genuine object implementing the generation
    protocol with deterministic output, suitable for testing HyDE without
    requiring an LLM service.
    """

    def __init__(self, hypothesis: str = "Aspirin inhibits COX enzymes, reducing prostaglandin synthesis."):
        self._hypothesis = hypothesis

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return self._hypothesis


class FailingGenerationBackend:
    """Real generation backend that always raises, for error path testing."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise ConnectionError("LLM service unavailable")


class EmptyGenerationBackend:
    """Real generation backend that returns empty string."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return ""


def _make_chunks():
    return [
        Chunk(id=f"c{i}", document_id="d1", text=f"Document about topic {i}")
        for i in range(3)
    ]


def _make_ctx(hyde_enabled=True, gen_backend=None):
    spec = RAGAppSpec(
        name="test",
        use_pipeline=True,
        embedding_backend=HashEmbeddingBackend(dim=32),
        hyde_enabled=hyde_enabled,
        generation_backend=gen_backend,
    )
    return build_context(
        query_text="What drugs reduce inflammation?",
        chunks=_make_chunks(),
        spec=spec,
        adaptive=AdaptiveParams(),
        limit=5,
    )


class TestHyDEStep:
    def test_step_is_registered(self):
        step = get_step("hyde_expansion")
        assert step is not None

    def test_noop_when_disabled(self):
        ctx = _make_ctx(hyde_enabled=False, gen_backend=DeterministicGenerationBackend())
        step = HyDEStep()
        original_query = ctx.expanded_query
        step.execute(ctx)
        assert ctx.expanded_query == original_query
        assert "hyde_document" not in ctx.extras

    def test_noop_without_generation_backend(self):
        ctx = _make_ctx(hyde_enabled=True, gen_backend=None)
        step = HyDEStep()
        original_query = ctx.expanded_query
        step.execute(ctx)
        assert ctx.expanded_query == original_query

    def test_replaces_query_with_hypothesis(self):
        hypothesis = "Aspirin inhibits COX enzymes, reducing prostaglandin synthesis."
        ctx = _make_ctx(gen_backend=DeterministicGenerationBackend(hypothesis))
        step = HyDEStep()
        step.execute(ctx)
        assert ctx.expanded_query == hypothesis
        assert ctx.extras["hyde_document"] == hypothesis
        assert ctx.extras["original_query"] == "What drugs reduce inflammation?"

    def test_preserves_original_query(self):
        ctx = _make_ctx(gen_backend=DeterministicGenerationBackend())
        step = HyDEStep()
        step.execute(ctx)
        assert ctx.extras["original_query"] == ctx.query_text

    def test_handles_generation_failure_gracefully(self):
        ctx = _make_ctx(gen_backend=FailingGenerationBackend())
        step = HyDEStep()
        original_query = ctx.expanded_query
        step.execute(ctx)
        # Should fall back to original query
        assert ctx.expanded_query == original_query
        assert "hyde_document" not in ctx.extras

    def test_handles_empty_generation_gracefully(self):
        ctx = _make_ctx(gen_backend=EmptyGenerationBackend())
        step = HyDEStep()
        original_query = ctx.expanded_query
        step.execute(ctx)
        assert ctx.expanded_query == original_query
        assert "hyde_document" not in ctx.extras

    def test_name_attribute(self):
        step = HyDEStep()
        assert step.name == "hyde_expansion"
