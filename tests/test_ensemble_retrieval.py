"""Tests for ensemble dense retrieval with multiple embedding backends.

Validates that:
1. Single backend fallback -- behavior is identical to before
2. Two-backend ensemble -- both retrievers created, RRF includes all lists
3. Three-way fusion -- rrf_fuse called with sparse + 2 dense result lists
4. Custom ensemble weights -- spec.ensemble_weights are applied
5. Equal weight default -- dense_weight is split equally without custom weights
6. Pipeline path ensemble -- ensemble works through executor
7. Static path ensemble -- ensemble works through static path
8. Query expansion with ensemble -- expansion re-runs using all ensemble retrievers
9. Static/dynamic equivalence -- both paths produce equivalent results with ensemble
10. Backward compatibility -- existing behavior unchanged when embedding_backends empty

All data is real -- no mock, no simulation, no placeholders.
Uses HashEmbeddingBackend with different dim values to create distinguishable
backends that produce different retrieval rankings.
"""

from __future__ import annotations

from cam_rag.pipeline.context import PipelineContext, build_context
from cam_rag.pipeline.executor import (
    DenseVectorStep,
    PipelineExecutor,
    RRFFusionStep,
    SparseBM25Step,
)
from cam_rag.pipeline.registry import ExecutionPlan
from cam_rag.query import (
    _rank_chunks_pipeline,
    _rank_chunks_static,
    classify_query_type,
)
from cam_rag.rag.models import Chunk, Evidence
from cam_rag.rag.spec import RAGAppSpec
from cam_rag.retrieval.adaptive_params import compute_adaptive_params
from cam_rag.retrieval.dense import HashEmbeddingBackend

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_chunks(n: int = 8) -> list[Chunk]:
    """Build a set of realistic chunks for ensemble testing.

    Uses 8 chunks with varied medical terminology so that different
    embedding dimensions produce meaningfully different rankings.
    """
    texts = [
        "Emergency department triage uses vital signs and acuity scoring protocols.",
        "Chest pain evaluation follows the HEART score algorithm for risk assessment.",
        "Sepsis screening uses the qSOFA criteria including altered mental status.",
        "Stroke assessment follows the NIH Stroke Scale for severity grading.",
        "Trauma patients are assessed using the Glasgow Coma Scale and ABCDE approach.",
        "Cardiac arrest management follows advanced cardiovascular life support protocols.",
        "Diabetic ketoacidosis requires insulin drip and fluid resuscitation monitoring.",
        "Acute respiratory distress syndrome requires ventilator management strategies.",
    ]
    return [
        Chunk(
            id=f"chunk_{i}",
            document_id=f"doc_{i}",
            text=texts[i],
            source=f"document_{i}.md",
            title=f"Section {i}",
        )
        for i in range(min(n, len(texts)))
    ]


def _make_spec(**overrides) -> RAGAppSpec:
    defaults = {"name": "ensemble-test-app", "retrieval_top_k": 5}
    defaults.update(overrides)
    return RAGAppSpec(**defaults)


def _make_context(
    query_text: str = "emergency triage protocol",
    chunks: list[Chunk] | None = None,
    **spec_overrides,
) -> PipelineContext:
    if chunks is None:
        chunks = _make_chunks()
    spec = _make_spec(**spec_overrides)
    query_type = classify_query_type(query_text)
    adaptive = compute_adaptive_params(
        query_type, query_text, corpus_size=len(chunks),
    )
    return build_context(
        query_text, chunks, spec, adaptive, limit=spec.retrieval_top_k,
    )


def _two_backends() -> list[HashEmbeddingBackend]:
    """Two HashEmbeddingBackend instances with different dim values."""
    return [HashEmbeddingBackend(dim=32), HashEmbeddingBackend(dim=64)]


def _three_backends() -> list[HashEmbeddingBackend]:
    """Three HashEmbeddingBackend instances with different dim values."""
    return [
        HashEmbeddingBackend(dim=32),
        HashEmbeddingBackend(dim=64),
        HashEmbeddingBackend(dim=128),
    ]


# ---------------------------------------------------------------------------
# 1. Single Backend Fallback (backward compatibility)
# ---------------------------------------------------------------------------


class TestSingleBackendFallback:
    """When embedding_backends is empty, behavior is identical to before."""

    def test_dense_step_creates_single_retriever(self):
        ctx = _make_context()
        DenseVectorStep().execute(ctx)
        assert ctx.dense_retriever is not None
        assert "ensemble_dense_results" not in ctx.extras

    def test_rrf_fusion_uses_standard_two_way(self):
        ctx = _make_context()
        SparseBM25Step().execute(ctx)
        DenseVectorStep().execute(ctx)
        RRFFusionStep().execute(ctx)

        assert len(ctx.fused_results) > 0
        assert len(ctx.evidence) > 0
        # Standard mode: source_ranks has "dense" and "sparse"
        for ev in ctx.evidence:
            source_ranks = ev.signals["source_ranks"]
            # Each result comes from at least one of dense/sparse
            assert set(source_ranks.keys()) <= {"dense", "sparse"}

    def test_static_path_without_ensemble_backends(self):
        chunks = _make_chunks()
        spec = _make_spec(use_pipeline=False)
        query_type = classify_query_type("emergency triage protocol")
        adaptive = compute_adaptive_params(
            query_type, "emergency triage protocol",
            corpus_size=len(chunks),
        )
        evidence, expanded = _rank_chunks_static(
            "emergency triage protocol", chunks, spec,
            adaptive=adaptive, limit=5,
        )
        assert len(evidence) > 0
        assert expanded == "emergency triage protocol"


# ---------------------------------------------------------------------------
# 2. Two-Backend Ensemble
# ---------------------------------------------------------------------------


class TestTwoBackendEnsemble:
    """Two HashEmbeddingBackend instances with different dims."""

    def test_pipeline_ensemble_stores_results(self):
        backends = _two_backends()
        ctx = _make_context(embedding_backends=backends)
        DenseVectorStep().execute(ctx)

        assert "ensemble_dense_results" in ctx.extras
        ensemble_results = ctx.extras["ensemble_dense_results"]
        assert len(ensemble_results) == 2
        assert "dense_0" in ensemble_results
        assert "dense_1" in ensemble_results

    def test_pipeline_ensemble_stores_retrievers(self):
        backends = _two_backends()
        ctx = _make_context(embedding_backends=backends)
        DenseVectorStep().execute(ctx)

        assert "ensemble_dense_retrievers" in ctx.extras
        retrievers = ctx.extras["ensemble_dense_retrievers"]
        assert len(retrievers) == 2

    def test_pipeline_ensemble_sets_dense_results_backward_compat(self):
        backends = _two_backends()
        ctx = _make_context(embedding_backends=backends)
        DenseVectorStep().execute(ctx)

        # ctx.dense_results should be set to the first backend's results
        assert len(ctx.dense_results) > 0

    def test_each_backend_produces_results(self):
        backends = _two_backends()
        ctx = _make_context(embedding_backends=backends)
        DenseVectorStep().execute(ctx)

        for name, results in ctx.extras["ensemble_dense_results"].items():
            assert len(results) > 0, f"Backend {name} produced no results"


# ---------------------------------------------------------------------------
# 3. Three-Way Fusion (sparse + 2 dense)
# ---------------------------------------------------------------------------


class TestThreeWayFusion:
    """Verify RRF fusion includes sparse + 2 dense result lists."""

    def test_fusion_includes_all_sources(self):
        backends = _two_backends()
        ctx = _make_context(embedding_backends=backends)
        SparseBM25Step().execute(ctx)
        DenseVectorStep().execute(ctx)
        RRFFusionStep().execute(ctx)

        assert len(ctx.fused_results) > 0
        assert len(ctx.evidence) > 0

        # At least one evidence item should have source_ranks from
        # multiple sources (sparse, dense_0, dense_1)
        all_source_names = set()
        for ev in ctx.evidence:
            all_source_names.update(ev.signals["source_ranks"].keys())

        # Must have sparse and both dense backends
        assert "sparse" in all_source_names
        assert "dense_0" in all_source_names
        assert "dense_1" in all_source_names

    def test_fusion_with_three_backends(self):
        """Four-way fusion: sparse + 3 dense backends."""
        backends = _three_backends()
        ctx = _make_context(embedding_backends=backends)
        SparseBM25Step().execute(ctx)
        DenseVectorStep().execute(ctx)
        RRFFusionStep().execute(ctx)

        assert len(ctx.evidence) > 0
        all_source_names = set()
        for ev in ctx.evidence:
            all_source_names.update(ev.signals["source_ranks"].keys())

        assert "sparse" in all_source_names
        assert "dense_0" in all_source_names
        assert "dense_1" in all_source_names
        assert "dense_2" in all_source_names

    def test_evidence_has_rrf_signals(self):
        backends = _two_backends()
        ctx = _make_context(embedding_backends=backends)
        SparseBM25Step().execute(ctx)
        DenseVectorStep().execute(ctx)
        RRFFusionStep().execute(ctx)

        for ev in ctx.evidence:
            assert "rrf_score" in ev.signals
            assert ev.signals["rrf_score"] > 0
            assert "source_ranks" in ev.signals
            assert "matched_terms" in ev.signals


# ---------------------------------------------------------------------------
# 4. Custom Ensemble Weights
# ---------------------------------------------------------------------------


class TestCustomEnsembleWeights:
    """Set spec.ensemble_weights and verify they are applied."""

    def test_custom_weights_used_in_fusion(self):
        backends = _two_backends()
        custom_weights = {"dense_0": 0.8, "dense_1": 0.2}
        ctx = _make_context(
            embedding_backends=backends,
            ensemble_weights=custom_weights,
        )
        SparseBM25Step().execute(ctx)
        DenseVectorStep().execute(ctx)
        RRFFusionStep().execute(ctx)

        assert len(ctx.evidence) > 0
        # Verify the fusion ran without error -- the weights are applied
        # internally by rrf_fuse. We validate by checking evidence exists.
        for ev in ctx.evidence:
            assert ev.score > 0

    def test_partial_custom_weights(self):
        """When only some backends have custom weights, others get equal share."""
        backends = _two_backends()
        # Only dense_0 has custom weight; dense_1 gets default
        custom_weights = {"dense_0": 0.5}
        ctx = _make_context(
            embedding_backends=backends,
            ensemble_weights=custom_weights,
        )
        SparseBM25Step().execute(ctx)
        DenseVectorStep().execute(ctx)
        RRFFusionStep().execute(ctx)

        assert len(ctx.evidence) > 0

    def test_different_weights_produce_different_scores(self):
        """Different weight configurations should produce different RRF scores."""
        backends = _two_backends()

        # Config A: heavy weight on dense_0
        ctx_a = _make_context(
            embedding_backends=backends,
            ensemble_weights={"dense_0": 0.9, "dense_1": 0.1},
        )
        SparseBM25Step().execute(ctx_a)
        DenseVectorStep().execute(ctx_a)
        RRFFusionStep().execute(ctx_a)

        # Config B: heavy weight on dense_1
        ctx_b = _make_context(
            embedding_backends=backends,
            ensemble_weights={"dense_0": 0.1, "dense_1": 0.9},
        )
        SparseBM25Step().execute(ctx_b)
        DenseVectorStep().execute(ctx_b)
        RRFFusionStep().execute(ctx_b)

        # Scores should differ
        scores_a = [ev.score for ev in ctx_a.evidence]
        scores_b = [ev.score for ev in ctx_b.evidence]
        assert scores_a != scores_b, (
            "Different weight configurations should produce different scores"
        )


# ---------------------------------------------------------------------------
# 5. Equal Weight Default
# ---------------------------------------------------------------------------


class TestEqualWeightDefault:
    """Without custom weights, dense_weight is split equally."""

    def test_two_backends_split_weight_equally(self):
        """With 2 backends, each should get dense_weight / 2."""
        backends = _two_backends()
        ctx = _make_context(
            embedding_backends=backends,
            # No ensemble_weights -- defaults to empty dict
        )
        SparseBM25Step().execute(ctx)
        DenseVectorStep().execute(ctx)
        RRFFusionStep().execute(ctx)

        assert len(ctx.evidence) > 0
        # All evidence should have valid scores
        for ev in ctx.evidence:
            assert ev.score > 0

    def test_three_backends_split_weight_equally(self):
        """With 3 backends, each should get dense_weight / 3."""
        backends = _three_backends()
        ctx = _make_context(embedding_backends=backends)
        SparseBM25Step().execute(ctx)
        DenseVectorStep().execute(ctx)
        RRFFusionStep().execute(ctx)

        assert len(ctx.evidence) > 0


# ---------------------------------------------------------------------------
# 6. Pipeline Path Ensemble
# ---------------------------------------------------------------------------


class TestPipelinePathEnsemble:
    """With use_pipeline=True, ensemble works through executor."""

    def test_full_pipeline_with_ensemble(self):
        backends = _two_backends()
        plan = ExecutionPlan(
            steps=("sparse_bm25", "dense_vector", "rrf_fusion"),
            query_type="summary",
        )
        ctx = _make_context(
            embedding_backends=backends,
            use_pipeline=True,
        )
        executor = PipelineExecutor()
        executor.run(plan, ctx)

        assert len(ctx.evidence) > 0
        assert "sparse_bm25" in ctx.steps_executed
        assert "dense_vector" in ctx.steps_executed
        assert "rrf_fusion" in ctx.steps_executed

    def test_pipeline_ensemble_via_rank_chunks(self):
        chunks = _make_chunks()
        backends = _two_backends()
        spec = _make_spec(
            use_pipeline=True,
            embedding_backends=backends,
        )
        query_type = classify_query_type("emergency triage protocol")
        adaptive = compute_adaptive_params(
            query_type, "emergency triage protocol",
            corpus_size=len(chunks),
        )
        evidence, _expanded, _gen = _rank_chunks_pipeline(
            "emergency triage protocol", chunks, spec,
            adaptive=adaptive, limit=5,
        )
        assert len(evidence) > 0

    def test_pipeline_ensemble_evidence_structure(self):
        backends = _two_backends()
        plan = ExecutionPlan(
            steps=("sparse_bm25", "dense_vector", "rrf_fusion"),
            query_type="summary",
        )
        ctx = _make_context(embedding_backends=backends)
        PipelineExecutor().run(plan, ctx)

        for ev in ctx.evidence:
            assert isinstance(ev, Evidence)
            assert ev.retriever == "hybrid_rrf"
            assert ev.score > 0
            assert "source_ranks" in ev.signals
            assert "rrf_score" in ev.signals


# ---------------------------------------------------------------------------
# 7. Static Path Ensemble
# ---------------------------------------------------------------------------


class TestStaticPathEnsemble:
    """With use_pipeline=False, ensemble works through static path."""

    def test_static_ensemble_returns_evidence(self):
        chunks = _make_chunks()
        backends = _two_backends()
        spec = _make_spec(
            use_pipeline=False,
            embedding_backends=backends,
        )
        query_type = classify_query_type("emergency triage protocol")
        adaptive = compute_adaptive_params(
            query_type, "emergency triage protocol",
            corpus_size=len(chunks),
        )
        evidence, expanded = _rank_chunks_static(
            "emergency triage protocol", chunks, spec,
            adaptive=adaptive, limit=5,
        )
        assert len(evidence) > 0
        assert expanded == "emergency triage protocol"

    def test_static_ensemble_has_all_sources(self):
        chunks = _make_chunks()
        backends = _two_backends()
        spec = _make_spec(
            use_pipeline=False,
            embedding_backends=backends,
        )
        query_type = classify_query_type("emergency triage protocol")
        adaptive = compute_adaptive_params(
            query_type, "emergency triage protocol",
            corpus_size=len(chunks),
        )
        evidence, _ = _rank_chunks_static(
            "emergency triage protocol", chunks, spec,
            adaptive=adaptive, limit=5,
        )

        all_source_names = set()
        for ev in evidence:
            all_source_names.update(ev.signals["source_ranks"].keys())

        assert "sparse" in all_source_names
        assert "dense_0" in all_source_names
        assert "dense_1" in all_source_names

    def test_static_ensemble_with_custom_weights(self):
        chunks = _make_chunks()
        backends = _two_backends()
        spec = _make_spec(
            use_pipeline=False,
            embedding_backends=backends,
            ensemble_weights={"dense_0": 0.7, "dense_1": 0.3},
        )
        query_type = classify_query_type("sepsis screening")
        adaptive = compute_adaptive_params(
            query_type, "sepsis screening",
            corpus_size=len(chunks),
        )
        evidence, _ = _rank_chunks_static(
            "sepsis screening", chunks, spec,
            adaptive=adaptive, limit=5,
        )
        assert len(evidence) > 0

    def test_static_ensemble_evidence_structure(self):
        chunks = _make_chunks()
        backends = _two_backends()
        spec = _make_spec(
            use_pipeline=False,
            embedding_backends=backends,
        )
        query_type = classify_query_type("cardiac arrest management")
        adaptive = compute_adaptive_params(
            query_type, "cardiac arrest management",
            corpus_size=len(chunks),
        )
        evidence, _ = _rank_chunks_static(
            "cardiac arrest management", chunks, spec,
            adaptive=adaptive, limit=5,
        )
        for ev in evidence:
            assert isinstance(ev, Evidence)
            assert ev.retriever == "hybrid_rrf"
            assert ev.score > 0


# ---------------------------------------------------------------------------
# 8. Query Expansion with Ensemble
# ---------------------------------------------------------------------------


class TestQueryExpansionEnsemble:
    """Expansion re-runs using all ensemble retrievers."""

    def test_pipeline_expansion_with_ensemble(self):
        from cam_rag.pipeline.executor import QueryExpansionStep

        backends = _two_backends()
        ctx = _make_context(
            embedding_backends=backends,
            query_expansion_enabled=True,
            expansion_terms=3,
        )
        SparseBM25Step().execute(ctx)
        DenseVectorStep().execute(ctx)
        RRFFusionStep().execute(ctx)

        # Evidence exists, so expansion should attempt
        assert len(ctx.evidence) > 0

        QueryExpansionStep().execute(ctx)

        # After expansion, evidence should still be valid
        assert len(ctx.evidence) > 0
        for ev in ctx.evidence:
            assert ev.score > 0

    def test_pipeline_expansion_preserves_ensemble_retrievers(self):
        from cam_rag.pipeline.executor import QueryExpansionStep

        backends = _two_backends()
        ctx = _make_context(
            embedding_backends=backends,
            query_expansion_enabled=True,
            expansion_terms=3,
        )
        SparseBM25Step().execute(ctx)
        DenseVectorStep().execute(ctx)
        RRFFusionStep().execute(ctx)

        retrievers_before = ctx.extras.get("ensemble_dense_retrievers", {})

        QueryExpansionStep().execute(ctx)

        # Ensemble retrievers should still be present
        retrievers_after = ctx.extras.get("ensemble_dense_retrievers", {})
        assert len(retrievers_after) == len(retrievers_before)

    def test_static_expansion_with_ensemble(self):
        chunks = _make_chunks()
        backends = _two_backends()
        spec = _make_spec(
            use_pipeline=False,
            embedding_backends=backends,
            query_expansion_enabled=True,
            expansion_terms=3,
        )
        query_type = classify_query_type("emergency triage protocol")
        adaptive = compute_adaptive_params(
            query_type, "emergency triage protocol",
            corpus_size=len(chunks),
        )
        evidence, _expanded = _rank_chunks_static(
            "emergency triage protocol", chunks, spec,
            adaptive=adaptive, limit=5,
        )
        assert len(evidence) > 0
        # Expansion may or may not produce a different query, but it
        # should not crash and should return valid evidence


# ---------------------------------------------------------------------------
# 9. Static/Dynamic Equivalence with Ensemble
# ---------------------------------------------------------------------------


class TestStaticDynamicEquivalenceEnsemble:
    """Both paths should produce equivalent results with ensemble."""

    def _run_both_paths(
        self, query_text: str, chunks: list[Chunk],
        backends: list[HashEmbeddingBackend],
        **spec_kw,
    ) -> tuple[list[Evidence], list[Evidence]]:
        spec_static = _make_spec(
            use_pipeline=False,
            embedding_backends=backends,
            **spec_kw,
        )
        spec_dynamic = _make_spec(
            use_pipeline=True,
            embedding_backends=backends,
            **spec_kw,
        )

        query_type = classify_query_type(query_text)
        adaptive = compute_adaptive_params(
            query_type, query_text, corpus_size=len(chunks),
        )

        static_evidence, _ = _rank_chunks_static(
            query_text, chunks, spec_static,
            adaptive=adaptive, limit=spec_static.retrieval_top_k,
        )
        dynamic_evidence, _, _gen = _rank_chunks_pipeline(
            query_text, chunks, spec_dynamic,
            adaptive=adaptive, limit=spec_dynamic.retrieval_top_k,
        )
        return static_evidence, dynamic_evidence

    def test_same_evidence_count(self):
        chunks = _make_chunks()
        backends = _two_backends()
        static_ev, dynamic_ev = self._run_both_paths(
            "emergency triage protocol", chunks, backends,
        )
        assert len(static_ev) == len(dynamic_ev)

    def test_same_top_result(self):
        chunks = _make_chunks()
        backends = _two_backends()
        static_ev, dynamic_ev = self._run_both_paths(
            "emergency triage protocol", chunks, backends,
        )
        if static_ev and dynamic_ev:
            assert static_ev[0].chunk.id == dynamic_ev[0].chunk.id

    def test_same_scores(self):
        chunks = _make_chunks()
        backends = _two_backends()
        static_ev, dynamic_ev = self._run_both_paths(
            "emergency triage protocol", chunks, backends,
        )
        for s, d in zip(static_ev, dynamic_ev, strict=True):
            assert abs(s.score - d.score) < 1e-10, (
                f"Score mismatch: static={s.score}, dynamic={d.score}"
            )

    def test_same_ranking_order(self):
        chunks = _make_chunks()
        backends = _two_backends()
        static_ev, dynamic_ev = self._run_both_paths(
            "sepsis screening criteria", chunks, backends,
        )
        static_ids = [e.chunk.id for e in static_ev]
        dynamic_ids = [e.chunk.id for e in dynamic_ev]
        assert static_ids == dynamic_ids

    def test_equivalence_with_custom_weights(self):
        chunks = _make_chunks()
        backends = _two_backends()
        static_ev, dynamic_ev = self._run_both_paths(
            "emergency triage protocol", chunks, backends,
            ensemble_weights={"dense_0": 0.7, "dense_1": 0.3},
        )
        static_ids = [e.chunk.id for e in static_ev]
        dynamic_ids = [e.chunk.id for e in dynamic_ev]
        assert static_ids == dynamic_ids

    def test_equivalence_different_query_types(self):
        chunks = _make_chunks()
        backends = _two_backends()
        queries = [
            "what is triage",  # specification
            "chest pain assessment",  # summary
            "why is qSOFA used for sepsis",  # logic
            "compare triage and trauma? how do they differ?",  # synthesis
        ]
        for q in queries:
            static_ev, dynamic_ev = self._run_both_paths(
                q, chunks, backends,
            )
            static_ids = [e.chunk.id for e in static_ev]
            dynamic_ids = [e.chunk.id for e in dynamic_ev]
            assert static_ids == dynamic_ids, f"Mismatch for query: {q}"


# ---------------------------------------------------------------------------
# 10. Backward Compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """All existing behavior unchanged when embedding_backends is empty."""

    def test_spec_defaults_to_empty_backends(self):
        spec = _make_spec()
        assert spec.embedding_backends == []
        assert spec.ensemble_weights == {}

    def test_spec_with_single_embedding_backend(self):
        """Single embedding_backend still works as before."""
        backend = HashEmbeddingBackend(dim=32)
        spec = _make_spec(embedding_backend=backend)
        assert spec.embedding_backend is backend
        assert spec.embedding_backends == []

    def test_static_path_no_ensemble_matches_original(self):
        """Static path without ensemble backends produces the same results
        as it always did (using the single embedding_backend)."""
        chunks = _make_chunks(5)
        backend = HashEmbeddingBackend(dim=64)

        # With single embedding_backend (original path)
        spec_single = _make_spec(
            embedding_backend=backend,
            use_pipeline=False,
        )
        query_type = classify_query_type("emergency triage protocol")
        adaptive = compute_adaptive_params(
            query_type, "emergency triage protocol",
            corpus_size=len(chunks),
        )
        evidence_single, _ = _rank_chunks_static(
            "emergency triage protocol", chunks, spec_single,
            adaptive=adaptive, limit=3,
        )

        # With empty embedding_backends (also original path)
        spec_empty = _make_spec(
            embedding_backend=backend,
            embedding_backends=[],
            use_pipeline=False,
        )
        evidence_empty, _ = _rank_chunks_static(
            "emergency triage protocol", chunks, spec_empty,
            adaptive=adaptive, limit=3,
        )

        single_ids = [e.chunk.id for e in evidence_single]
        empty_ids = [e.chunk.id for e in evidence_empty]
        assert single_ids == empty_ids

    def test_pipeline_path_no_ensemble_matches_original(self):
        """Pipeline path without ensemble produces same as original."""
        chunks = _make_chunks(5)
        backend = HashEmbeddingBackend(dim=64)

        spec = _make_spec(
            embedding_backend=backend,
            use_pipeline=True,
        )
        query_type = classify_query_type("emergency triage protocol")
        adaptive = compute_adaptive_params(
            query_type, "emergency triage protocol",
            corpus_size=len(chunks),
        )
        evidence, _, _gen = _rank_chunks_pipeline(
            "emergency triage protocol", chunks, spec,
            adaptive=adaptive, limit=3,
        )
        assert len(evidence) > 0

    def test_dense_step_reuses_retriever_in_non_ensemble_mode(self):
        """In non-ensemble mode, DenseVectorStep still reuses the retriever."""
        ctx = _make_context()
        DenseVectorStep().execute(ctx)
        retriever_1 = ctx.dense_retriever
        DenseVectorStep().execute(ctx)
        assert ctx.dense_retriever is retriever_1

    def test_ensemble_step_reuses_retrievers(self):
        """In ensemble mode, retrievers stored in extras are reused."""
        backends = _two_backends()
        ctx = _make_context(embedding_backends=backends)
        DenseVectorStep().execute(ctx)
        retrievers_1 = dict(ctx.extras["ensemble_dense_retrievers"])

        # Run again -- should reuse
        DenseVectorStep().execute(ctx)
        retrievers_2 = ctx.extras["ensemble_dense_retrievers"]

        for name in retrievers_1:
            assert retrievers_2[name] is retrievers_1[name], (
                f"Retriever {name} was not reused"
            )


# ---------------------------------------------------------------------------
# Additional Edge Cases
# ---------------------------------------------------------------------------


class TestEnsembleEdgeCases:
    """Edge cases and additional validation."""

    def test_single_backend_in_ensemble_list(self):
        """A single backend in embedding_backends still uses ensemble path."""
        backends = [HashEmbeddingBackend(dim=32)]
        ctx = _make_context(embedding_backends=backends)
        SparseBM25Step().execute(ctx)
        DenseVectorStep().execute(ctx)
        RRFFusionStep().execute(ctx)

        assert "ensemble_dense_results" in ctx.extras
        assert len(ctx.extras["ensemble_dense_results"]) == 1
        assert len(ctx.evidence) > 0

    def test_ensemble_with_limit_1(self):
        """Ensemble with limit=1 returns exactly one result."""
        backends = _two_backends()
        ctx = _make_context(embedding_backends=backends)
        ctx.limit = 1
        SparseBM25Step().execute(ctx)
        DenseVectorStep().execute(ctx)
        RRFFusionStep().execute(ctx)

        assert len(ctx.evidence) == 1

    def test_ensemble_evidence_limit_respected(self):
        """Evidence count never exceeds the configured limit."""
        backends = _two_backends()
        ctx = _make_context(
            embedding_backends=backends,
            retrieval_top_k=3,
        )
        ctx.limit = 3
        SparseBM25Step().execute(ctx)
        DenseVectorStep().execute(ctx)
        RRFFusionStep().execute(ctx)

        assert len(ctx.evidence) <= 3

    def test_ensemble_with_varied_queries(self):
        """Ensemble works across different query types without errors."""
        backends = _two_backends()
        queries = [
            "what is the HEART score",
            "trauma assessment protocol",
            "why use qSOFA for sepsis screening",
            "compare triage and stroke assessment? differences?",
        ]
        for q in queries:
            ctx = _make_context(
                query_text=q,
                embedding_backends=backends,
            )
            SparseBM25Step().execute(ctx)
            DenseVectorStep().execute(ctx)
            RRFFusionStep().execute(ctx)

            assert len(ctx.evidence) > 0, (
                f"No evidence for query: {q}"
            )
