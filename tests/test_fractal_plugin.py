"""Tests for FractalRetrieverPlugin protocol compliance and HashEmbedding retrieve."""

from __future__ import annotations

from cam_rag.pipeline.context import build_context
from cam_rag.pipeline.executor import PipelineExecutor
from cam_rag.pipeline.registry import ExecutionPlan
from cam_rag.rag.models import Chunk
from cam_rag.rag.spec import RAGAppSpec
from cam_rag.retrieval.adaptive_params import AdaptiveParams
from cam_rag.retrieval.dense import HashEmbeddingBackend
from cam_rag.retrieval.fractal import FractalRetrieverPlugin, HashEmbedding
from cam_rag.retrieval.models import RetrievalDocument, RetrievalResult
from cam_rag.retrieval.plugin import RetrieverPlugin


def _docs() -> list[RetrievalDocument]:
    return [
        RetrievalDocument(
            doc_id="ai",
            text=(
                "Artificial intelligence diagnoses diseases using deep learning models. "
                "Convolutional networks analyse radiology scans for tumour detection. "
                "Machine learning improves diagnostic accuracy in medical imaging. "
                "Transfer learning from ImageNet boosts radiology AI performance."
            ),
            metadata={"domain": "radiology", "title": "Radiology AI"},
        ),
        RetrievalDocument(
            doc_id="rome",
            text=(
                "The Western Roman Empire collapsed in 476 AD after centuries of decline. "
                "Economic troubles, barbarian invasions, and political instability were key factors. "
                "The Renaissance later revived classical art, science, and humanism in Europe. "
                "World War II, ending in 1945, reshaped global power structures forever."
            ),
            metadata={"domain": "history", "title": "Roman History"},
        ),
        RetrievalDocument(
            doc_id="dna",
            text=(
                "DNA stores genetic information using four nucleotide bases. "
                "RNA transcription converts DNA sequences into messenger RNA. "
                "Protein synthesis occurs at ribosomes during translation. "
                "Mutations in DNA can lead to genetic diseases or evolutionary advantages."
            ),
            metadata={"domain": "biology", "title": "Genetics"},
        ),
    ]


class TestFractalPluginProtocol:
    def test_isinstance_check(self) -> None:
        plugin = FractalRetrieverPlugin(backend=HashEmbedding(dim=64))
        assert isinstance(plugin, RetrieverPlugin)

    def test_plugin_name(self) -> None:
        plugin = FractalRetrieverPlugin(backend=HashEmbedding(dim=64))
        assert plugin.name == "fractal"

    def test_not_in_default_spec(self) -> None:
        spec = RAGAppSpec(name="default")
        assert spec.retriever_plugins == []


class TestHashEmbeddingRetrieve:
    def test_retrieve_returns_results(self) -> None:
        plugin = FractalRetrieverPlugin(backend=HashEmbedding(dim=64))
        plugin.index(_docs())
        results = plugin.retrieve("How does AI help diagnose disease?", k=3)
        assert len(results) >= 1
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert all(r.doc_id in {"ai", "rome", "dna"} for r in results)
        assert results[0].rank == 1
        assert results[0].score is not None
        assert results[0].metadata.get("query_type") in (
            "specification",
            "summary",
            "logic",
            "synthesis",
        )

    def test_empty_query_returns_nothing(self) -> None:
        plugin = FractalRetrieverPlugin(backend=HashEmbedding(dim=64))
        plugin.index(_docs())
        assert plugin.retrieve("", k=5) == []
        assert plugin.retrieve("   ", k=5) == []

    def test_retrieve_before_index_returns_nothing(self) -> None:
        plugin = FractalRetrieverPlugin(backend=HashEmbedding(dim=64))
        assert plugin.retrieve("anything", k=5) == []

    def test_respects_k(self) -> None:
        plugin = FractalRetrieverPlugin(backend=HashEmbedding(dim=64))
        plugin.index(_docs())
        results = plugin.retrieve("summarize the evidence", k=2)
        assert len(results) <= 2

    def test_reindex_replaces_corpus(self) -> None:
        plugin = FractalRetrieverPlugin(backend=HashEmbedding(dim=64))
        plugin.index(_docs())
        plugin.index(
            [RetrievalDocument(doc_id="only", text="Only this document remains here.")]
        )
        results = plugin.retrieve("document remains", k=5)
        assert all(r.doc_id == "only" for r in results)

    def test_cam_rag_hash_backend_works(self) -> None:
        plugin = FractalRetrieverPlugin(backend=HashEmbeddingBackend(dim=64))
        plugin.index(_docs())
        results = plugin.retrieve("DNA nucleotide bases", k=3)
        assert len(results) >= 1
        assert isinstance(results[0], RetrievalResult)

    def test_engine_has_three_levels(self) -> None:
        plugin = FractalRetrieverPlugin(backend=HashEmbedding(dim=64))
        plugin.index(_docs())
        stats = plugin.engine.stats()
        assert stats["documents"] == 3
        assert stats["entries_level_2"] == 3
        assert stats["entries_level_1"] >= 3
        assert stats["entries_level_0"] >= 3
        assert stats["derivatives"] >= 3


class TestFractalPluginPipeline:
    def test_plugin_participates_in_rrf_when_requested(self) -> None:
        chunks = [
            Chunk(id="ai", document_id="ai", text="AI diagnoses diseases with deep learning."),
            Chunk(id="rome", document_id="rome", text="Rome fell in 476 AD after decline."),
            Chunk(id="dna", document_id="dna", text="DNA stores genetic information in bases."),
        ]
        plugin = FractalRetrieverPlugin(backend=HashEmbedding(dim=64))
        spec = RAGAppSpec(
            name="test",
            use_pipeline=True,
            retriever_plugins=[plugin],
        )
        ctx = build_context(
            query_text="How does AI diagnose disease?",
            chunks=chunks,
            spec=spec,
            adaptive=AdaptiveParams(),
            limit=5,
        )
        plan = ExecutionPlan(
            steps=("sparse_bm25", "dense_vector", "rrf_fusion"),
            query_type="logic",
        )
        PipelineExecutor().run(plan, ctx)
        assert len(ctx.evidence) > 0
        assert "_plugins_indexed" in ctx.extras
        assert "fractal" in ctx.extras["_plugins_indexed"]


class TestFractalPluginOptions:
    def test_default_hash_embedding_backend(self) -> None:
        plugin = FractalRetrieverPlugin()
        plugin.index(_docs())
        results = plugin.retrieve("machine learning diagnosis", k=3)
        assert len(results) >= 1

    def test_non_adaptive_retrieve(self) -> None:
        plugin = FractalRetrieverPlugin(
            backend=HashEmbedding(dim=64),
            use_adaptive=False,
        )
        plugin.index(_docs())
        results = plugin.retrieve("How does AI diagnose disease?", k=3)
        assert len(results) >= 1
        assert results[0].rank == 1

    def test_empty_index(self) -> None:
        plugin = FractalRetrieverPlugin(backend=HashEmbedding(dim=64))
        plugin.index([])
        assert plugin.retrieve("query", k=5) == []


class TestFractalEngineAdapter:
    def test_learn_adapter_false(self) -> None:
        from cam_rag.retrieval.fractal import FractalRAG, HashEmbedding

        rag = FractalRAG(backend=HashEmbedding(dim=64))
        rag.add_document("d1", "Adapter-free document text for indexing.", learn_adapter=False)
        assert "d1" not in rag.adapters
        results, _ = rag.retrieve("document text", k=1)
        assert len(results[2]) == 1
