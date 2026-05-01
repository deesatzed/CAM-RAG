"""MTEB SearchProtocol implementation for full-pipeline retrieval benchmarking.

``CamRAGSearchModel`` wraps the entire CAM RAG pipeline (BM25 + dense + RRF +
reranking + routing) behind the MTEB ``SearchProtocol`` so that MTEB can
evaluate the system end-to-end via nDCG@10, not just the embedding in isolation.

Supports adaptive strategy routing: the pipeline automatically detects
embedding quality at index time and selects the optimal strategy
(dense_dominant, hybrid, or sparse_boost) at search time.
"""

from __future__ import annotations

import logging
from typing import Any

from cam_rag.pipeline.context import build_context
from cam_rag.pipeline.executor import PipelineExecutor, get_step
from cam_rag.pipeline.registry import ExecutionPlan
from cam_rag.pipeline.strategy import StrategyRouter
from cam_rag.rag.models import Chunk
from cam_rag.rag.spec import RAGAppSpec
from cam_rag.retrieval.adaptive_params import AdaptiveParams
from cam_rag.retrieval.dense import DenseVectorRetriever, HashEmbeddingBackend
from cam_rag.retrieval.models import RetrievalDocument
from cam_rag.retrieval.sparse import SparseBM25Retriever

logger = logging.getLogger(__name__)

# Steps that are relevant to retrieval scoring (not generation/verification)
_RETRIEVAL_STEPS = (
    "sparse_bm25",
    "hyde_expansion",
    "dense_vector",
    "adaptive_fusion_weights",
    "rrf_fusion",
    "title_boost",
    "moe_importance_scoring",
    "accuracy_contracts",
    "selective_filter",
    "complexity_routing",
    "query_expansion",
    "query_decompose",
    "cross_encoder_rerank",
    "reciprocal_neighbor_rerank",
    "multi_hop_retrieval",
    "score_normalize",
)

# Steps to skip in benchmark mode (generation, verification)
_SKIP_STEPS = frozenset({
    "confidence_scoring",
    "citation_grounding",
    "generation",
    "etf_verification",
})


class CamRAGSearchModel:
    """MTEB SearchProtocol implementation backed by the full CAM RAG pipeline.

    ``index(corpus)`` converts an HF Dataset into ``Chunk`` objects and
    pre-builds sparse and dense retrievers.  ``search(queries, top_k)``
    runs the pipeline for each query and returns scores.

    When ``spec.pipeline_strategy`` is ``"auto"``, the model detects
    embedding quality at index time and routes to the optimal strategy.
    """

    def __init__(
        self,
        spec: RAGAppSpec | None = None,
        *,
        name: str = "cam-rag-pipeline",
        retrieval_depth: int = 100,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ) -> None:
        self.spec = spec or RAGAppSpec(
            name="mteb-bench",
            use_pipeline=True,
            embedding_backend=HashEmbeddingBackend(),
        )
        self._retrieval_depth = retrieval_depth
        self._dense_weight = dense_weight
        self._sparse_weight = sparse_weight
        self.mteb_model_meta = {
            "name": name,
            "revision": None,
            "release_date": None,
        }
        # Populated by index()
        self._chunks: list[Chunk] = []
        self._chunk_by_id: dict[str, Chunk] = {}
        self._retrieval_docs: list[RetrievalDocument] = []
        self._sparse_retriever: SparseBM25Retriever | None = None
        self._dense_retriever: DenseVectorRetriever | None = None
        # Quality tier detected at index time
        self._quality_tier: Any = None
        self._strategy_router = StrategyRouter()
        self._corpus_signals: Any = None
        self._calibration_result: Any = None

    def index(self, corpus: Any, **kwargs: Any) -> None:
        """Build internal retrieval indexes from an MTEB corpus dataset.

        Parameters
        ----------
        corpus : datasets.Dataset
            HF Dataset with columns ``id``, ``text``, and optionally ``title``.
        **kwargs
            Additional keyword arguments from MTEB (task_metadata,
            hf_split, encode_kwargs, etc.) -- accepted but not used.
        """
        self._chunks = []
        self._chunk_by_id = {}
        self._retrieval_docs = []

        for row in corpus:
            doc_id = str(row["id"]) if "id" in row else str(row.get("_id", ""))
            text = str(row.get("text", ""))
            title = str(row.get("title", ""))
            full_text = f"{title}\n{text}".strip() if title else text

            chunk = Chunk(
                id=doc_id,
                document_id=doc_id,
                text=full_text,
                title=title,
            )
            self._chunks.append(chunk)
            self._chunk_by_id[doc_id] = chunk
            self._retrieval_docs.append(
                RetrievalDocument(
                    doc_id=doc_id,
                    text=full_text,
                    metadata={"chunk_id": doc_id},
                )
            )

        # Pre-build retrievers
        self._sparse_retriever = SparseBM25Retriever(
            self._retrieval_docs,
            tokenizer=self.spec.tokenize,
        )

        backend = self.spec.embedding_backend or HashEmbeddingBackend()
        self._dense_retriever = DenseVectorRetriever(
            self._retrieval_docs,
            backend=backend,
        )

        # Compute corpus signals for corpus-aware strategy routing
        try:
            from cam_rag.retrieval.corpus_signals import compute_corpus_signals
            doc_texts = [doc.text for doc in self._retrieval_docs]
            self._corpus_signals = compute_corpus_signals(
                doc_texts, tokenizer=self.spec.tokenize,
            )
            logger.info(
                "Corpus signals: avg_len=%.1f short_frac=%.2f "
                "vocab_overlap=%.3f corpus_size=%d",
                self._corpus_signals.avg_doc_length,
                self._corpus_signals.short_doc_fraction,
                self._corpus_signals.vocab_overlap_ratio,
                self._corpus_signals.corpus_size,
            )
        except Exception:
            logger.warning("Corpus signal computation failed")
            self._corpus_signals = None

        # Detect embedding quality for adaptive strategy routing
        pipeline_strategy = getattr(self.spec, "pipeline_strategy", "auto")
        if pipeline_strategy == "auto":
            self._detect_quality(backend)

        # Index retriever plugins
        for plugin in getattr(self.spec, "retriever_plugins", []):
            try:
                plugin.index(self._retrieval_docs)
            except Exception:
                logger.warning(
                    "Plugin %s failed to index", getattr(plugin, "name", "?"),
                )

        # Auto-calibration (when enabled and corpus is large enough)
        if getattr(self.spec, "auto_calibrate", False) and len(self._retrieval_docs) >= 50:
            self._run_calibration(backend)

        logger.info(
            "Indexed %d documents for SearchProtocol benchmark "
            "(quality_tier=%s, strategy=%s)",
            len(self._chunks),
            self._quality_tier.tier if self._quality_tier else "n/a",
            pipeline_strategy,
        )

    def _detect_quality(self, backend: Any) -> None:
        """Run embedding quality detection on a corpus sample."""
        try:
            from cam_rag.retrieval.quality_detector import EmbeddingQualityDetector

            detector = EmbeddingQualityDetector()
            sample_texts = [
                doc.text for doc in self._retrieval_docs[:200]
            ]
            self._quality_tier = detector.detect(
                backend, sample_texts, sample_size=200,
            )
            logger.info(
                "Detected embedding quality: tier=%s dim=%d "
                "dispersion=%.4f separation=%.4f",
                self._quality_tier.tier,
                self._quality_tier.dimensionality,
                self._quality_tier.dispersion,
                self._quality_tier.cluster_separation,
            )
        except Exception:
            logger.warning(
                "Quality detection failed, defaulting to hybrid strategy",
            )
            self._quality_tier = None

    def _run_calibration(self, backend: Any) -> None:
        """Run unsupervised pipeline parameter calibration."""
        try:
            from cam_rag.pipeline.calibration import PipelineCalibrator
            calibrator = PipelineCalibrator(seed=42)
            self._calibration_result = calibrator.calibrate(
                documents=self._retrieval_docs,
                backend=backend,
                tokenizer=self.spec.tokenize,
            )
            logger.info(
                "Auto-calibration: dense_w=%.2f sparse_w=%.2f depth=%d "
                "rrf_k=%.1f (score=%.4f, %d grid points)",
                self._calibration_result.dense_weight,
                self._calibration_result.sparse_weight,
                self._calibration_result.retrieval_depth,
                self._calibration_result.rrf_k,
                self._calibration_result.calibration_score,
                self._calibration_result.grid_points_evaluated,
            )
        except Exception:
            logger.warning("Auto-calibration failed, using defaults")
            self._calibration_result = None

    def search(
        self,
        queries: Any,
        top_k: int = 10,
        **kwargs: Any,
    ) -> dict[str, dict[str, float]]:
        """Run the full pipeline for each query and return ranked scores.

        Parameters
        ----------
        queries : datasets.Dataset
            HF Dataset with columns ``id`` and ``text``.
        top_k : int
            Number of results per query.

        Returns
        -------
        dict[str, dict[str, float]]
            ``{query_id: {doc_id: score, ...}, ...}``
        """
        results: dict[str, dict[str, float]] = {}

        # Select pipeline strategy based on quality tier
        quality_tier_name = (
            self._quality_tier.tier if self._quality_tier else "moderate"
        )
        strategy = self._strategy_router.select(
            quality_tier_name, self.spec, corpus_signals=self._corpus_signals,
        )

        # Build execution plan from strategy steps
        # Filter to only registered steps
        steps = tuple(
            s for s in strategy.steps
            if _step_is_registered(s) and s not in _SKIP_STEPS
        )
        plan = ExecutionPlan(steps=steps, query_type="summary")
        executor = PipelineExecutor()

        # Priority: explicit constructor args > calibration > strategy defaults
        # Determine if constructor args are non-default (user overrides)
        user_set_dense = self._dense_weight != 0.6
        user_set_sparse = self._sparse_weight != 0.4

        if user_set_dense:
            dense_weight = self._dense_weight
        elif self._calibration_result is not None:
            dense_weight = self._calibration_result.dense_weight
        elif strategy.dense_weight != 0.6:
            dense_weight = strategy.dense_weight
        else:
            dense_weight = self._dense_weight

        if user_set_sparse:
            sparse_weight = self._sparse_weight
        elif self._calibration_result is not None:
            sparse_weight = self._calibration_result.sparse_weight
        elif strategy.sparse_weight != 0.4:
            sparse_weight = strategy.sparse_weight
        else:
            sparse_weight = self._sparse_weight

        if self._calibration_result is not None and not user_set_dense:
            depth = self._calibration_result.retrieval_depth
        else:
            depth = self._retrieval_depth or strategy.retrieval_depth

        logger.debug(
            "Using strategy=%s steps=%d dense_w=%.2f sparse_w=%.2f depth=%d",
            strategy.name, len(steps), dense_weight, sparse_weight, depth,
        )

        for row in queries:
            query_id = str(row["id"]) if "id" in row else str(row.get("_id", ""))
            query_text = str(row.get("text", ""))

            ctx = build_context(
                query_text=query_text,
                chunks=self._chunks,
                spec=self.spec,
                adaptive=AdaptiveParams(
                    dense_k=max(depth, top_k * 3),
                    sparse_k=max(depth, top_k * 4),
                    final_k=depth,
                    dense_weight=dense_weight,
                    sparse_weight=sparse_weight,
                    rrf_k=(
                        self._calibration_result.rrf_k
                        if self._calibration_result is not None and not user_set_dense
                        else 60.0
                    ),
                ),
                limit=depth,
            )
            # Inject pre-built retrievers
            ctx.sparse_retriever = self._sparse_retriever
            ctx.dense_retriever = self._dense_retriever

            executor.run(plan, ctx)

            # Extract scores from evidence
            doc_scores: dict[str, float] = {}
            for item in ctx.evidence[:top_k]:
                doc_scores[item.chunk.id] = item.score
            results[query_id] = doc_scores

        return results


def _step_is_registered(name: str) -> bool:
    """Check if a step is registered without raising."""
    try:
        get_step(name)
        return True
    except KeyError:
        return False
