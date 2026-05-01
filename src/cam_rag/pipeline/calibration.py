"""Unsupervised pipeline parameter tuning via pseudo-relevance feedback."""

from __future__ import annotations

import logging
import math
import random
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from cam_rag.retrieval.dense import DenseVectorRetriever
from cam_rag.retrieval.fusion import rrf_fuse
from cam_rag.retrieval.models import RetrievalDocument, RetrievalResult
from cam_rag.retrieval.sparse import SparseBM25Retriever

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Outcome of a grid-search calibration run."""

    dense_weight: float
    sparse_weight: float
    retrieval_depth: int
    rrf_k: float
    calibration_score: float  # best MRR achieved during grid search
    grid_points_evaluated: int
    holdout_size: int


class PipelineCalibrator:
    """Unsupervised pipeline parameter tuning via pseudo-relevance feedback.

    Uses a grid search over (dense_weight, retrieval_depth, rrf_k) to find
    optimal pipeline parameters.  Generates pseudo-queries from holdout docs
    and evaluates retrieval quality via mean reciprocal rank (MRR).
    """

    _DEFAULT_GRID: dict[str, tuple[float | int, ...]] = {
        "dense_weight": (0.3, 0.5, 0.6, 0.7, 0.8, 1.0),
        "retrieval_depth": (50, 100, 150),
        "rrf_k": (30.0, 60.0, 90.0),
    }

    def __init__(
        self,
        *,
        seed: int = 42,
        max_pseudo_queries: int = 20,
    ) -> None:
        self._seed = seed
        self._max_pseudo_queries = max_pseudo_queries
        self._cache: dict[tuple[int, str], CalibrationResult] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calibrate(
        self,
        documents: list[RetrievalDocument],
        backend: Any,
        tokenizer: Callable[[str], list[str]],
        seed: int | None = None,
    ) -> CalibrationResult:
        """Run grid search calibration on the given corpus."""

        effective_seed = seed if seed is not None else self._seed

        # Build cache key from corpus fingerprint
        cache_key = self._cache_key(documents)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Too few documents -- return sensible defaults
        if len(documents) < 50:
            result = CalibrationResult(
                dense_weight=0.6,
                sparse_weight=0.4,
                retrieval_depth=100,
                rrf_k=60.0,
                calibration_score=0.0,
                grid_points_evaluated=0,
                holdout_size=len(documents),
            )
            self._cache[cache_key] = result
            return result

        # Deterministic train / holdout split
        rng = random.Random(effective_seed)
        indices = list(range(len(documents)))
        rng.shuffle(indices)
        split = int(len(indices) * 0.8)
        train_docs = [documents[i] for i in indices[:split]]
        holdout_docs = [documents[i] for i in indices[split:]]

        # Pseudo-query generation
        pseudo_queries = self._generate_pseudo_queries(
            train_docs, holdout_docs, tokenizer, rng
        )

        # Grid search
        best = self._grid_search(
            train_docs, pseudo_queries, backend, tokenizer
        )

        self._cache[cache_key] = best
        return best

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(
        documents: list[RetrievalDocument],
    ) -> tuple[int, str]:
        return (
            len(documents),
            "_".join(d.doc_id for d in documents[:10]),
        )

    def _generate_pseudo_queries(
        self,
        train_docs: list[RetrievalDocument],
        holdout_docs: list[RetrievalDocument],
        tokenizer: Callable[[str], list[str]],
        rng: random.Random,
    ) -> list[tuple[str, str]]:
        """Build pseudo-queries from holdout documents using TF-IDF.

        Returns a list of ``(query_text, expected_doc_id)`` tuples.
        """

        # Build document-frequency counter from the training set
        df_counter: Counter[str] = Counter()
        for doc in train_docs:
            unique_terms = set(tokenizer(doc.text))
            df_counter.update(unique_terms)

        n_train = len(train_docs)
        queries: list[tuple[str, str]] = []

        for doc in holdout_docs:
            tokens = tokenizer(doc.text)
            if not tokens:
                continue

            total_tokens = len(tokens)
            tf_counter = Counter(tokens)

            # Compute TF-IDF for each unique term
            scored_terms: list[tuple[str, float]] = []
            for term, count in tf_counter.items():
                tf = count / total_tokens
                idf = math.log(n_train / (1 + df_counter.get(term, 0)))
                scored_terms.append((term, tf * idf))

            # Take top-5 terms by TF-IDF
            scored_terms.sort(key=lambda item: -item[1])
            top_terms = [t for t, _ in scored_terms[:5]]
            if top_terms:
                queries.append((" ".join(top_terms), doc.doc_id))

        # Deterministic shuffle then cap
        rng.shuffle(queries)
        return queries[: self._max_pseudo_queries]

    def _grid_search(
        self,
        train_docs: list[RetrievalDocument],
        pseudo_queries: list[tuple[str, str]],
        backend: Any,
        tokenizer: Callable[[str], list[str]],
    ) -> CalibrationResult:
        """Evaluate every grid combination and return the best result."""

        # Build retrievers once
        sparse_retriever = SparseBM25Retriever(train_docs, tokenizer=tokenizer)
        dense_retriever = DenseVectorRetriever(train_docs, backend=backend)

        dense_weights = self._DEFAULT_GRID["dense_weight"]
        depths = self._DEFAULT_GRID["retrieval_depth"]
        rrf_ks = self._DEFAULT_GRID["rrf_k"]

        best_mrr = -1.0
        best_params: tuple[float, int, float] = (0.6, 100, 60.0)
        grid_points = 0

        for dw in dense_weights:
            dw_f = float(dw)
            sw_f = 1.0 - dw_f
            for depth in depths:
                depth_i = int(depth)
                for rk in rrf_ks:
                    rk_f = float(rk)
                    grid_points += 1

                    mrr = self._evaluate_combo(
                        sparse_retriever=sparse_retriever,
                        dense_retriever=dense_retriever,
                        pseudo_queries=pseudo_queries,
                        dense_weight=dw_f,
                        retrieval_depth=depth_i,
                        rrf_k=rk_f,
                    )

                    if mrr > best_mrr:
                        best_mrr = mrr
                        best_params = (dw_f, depth_i, rk_f)

        dw_best, depth_best, rk_best = best_params
        return CalibrationResult(
            dense_weight=dw_best,
            sparse_weight=round(1.0 - dw_best, 4),
            retrieval_depth=depth_best,
            rrf_k=rk_best,
            calibration_score=best_mrr,
            grid_points_evaluated=grid_points,
            holdout_size=len(pseudo_queries),
        )

    @staticmethod
    def _evaluate_combo(
        *,
        sparse_retriever: SparseBM25Retriever,
        dense_retriever: DenseVectorRetriever,
        pseudo_queries: list[tuple[str, str]],
        dense_weight: float,
        retrieval_depth: int,
        rrf_k: float,
    ) -> float:
        """Compute mean reciprocal rank for one parameter combo."""

        if not pseudo_queries:
            return 0.0

        total_rr = 0.0
        for query_text, expected_doc_id in pseudo_queries:
            dense_results = dense_retriever.retrieve(query_text, k=retrieval_depth)

            if dense_weight >= 1.0:
                # Pure dense -- skip BM25 and fusion
                rank = _find_rank(dense_results, expected_doc_id)
            else:
                sparse_results = sparse_retriever.retrieve(
                    query_text, k=retrieval_depth
                )
                fused = rrf_fuse(
                    dense_results,
                    sparse_results,
                    rrf_k=rrf_k,
                    weights=[dense_weight, 1.0 - dense_weight],
                    names=["dense", "sparse"],
                )
                rank = _find_rank_fused(fused, expected_doc_id)

            if rank > 0:
                total_rr += 1.0 / rank

        return total_rr / len(pseudo_queries)


def _find_rank(results: list[RetrievalResult], doc_id: str) -> int:
    """Return 1-based rank of *doc_id* in results, or 0 if absent."""

    for i, result in enumerate(results):
        if result.doc_id == doc_id:
            return i + 1
    return 0


def _find_rank_fused(
    fused: list[Any],
    doc_id: str,
) -> int:
    """Return 1-based rank of *doc_id* in fused results, or 0 if absent."""

    for i, result in enumerate(fused):
        if result.doc_id == doc_id:
            return i + 1
    return 0
