"""Pipeline strategy definitions and adaptive routing.

Defines four pre-built pipeline strategies based on benchmark evidence:

- **dense_dominant**: Skip BM25/RRF, use dense retrieval only + reranker.
  For strong embeddings (>= 2048d) where BM25 fusion dilutes quality.
- **strong_hybrid**: Full hybrid pipeline weighted toward dense retrieval.
  For strong embeddings on short-doc / keyword-heavy corpora where BM25
  keyword matching is still valuable despite high-quality embeddings.
- **hybrid**: Full BM25 + dense + RRF + reranker pipeline. For moderate
  embeddings where sparse and dense signals complement each other.
- **sparse_boost**: Boost BM25 weight for weak embeddings or
  keyword-heavy domains.

The ``StrategyRouter`` selects the appropriate strategy based on an
``EmbeddingQualityTier`` and optional ``CorpusSignals``, or an explicit
spec override.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PipelineStrategy:
    """A named pipeline configuration with step sequence and weight overrides."""

    name: str
    steps: tuple[str, ...]
    dense_weight: float
    sparse_weight: float
    retrieval_depth: int
    description: str = ""


# ---------------------------------------------------------------------------
# Step lists
# ---------------------------------------------------------------------------

# Full hybrid pipeline retrieval steps (current default)
_HYBRID_STEPS: tuple[str, ...] = (
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

# Dense-dominant: skip BM25/RRF entirely
_DENSE_DOMINANT_STEPS: tuple[str, ...] = (
    "hyde_expansion",
    "dense_vector",
    "dense_to_evidence",
    "moe_importance_scoring",
    "accuracy_contracts",
    "selective_filter",
    "complexity_routing",
    "cross_encoder_rerank",
    "score_normalize",
)

# Sparse-boost: same steps as hybrid, different weights
_SPARSE_BOOST_STEPS: tuple[str, ...] = _HYBRID_STEPS


# ---------------------------------------------------------------------------
# Pre-built strategies
# ---------------------------------------------------------------------------

DENSE_DOMINANT = PipelineStrategy(
    name="dense_dominant",
    steps=_DENSE_DOMINANT_STEPS,
    dense_weight=1.0,
    sparse_weight=0.0,
    retrieval_depth=100,
    description=(
        "Skip BM25/RRF entirely. For embeddings >= 2048d with high "
        "dispersion. Benchmark evidence: Qwen3-8B loses 2.6% from "
        "BM25 fusion; dense-only preserves embedding quality."
    ),
)

HYBRID = PipelineStrategy(
    name="hybrid",
    steps=_HYBRID_STEPS,
    dense_weight=0.6,
    sparse_weight=0.4,
    retrieval_depth=100,
    description=(
        "Full BM25 + dense + RRF + reranker. For moderate embeddings. "
        "Benchmark evidence: MiniLM-384d gains +30% from hybrid pipeline."
    ),
)

STRONG_HYBRID = PipelineStrategy(
    name="strong_hybrid",
    steps=_HYBRID_STEPS,
    dense_weight=0.7,
    sparse_weight=0.3,
    retrieval_depth=100,
    description=(
        "Hybrid pipeline tuned for strong embeddings on short-doc / "
        "keyword-heavy corpora. Preserves BM25 keyword matching while "
        "still weighting dense retrieval heavily. Benchmark evidence: "
        "NFCorpus loses -10% with dense_dominant; strong_hybrid recovers "
        "BM25 keyword signal."
    ),
)

SPARSE_BOOST = PipelineStrategy(
    name="sparse_boost",
    steps=_SPARSE_BOOST_STEPS,
    dense_weight=0.3,
    sparse_weight=0.7,
    retrieval_depth=150,
    description=(
        "Boost BM25 for weak embeddings or keyword-heavy domains. "
        "Higher retrieval depth gives the reranker more candidates."
    ),
)

# All built-in strategies
BUILTIN_STRATEGIES: dict[str, PipelineStrategy] = {
    "dense_dominant": DENSE_DOMINANT,
    "strong_hybrid": STRONG_HYBRID,
    "hybrid": HYBRID,
    "sparse_boost": SPARSE_BOOST,
}

# Tier → strategy name mapping
_TIER_STRATEGY_MAP: dict[str, str] = {
    "strong": "dense_dominant",
    "moderate": "hybrid",
    "weak": "sparse_boost",
}


class StrategyRouter:
    """Route queries to the optimal pipeline strategy.

    When ``spec.pipeline_strategy`` is ``"auto"``, the router uses the
    embedding quality tier to select the strategy. When the spec has an
    explicit strategy name, that overrides the tier.

    Custom strategies can be registered via ``register_strategy()``.
    """

    def __init__(
        self, strategies: dict[str, PipelineStrategy] | None = None
    ) -> None:
        self._strategies = dict(strategies or BUILTIN_STRATEGIES)

    def select(
        self,
        quality_tier: str,
        spec: Any,
        *,
        corpus_signals: Any | None = None,
    ) -> PipelineStrategy:
        """Select a pipeline strategy.

        Parameters
        ----------
        quality_tier:
            One of ``"strong"``, ``"moderate"``, ``"weak"``.
        spec:
            A ``RAGAppSpec`` (or anything with a ``pipeline_strategy`` attribute).
        corpus_signals:
            Optional ``CorpusSignals`` describing corpus characteristics.
            When provided and strategy is ``"auto"``, corpus-aware overrides
            are applied after tier-based selection.
        """
        explicit = getattr(spec, "pipeline_strategy", "auto")
        if explicit != "auto":
            if explicit not in self._strategies:
                raise KeyError(
                    f"Unknown pipeline strategy: {explicit!r}. "
                    f"Available: {sorted(self._strategies)}"
                )
            return self._strategies[explicit]

        strategy_name = _TIER_STRATEGY_MAP.get(quality_tier, "hybrid")

        if corpus_signals is not None:
            strategy_name = self._apply_corpus_overrides(
                strategy_name, quality_tier, corpus_signals,
            )

        return self._strategies[strategy_name]

    @staticmethod
    def _apply_corpus_overrides(
        strategy_name: str,
        quality_tier: str,
        signals: Any,
    ) -> str:
        """Override strategy based on corpus characteristics.

        Even with strong embeddings, short-doc / high-overlap corpora
        benefit from hybrid (BM25 + dense) rather than dense-only.
        """
        # Rule 1: Short documents favor hybrid even for strong embeddings
        short_frac = getattr(signals, "short_doc_fraction", 0.0)
        if quality_tier == "strong" and short_frac > 0.5:
            return "strong_hybrid"

        # Rule 2: High vocabulary overlap means BM25 is valuable
        overlap = getattr(signals, "vocab_overlap_ratio", 0.0)
        if quality_tier == "strong" and overlap > 0.3:
            return "strong_hybrid"

        # Rule 3: Very small corpus — sparse_boost's depth overshoots
        corpus_size = getattr(signals, "corpus_size", 0)
        if corpus_size < 1000 and strategy_name == "sparse_boost":
            return "hybrid"

        return strategy_name

    def register_strategy(self, strategy: PipelineStrategy) -> None:
        """Register a custom pipeline strategy."""
        self._strategies[strategy.name] = strategy

    @property
    def available(self) -> list[str]:
        """Names of all registered strategies."""
        return sorted(self._strategies)
