"""Pipeline strategy definitions and adaptive routing.

Defines three pre-built pipeline strategies based on benchmark evidence:

- **dense_dominant**: Skip BM25/RRF, use dense retrieval only + reranker.
  For strong embeddings (>= 2048d) where BM25 fusion dilutes quality.
- **hybrid**: Full BM25 + dense + RRF + reranker pipeline. For moderate
  embeddings where sparse and dense signals complement each other.
- **sparse_boost**: Boost BM25 weight for weak embeddings or
  keyword-heavy domains.

The ``StrategyRouter`` selects the appropriate strategy based on an
``EmbeddingQualityTier`` or an explicit spec override.
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

    def select(self, quality_tier: str, spec: Any) -> PipelineStrategy:
        """Select a pipeline strategy.

        Parameters
        ----------
        quality_tier:
            One of ``"strong"``, ``"moderate"``, ``"weak"``.
        spec:
            A ``RAGAppSpec`` (or anything with a ``pipeline_strategy`` attribute).
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
        return self._strategies[strategy_name]

    def register_strategy(self, strategy: PipelineStrategy) -> None:
        """Register a custom pipeline strategy."""
        self._strategies[strategy.name] = strategy

    @property
    def available(self) -> list[str]:
        """Names of all registered strategies."""
        return sorted(self._strategies)
