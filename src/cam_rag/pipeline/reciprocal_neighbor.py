"""Reciprocal Neighbor Reranking (RNR) — coherence-based score adjustment.

After initial retrieval and reranking, documents that are semantically
similar to OTHER high-ranked documents receive a coherence boost.  The
intuition: if multiple top-ranked docs cluster around the same subtopic,
they reinforce each other's relevance.

This is a post-reranking step that uses the embedding backend to compute
pairwise similarity within the evidence set, then adjusts scores based on
how many "neighbors" each document has in the top tier.
"""

from __future__ import annotations

import logging

from cam_rag.pipeline.context import PipelineContext
from cam_rag.retrieval.dense import cosine_similarity

logger = logging.getLogger(__name__)


class ReciprocalNeighborStep:
    """Adjust evidence scores based on reciprocal neighbor coherence.

    For each evidence item, compute its average cosine similarity to the
    other top-ranked items.  Items with high average similarity to other
    top items get a coherence boost.
    """

    name = "reciprocal_neighbor_rerank"

    def execute(self, ctx: PipelineContext) -> None:
        if not ctx.evidence or len(ctx.evidence) < 3:
            return

        backend = ctx.spec.embedding_backend
        if backend is None:
            return

        # Only compute over the top tier (first 20 or all if fewer)
        tier = ctx.evidence[:20]
        n = len(tier)

        # Embed all evidence texts
        try:
            if hasattr(backend, "embed_batch"):
                vectors = backend.embed_batch([item.chunk.text for item in tier])
            else:
                vectors = [backend.embed(item.chunk.text) for item in tier]
        except Exception:
            logger.debug("RNR: embedding failed, skipping")
            return

        if not vectors or not vectors[0]:
            return

        # Compute pairwise similarities
        similarities = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                try:
                    sim = cosine_similarity(vectors[i], vectors[j])
                except (ValueError, ZeroDivisionError):
                    sim = 0.0
                similarities[i][j] = sim
                similarities[j][i] = sim

        # Compute coherence score: average similarity to other top docs
        for i, item in enumerate(tier):
            others = [similarities[i][j] for j in range(n) if j != i]
            if others:
                coherence = sum(others) / len(others)
                # Boost: items coherent with neighbors get up to 1.2x
                boost = 1.0 + 0.2 * max(0.0, coherence)
                item.score *= boost
                item.signals["rnr_coherence"] = round(coherence, 4)
                item.signals["rnr_boost"] = round(boost, 4)

        # Re-sort after boosting
        ctx.evidence.sort(key=lambda e: e.score, reverse=True)
        for i, item in enumerate(ctx.evidence):
            item.rank = i
