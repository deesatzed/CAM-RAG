"""Embedding quality detection for adaptive pipeline strategy routing.

Classifies an embedding backend into quality tiers (strong/moderate/weak)
based on dimensionality, intra-corpus dispersion, and cluster separation.
The tier determines which pipeline strategy (dense_dominant, hybrid,
sparse_boost) is used for retrieval.

Detection runs once at index time on a small document sample.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EmbeddingQualityTier:
    """Result of embedding quality analysis."""

    tier: str  # "strong" | "moderate" | "weak"
    dimensionality: int
    dispersion: float  # intra-corpus cosine variance (higher = more discriminative)
    cluster_separation: float  # silhouette-like score (higher = better clusters)
    confidence: float  # 0-1 confidence in tier assignment


class EmbeddingQualityDetector:
    """Detect embedding quality from a corpus sample.

    Three signals are computed (no LLM calls):

    1. **Dimensionality**: Higher dim generally means richer representations.
       dim >= 2048 is a strong candidate; dim < 512 is a weak candidate.

    2. **Embedding dispersion**: Pairwise cosine variance over a sample.
       High dispersion means embeddings spread across the space
       (discriminative). Low dispersion means embeddings cluster in a
       narrow cone (less useful for retrieval).

    3. **Cluster separation**: k-means silhouette-like score on the sample.
       High separation means semantically distinct documents get distinct
       embeddings. Low separation means the model can't differentiate
       document topics.
    """

    def __init__(
        self,
        *,
        dim_strong_threshold: int = 2048,
        dim_weak_threshold: int = 512,
        dispersion_strong_threshold: float = 0.3,
        dispersion_weak_threshold: float = 0.15,
        separation_strong_threshold: float = 0.4,
        separation_weak_threshold: float = 0.2,
    ) -> None:
        self._dim_strong = dim_strong_threshold
        self._dim_weak = dim_weak_threshold
        self._disp_strong = dispersion_strong_threshold
        self._disp_weak = dispersion_weak_threshold
        self._sep_strong = separation_strong_threshold
        self._sep_weak = separation_weak_threshold

    def detect(
        self,
        backend: Any,
        sample_docs: list[str],
        *,
        sample_size: int = 200,
        seed: int = 42,
    ) -> EmbeddingQualityTier:
        """Classify the embedding backend's quality on the given corpus sample.

        Parameters
        ----------
        backend:
            Any object with a ``dim`` attribute and ``embed(text) -> list[float]``
            method. Optionally ``embed_batch(texts) -> list[list[float]]``.
        sample_docs:
            Corpus texts to sample from.
        sample_size:
            Maximum number of documents to embed for analysis.
        seed:
            Random seed for reproducible sampling.
        """
        dim = getattr(backend, "dim", 0)

        # Sample documents
        if len(sample_docs) > sample_size:
            rng = random.Random(seed)
            sample_texts = rng.sample(sample_docs, sample_size)
        else:
            sample_texts = list(sample_docs)

        if not sample_texts:
            return EmbeddingQualityTier(
                tier="weak",
                dimensionality=dim,
                dispersion=0.0,
                cluster_separation=0.0,
                confidence=0.5,
            )

        # Embed sample
        vectors = self._embed_sample(backend, sample_texts)
        if not vectors:
            return EmbeddingQualityTier(
                tier="weak",
                dimensionality=dim,
                dispersion=0.0,
                cluster_separation=0.0,
                confidence=0.5,
            )

        # Signal 1: dimensionality (already known)
        # Signal 2: embedding dispersion
        dispersion = _compute_dispersion(vectors)
        # Signal 3: cluster separation
        separation = _compute_cluster_separation(vectors, k=5, seed=seed)

        # Classify tier
        tier, confidence = self._classify(dim, dispersion, separation)

        logger.info(
            "Embedding quality: tier=%s dim=%d dispersion=%.4f "
            "separation=%.4f confidence=%.2f",
            tier, dim, dispersion, separation, confidence,
        )

        return EmbeddingQualityTier(
            tier=tier,
            dimensionality=dim,
            dispersion=dispersion,
            cluster_separation=separation,
            confidence=confidence,
        )

    def _embed_sample(
        self, backend: Any, texts: list[str]
    ) -> list[list[float]]:
        """Embed sample texts using batch or single embed."""
        if hasattr(backend, "embed_batch"):
            try:
                return backend.embed_batch(texts)
            except Exception:
                logger.warning(
                    "embed_batch failed in quality detector, falling back"
                )
        vectors = []
        for text in texts:
            try:
                vectors.append(backend.embed(text))
            except Exception:
                continue
        return vectors

    def _classify(
        self,
        dim: int,
        dispersion: float,
        separation: float,
    ) -> tuple[str, float]:
        """Map signals to tier and confidence."""
        # Count how many signals point to each tier
        strong_signals = 0
        weak_signals = 0
        total_signals = 3

        # Dimensionality signal
        if dim >= self._dim_strong:
            strong_signals += 1
        elif dim < self._dim_weak:
            weak_signals += 1

        # Dispersion signal
        if dispersion > self._disp_strong:
            strong_signals += 1
        elif dispersion < self._disp_weak:
            weak_signals += 1

        # Separation signal
        if separation > self._sep_strong:
            strong_signals += 1
        elif separation < self._sep_weak:
            weak_signals += 1

        # Classify based on majority voting
        if strong_signals >= 2:
            tier = "strong"
            confidence = strong_signals / total_signals
        elif weak_signals >= 2:
            tier = "weak"
            confidence = weak_signals / total_signals
        else:
            tier = "moderate"
            # Confidence is lower when signals disagree
            confidence = 1.0 - (strong_signals + weak_signals) / total_signals

        return tier, round(confidence, 4)


def _compute_dispersion(vectors: list[list[float]]) -> float:
    """Compute pairwise cosine distance variance (dispersion).

    Higher dispersion means embeddings are spread across the space,
    which indicates the model can differentiate documents.

    Samples up to 100 pairs for efficiency.
    """
    n = len(vectors)
    if n < 2:
        return 0.0

    # Sample pairs for efficiency
    max_pairs = min(100, n * (n - 1) // 2)
    rng = random.Random(42)

    similarities: list[float] = []
    indices = list(range(n))

    for _ in range(max_pairs):
        i, j = rng.sample(indices, 2)
        sim = _cosine_sim(vectors[i], vectors[j])
        similarities.append(sim)

    if not similarities:
        return 0.0

    # Variance of pairwise similarities
    mean_sim = sum(similarities) / len(similarities)
    variance = sum((s - mean_sim) ** 2 for s in similarities) / len(similarities)
    return variance


def _compute_cluster_separation(
    vectors: list[list[float]],
    k: int = 5,
    seed: int = 42,
    max_iter: int = 20,
) -> float:
    """Compute a simplified silhouette-like cluster separation score.

    Uses k-means with k clusters on the embedding vectors, then computes
    a simplified silhouette coefficient: for each point, (b - a) / max(a, b)
    where a = mean distance to same-cluster points, b = mean distance to
    nearest other cluster.

    Returns a value in [-1, 1] where higher is better. Typical good
    embeddings score 0.3-0.6.
    """
    n = len(vectors)
    if n < k + 1:
        return 0.0

    dim = len(vectors[0])
    rng = random.Random(seed)

    # Initialize centroids from random samples
    centroid_indices = rng.sample(range(n), k)
    centroids = [list(vectors[i]) for i in centroid_indices]

    # k-means iterations
    assignments = [0] * n
    for _ in range(max_iter):
        # Assign each vector to nearest centroid
        changed = False
        for i in range(n):
            best_cluster = 0
            best_dist = float("inf")
            for c in range(k):
                dist = _euclidean_dist(vectors[i], centroids[c])
                if dist < best_dist:
                    best_dist = dist
                    best_cluster = c
            if assignments[i] != best_cluster:
                changed = True
                assignments[i] = best_cluster

        if not changed:
            break

        # Update centroids
        for c in range(k):
            members = [vectors[i] for i in range(n) if assignments[i] == c]
            if members:
                centroids[c] = [
                    sum(m[d] for m in members) / len(members)
                    for d in range(dim)
                ]

    # Compute simplified silhouette
    # For efficiency, sample up to 100 points
    sample_indices = (
        rng.sample(range(n), min(100, n)) if n > 100 else list(range(n))
    )

    silhouette_scores: list[float] = []
    for i in sample_indices:
        cluster_i = assignments[i]

        # a = mean distance to same-cluster points (sample up to 20)
        same_cluster = [
            j for j in range(n) if assignments[j] == cluster_i and j != i
        ]
        if not same_cluster:
            continue
        if len(same_cluster) > 20:
            same_cluster = rng.sample(same_cluster, 20)
        a = sum(_euclidean_dist(vectors[i], vectors[j]) for j in same_cluster) / len(
            same_cluster
        )

        # b = min mean distance to any other cluster
        b = float("inf")
        for c in range(k):
            if c == cluster_i:
                continue
            other_cluster = [j for j in range(n) if assignments[j] == c]
            if not other_cluster:
                continue
            if len(other_cluster) > 20:
                other_cluster = rng.sample(other_cluster, 20)
            mean_dist = sum(
                _euclidean_dist(vectors[i], vectors[j]) for j in other_cluster
            ) / len(other_cluster)
            b = min(b, mean_dist)

        if b == float("inf"):
            continue

        denom = max(a, b)
        if denom > 0:
            silhouette_scores.append((b - a) / denom)

    if not silhouette_scores:
        return 0.0

    return sum(silhouette_scores) / len(silhouette_scores)


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _euclidean_dist(a: list[float], b: list[float]) -> float:
    """Euclidean distance between two vectors."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
