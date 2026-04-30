"""Tests for EmbeddingQualityDetector and EmbeddingQualityTier."""

from __future__ import annotations

import hashlib
import math
import random

from cam_rag.retrieval.dense import HashEmbeddingBackend
from cam_rag.retrieval.quality_detector import (
    EmbeddingQualityDetector,
    EmbeddingQualityTier,
    _compute_cluster_separation,
    _compute_dispersion,
    _cosine_sim,
)


def _make_docs(n: int = 50) -> list[str]:
    """Generate simple varied documents for testing."""
    topics = [
        "The cat sat on the mat and watched the birds fly by",
        "Machine learning models use gradient descent for optimization",
        "Photosynthesis converts sunlight into chemical energy in plants",
        "The stock market fluctuated wildly during the economic crisis",
        "Quantum computing uses qubits that exist in superposition states",
    ]
    docs = []
    for i in range(n):
        base = topics[i % len(topics)]
        docs.append(f"{base} — document {i}")
    return docs


class TestEmbeddingQualityTier:
    def test_tier_fields(self) -> None:
        tier = EmbeddingQualityTier(
            tier="strong",
            dimensionality=4096,
            dispersion=0.35,
            cluster_separation=0.45,
            confidence=0.8,
        )
        assert tier.tier == "strong"
        assert tier.dimensionality == 4096
        assert tier.dispersion == 0.35
        assert tier.cluster_separation == 0.45
        assert tier.confidence == 0.8

    def test_tier_is_frozen(self) -> None:
        tier = EmbeddingQualityTier(
            tier="weak", dimensionality=256,
            dispersion=0.1, cluster_separation=0.1, confidence=0.5,
        )
        try:
            tier.tier = "strong"  # type: ignore[misc]
            assert False, "Should have raised"
        except AttributeError:
            pass


class TestDispersion:
    def test_single_vector(self) -> None:
        assert _compute_dispersion([[1.0, 0.0, 0.0]]) == 0.0

    def test_identical_vectors(self) -> None:
        vecs = [[1.0, 0.0, 0.0]] * 10
        assert _compute_dispersion(vecs) == 0.0

    def test_diverse_vectors_higher_dispersion(self) -> None:
        # Orthogonal vectors should have higher dispersion than identical
        rng = random.Random(42)
        diverse = []
        for _ in range(20):
            v = [rng.gauss(0, 1) for _ in range(10)]
            norm = math.sqrt(sum(x * x for x in v))
            diverse.append([x / norm for x in v])
        disp = _compute_dispersion(diverse)
        assert disp > 0.0

    def test_empty(self) -> None:
        assert _compute_dispersion([]) == 0.0


class TestClusterSeparation:
    def test_well_separated_clusters(self) -> None:
        # Create 3 clusters that are far apart in 10d space
        vectors: list[list[float]] = []
        rng = random.Random(42)
        for cluster_id in range(3):
            center = [0.0] * 10
            center[cluster_id * 3] = 10.0  # very distinct
            for _ in range(10):
                v = [c + rng.gauss(0, 0.1) for c in center]
                vectors.append(v)

        sep = _compute_cluster_separation(vectors, k=3, seed=42)
        # Well-separated clusters should have positive silhouette
        assert sep > 0.3

    def test_random_cloud_low_separation(self) -> None:
        # Random gaussian cloud: clusters are meaningless
        rng = random.Random(42)
        vectors = [[rng.gauss(0, 1) for _ in range(10)] for _ in range(30)]
        sep = _compute_cluster_separation(vectors, k=5, seed=42)
        # Should be relatively low (not strongly clustered)
        assert sep < 0.5

    def test_too_few_vectors(self) -> None:
        assert _compute_cluster_separation([[1.0, 2.0]], k=5) == 0.0


class TestEmbeddingQualityDetector:
    def test_hash_backend_classified_as_weak(self) -> None:
        """HashEmbeddingBackend (256d, random-ish) should be weak."""
        backend = HashEmbeddingBackend(dim=256)
        detector = EmbeddingQualityDetector()
        docs = _make_docs(50)
        tier = detector.detect(backend, docs, sample_size=50)
        assert tier.tier == "weak"
        assert tier.dimensionality == 256
        assert 0.0 <= tier.confidence <= 1.0

    def test_high_dim_high_dispersion_is_strong(self) -> None:
        """A backend with dim=4096 and semantically-structured embeddings
        should be classified as strong."""

        class HighDimBackend:
            dim = 4096

            def embed(self, text: str) -> list[float]:
                # Simulate a real embedding model: documents about the same
                # topic get similar embeddings, different topics get different
                # embeddings. This produces high dispersion and high cluster
                # separation.
                rng = random.Random(text)
                v = [0.0] * self.dim
                # Topic signal: first 100 dims encode topic via deterministic hash
                topic_key = text.split("—")[0] if "—" in text else text[:20]
                topic_hash = int(hashlib.md5(topic_key.encode()).hexdigest(), 16)
                topic_rng = random.Random(topic_hash)
                for i in range(100):
                    v[i] = topic_rng.gauss(0, 3.0)
                # Noise dims: add small variation
                for i in range(100, self.dim):
                    v[i] = rng.gauss(0, 0.1)
                norm = math.sqrt(sum(x * x for x in v))
                return [x / norm for x in v] if norm > 0 else v

        backend = HighDimBackend()
        detector = EmbeddingQualityDetector()
        docs = _make_docs(50)
        tier = detector.detect(backend, docs, sample_size=50)
        assert tier.tier == "strong"
        assert tier.dimensionality == 4096

    def test_moderate_dim_moderate_dispersion(self) -> None:
        """A backend with dim=768 and topic-structured embeddings should
        be moderate (dim is between thresholds but signals are decent)."""

        class ModerateDimBackend:
            dim = 768

            def embed(self, text: str) -> list[float]:
                rng = random.Random(text)
                v = [0.0] * self.dim
                # Topic signal (weaker than strong backend)
                topic_key = text.split("—")[0] if "—" in text else text[:20]
                topic_hash = int(hashlib.md5(topic_key.encode()).hexdigest(), 16)
                topic_rng = random.Random(topic_hash)
                for i in range(50):
                    v[i] = topic_rng.gauss(0, 2.0)
                for i in range(50, self.dim):
                    v[i] = rng.gauss(0, 0.3)
                norm = math.sqrt(sum(x * x for x in v))
                return [x / norm for x in v] if norm > 0 else v

        backend = ModerateDimBackend()
        detector = EmbeddingQualityDetector()
        docs = _make_docs(50)
        tier = detector.detect(backend, docs, sample_size=50)
        # 768d is between weak and strong thresholds
        assert tier.tier in ("moderate", "strong")

    def test_empty_corpus(self) -> None:
        backend = HashEmbeddingBackend(dim=256)
        detector = EmbeddingQualityDetector()
        tier = detector.detect(backend, [], sample_size=50)
        assert tier.tier == "weak"
        assert tier.confidence == 0.5

    def test_deterministic_with_same_seed(self) -> None:
        backend = HashEmbeddingBackend(dim=256)
        detector = EmbeddingQualityDetector()
        docs = _make_docs(100)
        tier1 = detector.detect(backend, docs, seed=42)
        tier2 = detector.detect(backend, docs, seed=42)
        assert tier1 == tier2

    def test_sample_size_limits_embeddings(self) -> None:
        """With sample_size=10 on 100 docs, only 10 should be embedded."""
        call_count = 0

        class CountingBackend:
            dim = 256

            def embed(self, text: str) -> list[float]:
                nonlocal call_count
                call_count += 1
                return [0.0] * self.dim

        backend = CountingBackend()
        detector = EmbeddingQualityDetector()
        docs = _make_docs(100)
        detector.detect(backend, docs, sample_size=10)
        assert call_count == 10

    def test_uses_embed_batch_when_available(self) -> None:
        """Detector should use embed_batch when backend supports it."""
        batch_called = False

        class BatchBackend:
            dim = 256

            def embed(self, text: str) -> list[float]:
                return [0.0] * self.dim

            def embed_batch(self, texts: list[str]) -> list[list[float]]:
                nonlocal batch_called
                batch_called = True
                return [[0.0] * self.dim for _ in texts]

        backend = BatchBackend()
        detector = EmbeddingQualityDetector()
        docs = _make_docs(20)
        detector.detect(backend, docs, sample_size=20)
        assert batch_called

    def test_confidence_between_0_and_1(self) -> None:
        backend = HashEmbeddingBackend(dim=256)
        detector = EmbeddingQualityDetector()
        docs = _make_docs(30)
        tier = detector.detect(backend, docs, sample_size=30)
        assert 0.0 <= tier.confidence <= 1.0


class TestCosineSim:
    def test_identical_vectors(self) -> None:
        v = [1.0, 2.0, 3.0]
        assert abs(_cosine_sim(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self) -> None:
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(_cosine_sim(a, b)) < 1e-6

    def test_zero_vector(self) -> None:
        assert _cosine_sim([0.0, 0.0], [1.0, 0.0]) == 0.0
