"""Tests for LocalSentenceTransformerBackend using real all-MiniLM-L6-v2."""

from __future__ import annotations

import math

import pytest

from cam_rag.finetuning.backend import LocalSentenceTransformerBackend, _auto_device

# ---------------------------------------------------------------------------
# Real model used for all tests (22M params, loads in <1s)
# ---------------------------------------------------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"


@pytest.fixture(scope="module")
def backend() -> LocalSentenceTransformerBackend:
    """Module-scoped backend so the model is loaded once for all tests."""
    return LocalSentenceTransformerBackend(MODEL_NAME)


# ---------------------------------------------------------------------------
# Construction & properties
# ---------------------------------------------------------------------------


def test_backend_loads_from_hub(backend: LocalSentenceTransformerBackend) -> None:
    assert backend.dim == 384
    assert backend.model_name_or_path == MODEL_NAME


def test_backend_device_is_string(backend: LocalSentenceTransformerBackend) -> None:
    assert backend._device in ("cpu", "mps", "cuda")


def test_auto_device_returns_string() -> None:
    device = _auto_device()
    assert isinstance(device, str)
    assert device in ("cpu", "mps", "cuda")


def test_backend_explicit_device() -> None:
    b = LocalSentenceTransformerBackend(MODEL_NAME, device="cpu")
    assert b._device == "cpu"
    assert b.dim == 384


def test_backend_custom_max_seq_length() -> None:
    b = LocalSentenceTransformerBackend(MODEL_NAME, max_seq_length=64)
    assert b._model.max_seq_length == 64


# ---------------------------------------------------------------------------
# embed()
# ---------------------------------------------------------------------------


def test_embed_returns_correct_dimension(backend: LocalSentenceTransformerBackend) -> None:
    vec = backend.embed("The patient presented with acute thrombocytopenia.")
    assert len(vec) == 384
    assert all(isinstance(v, float) for v in vec)


def test_embed_nonzero_vector(backend: LocalSentenceTransformerBackend) -> None:
    vec = backend.embed("Clinical pharmacology of metformin.")
    norm = math.sqrt(sum(v * v for v in vec))
    assert norm > 0.5  # normalized vectors have norm ~1.0


def test_embed_different_texts_differ(backend: LocalSentenceTransformerBackend) -> None:
    v1 = backend.embed("Aspirin reduces platelet aggregation.")
    v2 = backend.embed("The weather in Paris is lovely today.")
    # Cosine similarity should be significantly less than 1.0
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    cos_sim = dot / (n1 * n2)
    assert cos_sim < 0.8  # very different texts


def test_embed_similar_texts_similar(backend: LocalSentenceTransformerBackend) -> None:
    v1 = backend.embed("Metformin is used to treat type 2 diabetes.")
    v2 = backend.embed("Type 2 diabetes is commonly managed with metformin.")
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    cos_sim = dot / (n1 * n2)
    assert cos_sim > 0.7  # semantically similar


# ---------------------------------------------------------------------------
# embed_batch()
# ---------------------------------------------------------------------------


def test_embed_batch_returns_correct_shape(backend: LocalSentenceTransformerBackend) -> None:
    texts = [
        "Acute myocardial infarction treatment.",
        "Machine learning in healthcare.",
        "Antibiotic resistance mechanisms.",
    ]
    vecs = backend.embed_batch(texts)
    assert len(vecs) == 3
    assert all(len(v) == 384 for v in vecs)


def test_embed_batch_matches_individual(backend: LocalSentenceTransformerBackend) -> None:
    texts = ["Alpha protocol.", "Beta protocol."]
    batch_vecs = backend.embed_batch(texts)
    single_vecs = [backend.embed(t) for t in texts]
    # Should be identical (or nearly so due to float precision)
    for bv, sv in zip(batch_vecs, single_vecs):
        for a, b in zip(bv, sv):
            assert abs(a - b) < 1e-5


# ---------------------------------------------------------------------------
# Protocol compatibility
# ---------------------------------------------------------------------------


def test_satisfies_embedding_backend_protocol(backend: LocalSentenceTransformerBackend) -> None:
    """Backend has the dim attribute and embed method expected by EmbeddingBackend."""
    assert hasattr(backend, "dim")
    assert isinstance(backend.dim, int)
    assert callable(backend.embed)
    result = backend.embed("test")
    assert isinstance(result, list)
    assert all(isinstance(v, float) for v in result)
