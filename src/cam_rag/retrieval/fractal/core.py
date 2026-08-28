"""
Shared primitives for Fractal Latent RAG.

All embedding, normalization, and chunking logic lives here — ONE copy.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Protocol for pluggable embedding backends."""
    def embed(self, text: str) -> np.ndarray: ...
    @property
    def dim(self) -> int: ...


class HashEmbedding:
    """Deterministic MD5-seeded random vectors. For testing and reproducibility only.
    These vectors have NO semantic meaning — cosine similarity is random noise."""

    def __init__(self, dim: int = 64):
        self._dim = dim

    def embed(self, text: str) -> np.ndarray:
        seed = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self._dim).astype(np.float32)
        return vec / np.linalg.norm(vec)

    @property
    def dim(self) -> int:
        return self._dim


class SentenceTransformerEmbedding:
    """Real semantic embeddings via sentence-transformers.
    Default model: BAAI/bge-m3 (1024 dim, multi-granularity native).
    """

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:
        return self._model.encode(text, normalize_embeddings=True).astype(np.float32)

    @property
    def dim(self) -> int:
        return self._dim


def normalize(vec: np.ndarray) -> np.ndarray:
    """Safe L2 normalization. Returns zero vector on zero-norm input."""
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def chunk_fractal(text: str) -> tuple[list[str], list[str], str]:
    """Fractal chunking: sentences (L0) -> paragraphs (L1) -> full doc (L2).

    Sentence splitting uses regex boundaries that handle decimals, percentages,
    abbreviations, and statistical notation common in scientific text.
    Fragments shorter than 30 chars are merged into the preceding sentence.
    """
    import re

    # Split on period/!/? followed by whitespace and an uppercase letter.
    # This avoids splitting on decimals (0.79), abbreviations (e.g.), and
    # statistical notation (P = 0.001).
    raw = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    # Clean and filter
    sentences = [s.strip() for s in raw if s.strip()]

    # Merge short fragments (< 30 chars) into previous sentence
    if len(sentences) > 1:
        merged = [sentences[0]]
        for s in sentences[1:]:
            if len(s) < 30 and merged:
                merged[-1] = merged[-1] + ' ' + s
            else:
                merged.append(s)
        sentences = merged

    # Ensure non-empty
    if not sentences:
        sentences = [text] if text.strip() else []

    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()] or [text]

    # Virtual paragraph chunking for single-paragraph docs
    # When \n\n split produces only 1 paragraph but the doc has 4+ sentences,
    # group sentences into virtual paragraphs of ~3 sentences each.
    # This prevents L1 from being degenerate (identical to L2).
    if len(paragraphs) == 1 and len(sentences) >= 4:
        chunk_size = 3
        virtual_paras = []
        for i in range(0, len(sentences), chunk_size):
            group = sentences[i:i + chunk_size]
            virtual_paras.append(' '.join(group))
        paragraphs = virtual_paras

    return sentences, paragraphs, text
