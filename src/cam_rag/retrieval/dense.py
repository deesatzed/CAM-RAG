"""Dependency-free dense retrieval protocol and hash embedding backend."""

from __future__ import annotations

import hashlib
import logging
import math
from collections.abc import Iterable, Sequence
from typing import Protocol

from cam_rag.retrieval.models import RetrievalDocument, RetrievalResult
from cam_rag.retrieval.sparse import tokenize

logger = logging.getLogger(__name__)


class EmbeddingBackend(Protocol):
    """Minimal embedding backend contract for dense retrieval."""

    dim: int

    def embed(self, text: str) -> list[float]:
        """Return a numeric embedding vector for text."""


class HashEmbeddingBackend:
    """Deterministic feature-hash embedding backend for smoke tests.

    This is not intended to be a SOTA semantic model. It provides the
    same interface as a real embedding backend so app and pipeline code
    can be wired before optional model-backed dense retrievers are
    introduced.
    """

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for term in tokenize(text):
            index = _stable_index(term, self.dim)
            sign = (
                -1.0
                if _stable_index(f"{term}:sign", 2)
                else 1.0
            )
            vector[index] += sign
        return _normalize(vector)


class DenseVectorRetriever:
    """In-memory dense retriever over embedded documents."""

    def __init__(
        self,
        documents: Iterable[RetrievalDocument],
        *,
        backend: EmbeddingBackend | None = None,
    ) -> None:
        self.backend = backend or HashEmbeddingBackend()
        self.documents = list(documents)
        self._vectors = [
            self._safe_embed(doc) for doc in self.documents
        ]

    def _safe_embed(
        self, doc: RetrievalDocument
    ) -> list[float]:
        """Embed a document, returning a zero vector on failure."""
        try:
            return self.backend.embed(doc.text)
        except Exception:
            logger.warning(
                "embedding failed for document %s, "
                "using zero vector",
                doc.doc_id,
            )
            return [0.0] * self.backend.dim

    def retrieve(
        self, query: str, k: int = 20
    ) -> list[RetrievalResult]:
        """Return top-*k* docs by cosine similarity to *query*."""
        if k <= 0 or not self.documents:
            return []
        query_vector = self.backend.embed(query)
        if not any(query_vector):
            return []

        scored: list[tuple[int, float]] = []
        for index, vector in enumerate(self._vectors):
            score = cosine_similarity(query_vector, vector)
            if score > 0:
                scored.append((index, score))

        scored.sort(
            key=lambda item: (
                -item[1],
                self.documents[item[0]].doc_id,
            )
        )
        return [
            RetrievalResult(
                doc_id=self.documents[index].doc_id,
                text=self.documents[index].text,
                score=score,
                rank=rank,
                metadata=dict(self.documents[index].metadata),
            )
            for rank, (index, score) in enumerate(
                scored[:k], start=1
            )
        ]


def cosine_similarity(
    left: Sequence[float], right: Sequence[float]
) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same length")
    left_norm = math.sqrt(
        sum(value * value for value in left)
    )
    right_norm = math.sqrt(
        sum(value * value for value in right)
    )
    if left_norm == 0 or right_norm == 0:
        return 0.0
    dot = sum(
        l_value * r_value
        for l_value, r_value in zip(left, right, strict=True)
    )
    return dot / (left_norm * right_norm)


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _stable_index(value: str, modulo: int) -> int:
    digest = hashlib.blake2b(
        value.encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") % modulo
