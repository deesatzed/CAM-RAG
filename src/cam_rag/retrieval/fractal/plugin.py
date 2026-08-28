"""Fractal multi-scale retriever as a CAM-RAG ``RetrieverPlugin``.

``FractalRetrieverPlugin`` indexes documents at sentence / paragraph /
document scales, computes derivative signals between levels, and retrieves
via ``FractalRAG.retrieve_adaptive``.  It is **opt-in**: register it on
``RAGAppSpec.retriever_plugins``; it is not added to the default spec.

Example::

    from cam_rag.rag.spec import RAGAppSpec
    from cam_rag.retrieval.fractal import FractalRetrieverPlugin, HashEmbedding

    plugin = FractalRetrieverPlugin(backend=HashEmbedding(dim=64))
    spec = RAGAppSpec(name="my-app", retriever_plugins=[plugin])
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from cam_rag.retrieval.fractal.core import EmbeddingBackend, HashEmbedding
from cam_rag.retrieval.fractal.engine import FractalRAG
from cam_rag.retrieval.models import RetrievalDocument, RetrievalResult

logger = logging.getLogger(__name__)


class _ListToArrayBackend:
    """Adapt a CAM-RAG ``EmbeddingBackend`` (list[float]) to fractal (ndarray)."""

    def __init__(self, backend: Any) -> None:
        self._backend = backend
        self._dim = int(backend.dim)

    def embed(self, text: str) -> np.ndarray:
        vec = self._backend.embed(text)
        return np.asarray(vec, dtype=np.float32)

    @property
    def dim(self) -> int:
        return self._dim


def _coerce_backend(backend: Any | None) -> EmbeddingBackend:
    """Accept fractal HashEmbedding, CAM-RAG HashEmbeddingBackend, or None."""
    if backend is None:
        return HashEmbedding()
    sample = backend.embed("probe")
    if isinstance(sample, np.ndarray):
        return backend
    return _ListToArrayBackend(backend)


class FractalRetrieverPlugin:
    """Multi-scale fractal retriever implementing ``RetrieverPlugin``.

    Parameters
    ----------
    backend:
        Embedding backend.  Defaults to ``HashEmbedding`` (deterministic,
        no GPU / sentence-transformers).  CAM-RAG ``HashEmbeddingBackend``
        and numpy-returning fractal backends are both accepted.
    adapter_strength:
        Per-document adapter magnitude passed to ``FractalRAG``.
    use_adaptive:
        When True (default), ``retrieve`` uses ``retrieve_adaptive``.
        When False, uses raw multi-scale ``retrieve``.
    """

    def __init__(
        self,
        backend: Any | None = None,
        *,
        adapter_strength: float = 0.25,
        use_adaptive: bool = True,
    ) -> None:
        self._backend = _coerce_backend(backend)
        self._adapter_strength = adapter_strength
        self._use_adaptive = use_adaptive
        self._rag = FractalRAG(
            backend=self._backend,
            adapter_strength=adapter_strength,
        )
        self._documents: list[RetrievalDocument] = []

    @property
    def name(self) -> str:
        return "fractal"

    @property
    def engine(self) -> FractalRAG:
        """The underlying ``FractalRAG`` engine (for tests / inspection)."""
        return self._rag

    def index(self, documents: list[RetrievalDocument]) -> None:
        """Build a fresh three-level fractal index from *documents*."""
        self._documents = list(documents)
        self._rag = FractalRAG(
            backend=self._backend,
            adapter_strength=self._adapter_strength,
        )
        for doc in documents:
            metadata = dict(doc.metadata) if doc.metadata else None
            title = None
            if metadata:
                raw_title = metadata.get("title")
                if isinstance(raw_title, str) and raw_title.strip():
                    title = raw_title
            self._rag.add_document(
                doc.doc_id,
                doc.text,
                metadata=metadata,
                title=title,
            )
        logger.info(
            "fractal index built: %d documents, L0=%d L1=%d L2=%d",
            len(documents),
            len(self._rag.index[0]),
            len(self._rag.index[1]),
            len(self._rag.index[2]),
        )

    def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]:
        """Retrieve top-*k* documents via adaptive fractal retrieval.

        Returns document-level (L2) hits as ``RetrievalResult`` so the plugin
        can participate in N-way RRF fusion alongside BM25 and dense.
        """
        if not query or not query.strip():
            return []
        if not self._rag.docs:
            return []

        if self._use_adaptive:
            results, qtype = self._rag.retrieve_adaptive(query, k=k)
        else:
            results, qtype = self._rag.retrieve(query, k=k)

        ranked = results.get(2, [])
        out: list[RetrievalResult] = []
        for rank, (entry, score) in enumerate(ranked, start=1):
            doc_id = entry.id
            text = self._rag.docs.get(doc_id, entry.text)
            meta = dict(self._rag.doc_metadata.get(doc_id, {}))
            meta["query_type"] = qtype
            meta["fractal_level"] = 2
            out.append(
                RetrievalResult(
                    doc_id=doc_id,
                    text=text,
                    score=float(score),
                    rank=rank,
                    metadata=meta,
                )
            )
        return out
