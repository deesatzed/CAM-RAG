"""CAM-RAG: Compliance-Aware Retrieval-Augmented Generation toolkit.

Deterministic retrieval, embedding-based retrieval, and utilities for
regulated-domain RAG pipelines.
"""

from cam_rag.deterministic import DeterministicRetriever, Document, RetrievalResult
from cam_rag.rag import (
    Chunk,
    Citation,
    CorpusDocument,
    Evidence,
    RAGAnswer,
    RAGAppSpec,
    RAGPolicy,
    RAGTrace,
)

__all__ = [
    "Chunk",
    "Citation",
    "CorpusDocument",
    "DeterministicRetriever",
    "Document",
    "Evidence",
    "RAGAnswer",
    "RAGAppSpec",
    "RAGPolicy",
    "RAGTrace",
    "RetrievalResult",
]
