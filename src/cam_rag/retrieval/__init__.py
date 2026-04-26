"""Lightweight reusable retrieval primitives."""

from cam_rag.retrieval.adaptive_params import AdaptiveParams, compute_adaptive_params
from cam_rag.retrieval.fusion import rrf_fuse
from cam_rag.retrieval.models import FusedResult, RetrievalDocument, RetrievalResult
from cam_rag.retrieval.sparse import SparseBM25Retriever, bm25_search, tokenize

__all__ = [
    "AdaptiveParams",
    "FusedResult",
    "RetrievalDocument",
    "RetrievalResult",
    "SparseBM25Retriever",
    "bm25_search",
    "compute_adaptive_params",
    "rrf_fuse",
    "tokenize",
]
