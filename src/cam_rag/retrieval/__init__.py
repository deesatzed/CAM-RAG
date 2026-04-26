"""Lightweight reusable retrieval primitives."""

from cam_rag.retrieval.adaptive_params import AdaptiveParams, compute_adaptive_params
from cam_rag.retrieval.dense import (
    DenseVectorRetriever,
    EmbeddingBackend,
    HashEmbeddingBackend,
    cosine_similarity,
)
from cam_rag.retrieval.fusion import rrf_fuse
from cam_rag.retrieval.models import FusedResult, RetrievalDocument, RetrievalResult
from cam_rag.retrieval.query_expansion import (
    DEFAULT_EXPANSION_STOPWORDS,
    build_expanded_query,
    extract_expansion_terms,
    filter_expansion_terms,
    tokenize_expansion_text,
)
from cam_rag.retrieval.sparse import SparseBM25Retriever, bm25_search, tokenize

__all__ = [
    "AdaptiveParams",
    "DEFAULT_EXPANSION_STOPWORDS",
    "DenseVectorRetriever",
    "EmbeddingBackend",
    "FusedResult",
    "HashEmbeddingBackend",
    "RetrievalDocument",
    "RetrievalResult",
    "SparseBM25Retriever",
    "bm25_search",
    "compute_adaptive_params",
    "cosine_similarity",
    "build_expanded_query",
    "extract_expansion_terms",
    "filter_expansion_terms",
    "rrf_fuse",
    "tokenize",
    "tokenize_expansion_text",
]
