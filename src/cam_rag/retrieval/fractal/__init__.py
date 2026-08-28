"""Fractal multi-scale retrieval with derivative signals between levels.

Ported from fractLrag (deesatzed/fractLrag) into CAM-RAG as an optional
``RetrieverPlugin``.  The fractal index is **not** enabled by default —
callers must pass ``FractalRetrieverPlugin`` in ``RAGAppSpec.retriever_plugins``.

Documents are indexed at three self-similar levels (sentence / paragraph /
document).  First- and second-order derivative signals between levels act as
scoring bonuses.  ``retrieve_adaptive`` classifies the query type and routes
to flat, reranked, or RRF retrieval.

Hash embeddings work out of the box so tests do not need sentence-transformers
or a GPU.  Real semantic backends (e.g. BGE-M3) are optional.
"""

from cam_rag.retrieval.fractal.core import (
    EmbeddingBackend,
    HashEmbedding,
    SentenceTransformerEmbedding,
    normalize,
)
from cam_rag.retrieval.fractal.engine import FractalRAG, IndexEntry
from cam_rag.retrieval.fractal.plugin import FractalRetrieverPlugin
from cam_rag.retrieval.fractal.profile import DocumentProfile
from cam_rag.retrieval.fractal.query import (
    classify_query_type,
    extract_domain_hints,
    get_type_weights,
)
from cam_rag.retrieval.fractal.storage import DimensionMismatchError, load, save

__all__ = [
    "DimensionMismatchError",
    "DocumentProfile",
    "EmbeddingBackend",
    "FractalRAG",
    "FractalRetrieverPlugin",
    "HashEmbedding",
    "IndexEntry",
    "SentenceTransformerEmbedding",
    "classify_query_type",
    "extract_domain_hints",
    "get_type_weights",
    "load",
    "normalize",
    "save",
]
