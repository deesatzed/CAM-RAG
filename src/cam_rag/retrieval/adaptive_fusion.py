"""Adaptive IDF-based fusion weight tuning.

When query terms have high IDF (rare in the corpus), BM25 is more
discriminative and should receive higher weight.  When query terms
are common, dense semantic retrieval adds more value.  This module
computes per-query adaptive sparse/dense weights using a sigmoid
mapping from mean query-term IDF.
"""

from __future__ import annotations

import math


def compute_idf_adaptive_weights(
    query_terms: list[str],
    document_frequencies: dict[str, int],
    num_documents: int,
    *,
    base_dense_weight: float = 0.6,
    base_sparse_weight: float = 0.4,
    alpha: float = 1.5,
    midpoint: float = 3.0,
) -> tuple[float, float]:
    """Compute adaptive sparse/dense fusion weights from query-term IDF.

    Parameters
    ----------
    query_terms : list[str]
        Tokenized query terms.
    document_frequencies : dict[str, int]
        Number of documents containing each term (from the BM25 index).
    num_documents : int
        Total number of documents in the corpus.
    base_dense_weight : float
        Default dense weight when IDF is at midpoint.
    base_sparse_weight : float
        Default sparse weight when IDF is at midpoint.
    alpha : float
        Controls the steepness of the sigmoid transition.
    midpoint : float
        IDF value at which weights are equal to base values.

    Returns
    -------
    tuple[float, float]
        ``(dense_weight, sparse_weight)`` normalised to sum to 1.0.
    """
    if not query_terms or num_documents <= 0:
        return base_dense_weight, base_sparse_weight

    # Compute IDF for each query term: log(N / (df + 1))
    idfs: list[float] = []
    for term in query_terms:
        df = document_frequencies.get(term, 0)
        idf = math.log((num_documents + 1) / (df + 1))
        idfs.append(idf)

    mean_idf = sum(idfs) / len(idfs)

    # Sigmoid: high IDF → boost sparse; low IDF → boost dense
    # sigmoid outputs 0..1; at midpoint → 0.5
    sigmoid = 1.0 / (1.0 + math.exp(-alpha * (mean_idf - midpoint)))

    # Interpolate: sigmoid=1 → max sparse; sigmoid=0 → max dense
    max_sparse_boost = 0.3  # Maximum shift toward sparse
    sparse_weight = base_sparse_weight + max_sparse_boost * (sigmoid - 0.5)
    dense_weight = 1.0 - sparse_weight

    # Clamp to valid range
    sparse_weight = max(0.1, min(0.9, sparse_weight))
    dense_weight = max(0.1, min(0.9, dense_weight))

    # Normalise
    total = sparse_weight + dense_weight
    return dense_weight / total, sparse_weight / total
