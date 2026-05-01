"""BM25 anti-flooding step to prune low-IDF sparse matches.

When all matched query terms in a BM25 result have high document
frequency (appear in >10% of corpus), the match is likely a
"flood" — a document matching only common words.  This step prunes
such results before RRF fusion, reducing noise in the hybrid pipeline.
"""

from __future__ import annotations

import logging

from cam_rag.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)

# A term is "high-frequency" if its doc frequency exceeds this fraction
_HIGH_FREQ_THRESHOLD = 0.10  # 10% of corpus


class BM25AntiFloodStep:
    """Prune BM25 results where all matched query terms are high-frequency.

    Accesses ``ctx.sparse_retriever._document_frequencies`` to check
    term document frequencies.  For each sparse result, tokenizes the
    document text and finds which query terms matched.  If every matched
    term has ``doc_freq > 10% * corpus_size``, the result is pruned.

    Safety valve: if ALL query terms are high-frequency, pruning is
    skipped entirely (otherwise every result would be removed).
    """

    name = "bm25_antiflood"

    def execute(self, ctx: PipelineContext) -> None:
        if ctx.sparse_retriever is None:
            return
        if not ctx.sparse_results:
            return

        doc_freqs = getattr(ctx.sparse_retriever, "_document_frequencies", None)
        if doc_freqs is None:
            return

        corpus_size = len(ctx.sparse_retriever.documents)
        if corpus_size == 0:
            return

        threshold = _HIGH_FREQ_THRESHOLD * corpus_size
        query_terms = set(ctx.spec.tokenize(ctx.expanded_query))
        if not query_terms:
            return

        # Safety valve: if ALL query terms are high-frequency, skip pruning
        all_high_freq = all(
            doc_freqs.get(term, 0) > threshold for term in query_terms
        )
        if all_high_freq:
            return

        # Check if any query terms are high-frequency at all
        high_freq_terms = {
            term for term in query_terms if doc_freqs.get(term, 0) > threshold
        }
        if not high_freq_terms:
            # No high-frequency terms exist → nothing to prune
            return

        filtered = []
        pruned = 0
        for result in ctx.sparse_results:
            doc_terms = set(ctx.spec.tokenize(result.text))
            matched = query_terms & doc_terms
            if not matched:
                filtered.append(result)
                continue
            # If ALL matched terms are high-freq → prune this result
            if matched <= high_freq_terms:
                pruned += 1
                continue
            filtered.append(result)

        if pruned > 0:
            logger.debug(
                "bm25_antiflood: pruned %d/%d sparse results",
                pruned,
                len(ctx.sparse_results),
            )
        ctx.sparse_results = filtered
