"""Query decomposition for multi-aspect retrieval.

Some queries contain multiple aspects (e.g., "aspirin reduces inflammation
and prevents clots").  This step decomposes queries at conjunction boundaries,
runs BM25 retrieval for each sub-query, and merges results.  This helps find
documents that match individual aspects rather than requiring all terms.

This is a lightweight heuristic decomposition (no LLM needed) that detects
'and', 'or', commas, and semicolons as aspect boundaries.
"""

from __future__ import annotations

import re

from cam_rag.pipeline.context import PipelineContext
from cam_rag.rag.models import Evidence

# Conjunction patterns that often separate query aspects
_SPLIT_RE = re.compile(
    r"\b(?:and|or|as well as|in addition to|along with)\b|[;,]",
    re.IGNORECASE,
)


class QueryDecomposeStep:
    """Decompose multi-aspect queries and merge sub-query retrievals.

    Only activates when the query contains detectable aspect boundaries.
    For single-aspect queries, this is a no-op.
    """

    name = "query_decompose"

    def execute(self, ctx: PipelineContext) -> None:
        if not ctx.evidence:
            return
        if ctx.sparse_retriever is None:
            return

        sub_queries = _decompose(ctx.query_text)
        if len(sub_queries) <= 1:
            return  # Single-aspect query — nothing to decompose

        # Run BM25 for each sub-query
        existing_ids = {item.chunk.id for item in ctx.evidence}
        new_evidence: list[Evidence] = []

        for sub_q in sub_queries:
            sub_q = sub_q.strip()
            if len(sub_q) < 5:  # Skip trivially short fragments
                continue

            sparse_results = ctx.sparse_retriever.retrieve(sub_q, k=20)
            query_terms = set(ctx.spec.tokenize(sub_q))

            for result in sparse_results:
                if result.doc_id in existing_ids:
                    continue
                chunk = ctx.chunk_by_id.get(result.doc_id)
                if chunk is None:
                    continue

                matched = sorted(query_terms & set(ctx.spec.tokenize(chunk.text)))
                new_evidence.append(
                    Evidence(
                        chunk=chunk,
                        score=result.score * 0.8,  # Slight discount for sub-query matches
                        retriever="query_decompose",
                        rank=0,
                        signals={
                            "sub_query": sub_q,
                            "matched_terms": matched,
                        },
                    )
                )
                existing_ids.add(result.doc_id)

        if new_evidence:
            ctx.evidence.extend(new_evidence)
            ctx.evidence.sort(key=lambda e: e.score, reverse=True)
            for i, item in enumerate(ctx.evidence):
                item.rank = i


def _decompose(query: str) -> list[str]:
    """Split a query into sub-aspects at conjunction boundaries."""
    parts = _SPLIT_RE.split(query)
    # Filter out empty/whitespace-only parts
    return [p.strip() for p in parts if p.strip()]
