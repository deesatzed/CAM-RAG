"""Intersection-boost scoring step for evidence found by multiple retrievers.

When a document is retrieved by both sparse (BM25) and dense retrieval,
it signals strong relevance — the document matches both lexically and
semantically.  This step applies a multiplicative boost to such
documents, similar to ``title_boost.py``.
"""

from __future__ import annotations

from cam_rag.pipeline.context import PipelineContext

_BOOST_FACTOR = 1.2


class IntersectionBoostStep:
    """Boost evidence scores for documents found by both sparse and dense retrievers.

    Checks ``signals["source_ranks"]`` for each evidence item.  If both
    a ``"sparse"`` key and any dense key (``"dense"``, ``"dense_0"``,
    ``"dense_1"``, or ``"plugin_*"``) are present, the item receives a
    1.2x multiplicative boost.  Records ``signals["intersection_boost"]``
    as True/False and re-sorts evidence after boosting.
    """

    name = "intersection_boost"

    def execute(self, ctx: PipelineContext) -> None:
        if not ctx.evidence:
            return

        for item in ctx.evidence:
            source_ranks = item.signals.get("source_ranks", {})
            has_sparse = "sparse" in source_ranks
            has_dense = any(
                k.startswith("dense") or k.startswith("plugin_")
                for k in source_ranks
                if k != "sparse"
            )
            if has_sparse and has_dense:
                item.score *= _BOOST_FACTOR
                item.signals["intersection_boost"] = True
            else:
                item.signals["intersection_boost"] = False

        # Re-sort after boosting
        ctx.evidence.sort(key=lambda e: e.score, reverse=True)
        for i, item in enumerate(ctx.evidence):
            item.rank = i
