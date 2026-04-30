"""Title-boost scoring step for evidence with titled documents.

When chunks have a ``title`` field, query terms appearing in the title
indicate a much stronger topical match than body-only matches.  This step
boosts evidence scores for title-matching documents.
"""

from __future__ import annotations

from cam_rag.pipeline.context import PipelineContext


class TitleBoostStep:
    """Boost evidence scores when query terms appear in chunk titles.

    Adds a multiplicative boost to evidence items whose chunk title
    shares terms with the query.  The boost is proportional to the
    fraction of query terms found in the title.  No-op when chunks
    have no titles or when the boost is disabled via spec.
    """

    name = "title_boost"

    def execute(self, ctx: PipelineContext) -> None:
        if not ctx.evidence:
            return

        query_terms = set(ctx.spec.tokenize(ctx.query_text))
        if not query_terms:
            return

        for item in ctx.evidence:
            title = item.chunk.title
            if not title:
                continue
            title_terms = set(ctx.spec.tokenize(title))
            overlap = query_terms & title_terms
            if overlap:
                # Boost proportional to title term overlap
                fraction = len(overlap) / len(query_terms)
                boost = 1.0 + 0.5 * fraction  # up to 1.5x
                item.score *= boost
                item.signals["title_boost"] = boost
                item.signals["title_matched_terms"] = sorted(overlap)

        # Re-sort after boosting
        ctx.evidence.sort(key=lambda e: e.score, reverse=True)
        for i, item in enumerate(ctx.evidence):
            item.rank = i
