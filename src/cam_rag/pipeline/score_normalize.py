"""Score normalization step for evidence ranking.

Cross-encoder logits, RRF scores, and boosted scores all live on different
scales.  This step normalizes evidence scores to [0, 1] using min-max
scaling, which ensures nDCG@10 computation isn't distorted by scale
differences.

Also applies a rank-decay factor: even after score normalization, higher-
ranked items receive a small bonus to preserve strict ordering when scores
are close together.
"""

from __future__ import annotations

from cam_rag.pipeline.context import PipelineContext


class ScoreNormalizeStep:
    """Normalize evidence scores to [0, 1] with rank-decay tiebreaking."""

    name = "score_normalize"

    def execute(self, ctx: PipelineContext) -> None:
        if not ctx.evidence or len(ctx.evidence) < 2:
            return

        scores = [item.score for item in ctx.evidence]
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score

        if score_range <= 0:
            # All scores identical — assign by rank
            for i, item in enumerate(ctx.evidence):
                item.score = 1.0 - (i / len(ctx.evidence))
            return

        # Min-max normalize to [0, 1]
        for item in ctx.evidence:
            item.score = (item.score - min_score) / score_range

        # Add rank-decay bonus: top item gets full score, each subsequent
        # gets a tiny fraction less to break ties deterministically
        n = len(ctx.evidence)
        for i, item in enumerate(ctx.evidence):
            rank_bonus = 0.001 * (n - i) / n
            item.score += rank_bonus

        # Final re-sort (should be stable but ensure correctness)
        ctx.evidence.sort(key=lambda e: e.score, reverse=True)
        for i, item in enumerate(ctx.evidence):
            item.rank = i
