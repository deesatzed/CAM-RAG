"""Complexity-based routing for pipeline steps.

Assigns a routing signal ("deep", "standard", or "fast") to each evidence
item based on its chunk's complexity score and MoE combined score.  Downstream
steps can inspect ``evidence.signals["routing"]`` to adjust their behaviour
— for example, the cross-encoder reranker can skip "fast" items and the
multi-hop step can ignore them for term extraction.
"""

from __future__ import annotations

from cam_rag.pipeline.context import PipelineContext
from cam_rag.rag.models import Evidence

ROUTING_DEEP = "deep"
ROUTING_STANDARD = "standard"
ROUTING_FAST = "fast"


def assign_routing(evidence: list[Evidence]) -> None:
    """Tag each evidence item with a routing signal.

    Rules:
    - High: complexity_score > 0.66 OR moe_combined_score > 0.7 → "deep"
    - Low: complexity_score <= 0.33 AND moe_combined_score <= 0.3 → "fast"
    - Otherwise: "standard"

    When metadata keys are missing the item defaults to "standard".
    """
    for item in evidence:
        meta = item.chunk.metadata
        complexity_score: float = meta.get("complexity_score", 0.5)
        moe_combined: float = meta.get("moe_combined_score", 0.5)

        if complexity_score > 0.66 or moe_combined > 0.7:
            item.signals["routing"] = ROUTING_DEEP
        elif complexity_score <= 0.33 and moe_combined <= 0.3:
            item.signals["routing"] = ROUTING_FAST
        else:
            item.signals["routing"] = ROUTING_STANDARD


class ComplexityRoutingStep:
    """Pipeline step that annotates evidence with routing signals."""

    name = "complexity_routing"

    def execute(self, ctx: PipelineContext) -> None:
        if not ctx.spec.complexity_routing_enabled:
            return
        if not ctx.evidence:
            return
        assign_routing(ctx.evidence)
