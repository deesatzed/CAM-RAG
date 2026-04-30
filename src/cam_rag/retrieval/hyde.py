"""HyDE (Hypothetical Document Embeddings) pipeline step.

Instead of embedding the raw query for dense retrieval, HyDE uses an LLM
to generate a hypothetical answer/document and then embeds *that*.  The
BM25 sparse path still uses the original query, so the two views are
complementary.

The step runs AFTER ``sparse_bm25`` (which needs the original query)
but BEFORE ``dense_vector`` (which will embed whatever is in
``ctx.expanded_query``).
"""

from __future__ import annotations

import logging

from cam_rag.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)

_HYDE_SYSTEM_PROMPT = (
    "You are a scientific writing assistant. Write a short, factual "
    "scientific paragraph that directly answers the following question. "
    "Do not include citations or references. Keep it under 150 words."
)


class HyDEStep:
    """Generate a hypothetical document and set it as the dense query.

    Requires ``ctx.spec.generation_backend`` to be configured.
    No-op when ``ctx.spec.hyde_enabled`` is False or no generation
    backend is available.
    """

    name = "hyde_expansion"

    def execute(self, ctx: PipelineContext) -> None:
        if not ctx.spec.hyde_enabled:
            return

        backend = ctx.spec.generation_backend
        if backend is None:
            logger.debug("HyDE: no generation_backend configured, skipping")
            return

        # Preserve the original query for BM25 (already run by this point)
        ctx.extras["original_query"] = ctx.query_text

        user_prompt = ctx.query_text
        try:
            hypothesis = backend.generate(_HYDE_SYSTEM_PROMPT, user_prompt)
        except Exception:
            logger.warning("HyDE: generation failed, falling back to original query")
            return

        if not hypothesis or not hypothesis.strip():
            logger.debug("HyDE: generation returned empty, keeping original query")
            return

        ctx.extras["hyde_document"] = hypothesis
        ctx.expanded_query = hypothesis
        logger.debug(
            "HyDE: replaced query with hypothetical document (%d chars)",
            len(hypothesis),
        )
