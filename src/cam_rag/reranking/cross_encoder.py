"""Local cross-encoder reranker using sentence-transformers.

Dedicated cross-encoder models (e.g. ``cross-encoder/ms-marco-MiniLM-L-6-v2``)
outperform LLM-prompt-based reranking for passage re-scoring.  This backend
wraps the sentence-transformers ``CrossEncoder`` class and implements the
same ``rerank(query, passages) -> list[float]`` protocol as the existing
Ollama/OpenRouter reranker backends.
"""

from __future__ import annotations

import logging
import platform

logger = logging.getLogger(__name__)


def _auto_device() -> str:
    """Select the best available device for inference."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class LocalCrossEncoderBackend:
    """Cross-encoder reranker using a local sentence-transformers model.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model name or local path to a cross-encoder model.
    device : str or None
        Device for inference.  Auto-detects CUDA/MPS/CPU when ``None``.
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        from sentence_transformers import CrossEncoder

        resolved_device = device or _auto_device()
        self._model = CrossEncoder(
            model_name_or_path,
            device=resolved_device,
            trust_remote_code=trust_remote_code,
        )
        self._model_name = model_name_or_path
        logger.info(
            "Loaded cross-encoder %s on %s",
            model_name_or_path,
            resolved_device,
        )

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        """Score each passage against the query.

        Parameters
        ----------
        query : str
            The search query.
        passages : list[str]
            Passages to re-score.

        Returns
        -------
        list[float]
            One relevance score per passage, in the same order.
        """
        if not passages:
            return []
        pairs = [(query, p) for p in passages]
        scores = self._model.predict(pairs, show_progress_bar=False)
        return [float(s) for s in scores]
