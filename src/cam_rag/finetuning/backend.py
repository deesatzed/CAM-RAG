"""Local sentence-transformers embedding backend.

Loads a model from disk (fine-tuned) or HuggingFace hub and provides
the same ``EmbeddingBackend`` protocol used by the rest of cam_rag.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class LocalSentenceTransformerBackend:
    """Embedding backend backed by a local sentence-transformers model.

    Satisfies the ``EmbeddingBackend`` protocol (``dim: int``, ``embed(text) -> list[float]``).
    Uses a plain class rather than a dataclass because
    ``SentenceTransformer`` objects cannot be stored in ``__slots__``.

    Parameters
    ----------
    model_name_or_path:
        HuggingFace model identifier *or* local directory path.
    device:
        PyTorch device string. When ``None``, auto-detects MPS → CPU.
    max_seq_length:
        Optional override for the model's maximum input length.
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str | None = None,
        max_seq_length: int | None = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self._device = device or _auto_device()
        self._model = _load_model(model_name_or_path, self._device, max_seq_length)
        # get_embedding_dimension() is the new name (>=5.x); fall back for older versions
        if hasattr(self._model, "get_embedding_dimension"):
            self.dim: int = self._model.get_embedding_dimension()
        else:
            self.dim: int = self._model.get_sentence_embedding_dimension()  # type: ignore[assignment]
        logger.info(
            "loaded sentence-transformer %s (dim=%d, device=%s)",
            model_name_or_path,
            self.dim,
            self._device,
        )

    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        vector = self._model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return vector.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts efficiently."""
        vectors = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return vectors.tolist()


def _auto_device() -> str:
    """Pick the best available PyTorch device."""
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except (ImportError, AttributeError):
        pass
    return "cpu"


def _load_model(
    model_name_or_path: str,
    device: str,
    max_seq_length: int | None,
) -> Any:
    """Import and instantiate a SentenceTransformer."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for LocalSentenceTransformerBackend. "
            'Install with: pip install -e ".[finetune]"'
        ) from exc

    model = SentenceTransformer(model_name_or_path, device=device)
    if max_seq_length is not None:
        model.max_seq_length = max_seq_length
    return model
