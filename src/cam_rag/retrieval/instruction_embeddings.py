"""Instruction-aware embedding backend wrapper.

Many embedding models (Qwen3-Embedding, E5-Instruct, etc.) support asymmetric
query-document encoding via an instruction prefix on the query side.  This
wrapper decorates any ``EmbeddingBackend`` to prepend task-specific
instructions to queries while leaving document embeddings untouched.
"""

from __future__ import annotations


class InstructionEmbeddingBackend:
    """Wrap an embedding backend with query-side instruction prefixing.

    Documents are embedded unchanged via ``embed()`` / ``embed_batch()``.
    Queries use ``embed_query()`` / ``embed_query_batch()`` which prepend
    ``Instruct: {instruction}\\nQuery: {text}``.

    Parameters
    ----------
    backend : object
        Any object satisfying the ``EmbeddingBackend`` protocol
        (``dim: int`` + ``embed(text: str) -> list[float]``).
    instruction : str
        Task-specific instruction prepended to queries.
    """

    def __init__(self, backend: object, *, instruction: str) -> None:
        self._backend = backend
        self._instruction = instruction

    @property
    def dim(self) -> int:
        return self._backend.dim  # type: ignore[union-attr]

    # --- Document embedding (pass-through) ---

    def embed(self, text: str) -> list[float]:
        """Embed a document — no instruction prefix."""
        return self._backend.embed(text)  # type: ignore[union-attr]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed documents in batch — no instruction prefix."""
        if hasattr(self._backend, "embed_batch"):
            return self._backend.embed_batch(texts)  # type: ignore[union-attr]
        return [self.embed(t) for t in texts]

    # --- Query embedding (with instruction prefix) ---

    def embed_query(self, text: str) -> list[float]:
        """Embed a query with the instruction prefix."""
        prefixed = f"Instruct: {self._instruction}\nQuery: {text}"
        return self._backend.embed(prefixed)  # type: ignore[union-attr]

    def embed_query_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed queries in batch with the instruction prefix."""
        prefixed = [f"Instruct: {self._instruction}\nQuery: {t}" for t in texts]
        if hasattr(self._backend, "embed_batch"):
            return self._backend.embed_batch(prefixed)  # type: ignore[union-attr]
        return [self._backend.embed(p) for p in prefixed]  # type: ignore[union-attr]
