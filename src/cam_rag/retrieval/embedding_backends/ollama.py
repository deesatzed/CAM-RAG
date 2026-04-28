"""Ollama embedding backend using the local /api/embed endpoint.

Supports any Ollama-hosted embedding model including:
- ``nomic-embed-text-v2-moe`` (768 dims, 512 token context)
- ``bge-m3`` (1024 dims, 8192 token context)
- ``mxbai-embed-large`` (1024 dims)

Requires a running Ollama instance (default: ``http://localhost:11434``).
No API key required for local inference.

The API is called via stdlib ``urllib`` -- no external HTTP library needed.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "bge-m3"
_DEFAULT_DIM = 1024
_DEFAULT_BASE_URL = "http://localhost:11434"
_MAX_BATCH_SIZE = 64


@dataclass(slots=True)
class OllamaEmbeddingBackend:
    """Embedding backend that calls a local Ollama ``/api/embed`` endpoint.

    Implements the ``EmbeddingBackend`` protocol (``dim`` + ``embed``).
    Also provides ``embed_batch`` for efficient multi-document embedding
    and ``embed_for_indexing`` for document-time prefix handling.

    Task-prefix support
    -------------------
    Some models (e.g. ``nomic-embed-text-v2-moe``) benefit from prefixes
    like ``search_query: `` and ``search_document: ``.  Set
    ``prefix_query`` / ``prefix_document`` to enable this.  ``embed()``
    prepends ``prefix_query`` (the common query-time path) and
    ``embed_for_indexing()`` prepends ``prefix_document``.

    .. note::

       The ``EmbeddingBackend`` protocol defines only ``embed(text)``.
       ``DenseVectorRetriever`` calls ``embed()`` for both documents and
       queries.  For models that require separate prefixes, call
       ``embed_for_indexing()`` at index time or update the retriever to
       do so.  The performance loss from using query-prefix at index time
       is modest (a few percent per the Nomic paper).
    """

    model: str = _DEFAULT_MODEL
    dim: int = _DEFAULT_DIM
    base_url: str = _DEFAULT_BASE_URL
    timeout: int = 60
    prefix_query: str = ""
    prefix_document: str = ""
    _cache: dict[str, list[float]] = field(
        default_factory=dict, repr=False,
    )

    def embed(self, text: str) -> list[float]:
        """Embed a single text string with query-time prefix.

        Results are cached per *raw text* (prefix included).
        """
        prefixed = f"{self.prefix_query}{text}" if self.prefix_query else text
        if prefixed in self._cache:
            return self._cache[prefixed]
        vectors = self._call_api([prefixed])
        if vectors:
            self._cache[prefixed] = vectors[0]
            return vectors[0]
        logger.warning("Ollama embed returned empty; using zero vector")
        return [0.0] * self.dim

    def embed_for_indexing(self, text: str) -> list[float]:
        """Embed a single text string with document-time prefix.

        Use this at index time for models that distinguish query vs.
        document embeddings (e.g. ``nomic-embed-text-v2-moe``).
        """
        prefixed = (
            f"{self.prefix_document}{text}" if self.prefix_document else text
        )
        if prefixed in self._cache:
            return self._cache[prefixed]
        vectors = self._call_api([prefixed])
        if vectors:
            self._cache[prefixed] = vectors[0]
            return vectors[0]
        logger.warning(
            "Ollama embed_for_indexing returned empty; using zero vector",
        )
        return [0.0] * self.dim

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in batched API calls.

        Uses ``prefix_query`` for all texts.  For index-time batches,
        callers should prepend ``prefix_document`` before calling.
        """
        results: list[list[float]] = []
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            prefixed = (
                f"{self.prefix_query}{text}" if self.prefix_query else text
            )
            if prefixed in self._cache:
                results.append(self._cache[prefixed])
            else:
                results.append([])  # placeholder
                uncached_indices.append(i)
                uncached_texts.append(prefixed)

        # Call API in batches for uncached texts
        for batch_start in range(0, len(uncached_texts), _MAX_BATCH_SIZE):
            batch = uncached_texts[
                batch_start : batch_start + _MAX_BATCH_SIZE
            ]
            vectors = self._call_api(batch)
            for j, vec in enumerate(vectors):
                idx = uncached_indices[batch_start + j]
                results[idx] = vec
                self._cache[uncached_texts[batch_start + j]] = vec

        # Fill any remaining empty slots with zero vectors
        for i in range(len(results)):
            if not results[i]:
                results[i] = [0.0] * self.dim

        return results

    def is_available(self) -> bool:
        """Check if the Ollama server is reachable."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        """Make a single API call to the Ollama embed endpoint."""
        payload = json.dumps({
            "model": self.model,
            "input": texts if len(texts) > 1 else texts[0],
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(
                req, timeout=self.timeout,
            ) as resp:
                body = json.loads(resp.read())
        except (urllib.error.URLError, OSError) as exc:
            logger.error("Ollama embedding API call failed: %s", exc)
            return [[0.0] * self.dim for _ in texts]
        except json.JSONDecodeError as exc:
            logger.error("Ollama returned invalid JSON: %s", exc)
            return [[0.0] * self.dim for _ in texts]

        return _parse_ollama_response(body, len(texts), self.dim)

    def clear_cache(self) -> None:
        """Clear the in-memory embedding cache."""
        self._cache.clear()


def _parse_ollama_response(
    body: dict,
    expected_count: int,
    dim: int,
) -> list[list[float]]:
    """Parse the Ollama ``/api/embed`` response.

    Ollama returns ``{"model": "...", "embeddings": [[...], [...]]}``
    where ``embeddings`` is always a list of lists (one per input).
    """
    embeddings = body.get("embeddings", [])
    if not isinstance(embeddings, list):
        logger.warning("Unexpected Ollama embedding response shape")
        return [[0.0] * dim for _ in range(expected_count)]

    vectors: list[list[float]] = []
    for entry in embeddings:
        if isinstance(entry, list) and len(entry) > 0:
            vectors.append([float(v) for v in entry])
        else:
            vectors.append([0.0] * dim)

    # Pad if fewer vectors returned than expected
    while len(vectors) < expected_count:
        vectors.append([0.0] * dim)

    return vectors[:expected_count]
