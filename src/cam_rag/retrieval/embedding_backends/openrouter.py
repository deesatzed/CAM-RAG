"""OpenRouter embedding backend using the OpenAI-compatible embeddings API.

Default model: ``perplexity/pplx-embed-v1-4b`` (2560 dimensions, 32K context).
Requires ``OPENROUTER_API_KEY`` environment variable.

The API is called via stdlib ``urllib`` — no external HTTP library needed.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "perplexity/pplx-embed-v1-4b"
_DEFAULT_DIM = 2560
_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_MAX_BATCH_SIZE = 64


@dataclass(slots=True)
class OpenRouterEmbeddingBackend:
    """Embedding backend that calls OpenRouter's ``/embeddings`` endpoint.

    Implements the ``EmbeddingBackend`` protocol (``dim`` + ``embed``).
    Also provides ``embed_batch`` for efficient multi-document embedding.
    """

    model: str = _DEFAULT_MODEL
    dim: int = _DEFAULT_DIM
    api_key: str = ""
    base_url: str = _DEFAULT_BASE_URL
    timeout: int = 60
    _cache: dict[str, list[float]] = field(
        default_factory=dict, repr=False,
    )

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY must be set in the environment "
                "or passed explicitly"
            )

    def embed(self, text: str) -> list[float]:
        """Embed a single text string. Results are cached per-text."""
        if text in self._cache:
            return self._cache[text]
        vectors = self._call_api([text])
        if vectors:
            self._cache[text] = vectors[0]
            return vectors[0]
        logger.warning("OpenRouter embed returned empty; using zero vector")
        return [0.0] * self.dim

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in batched API calls."""
        results: list[list[float]] = []
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            if text in self._cache:
                results.append(self._cache[text])
            else:
                results.append([])  # placeholder
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Call API in batches for uncached texts
        for batch_start in range(0, len(uncached_texts), _MAX_BATCH_SIZE):
            batch = uncached_texts[
                batch_start:batch_start + _MAX_BATCH_SIZE
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

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        """Make a single API call to OpenRouter embeddings endpoint."""
        payload = json.dumps({
            "model": self.model,
            "input": texts if len(texts) > 1 else texts[0],
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        try:
            with urllib.request.urlopen(
                req, timeout=self.timeout,
            ) as resp:
                body = json.loads(resp.read())
        except (urllib.error.URLError, OSError) as exc:
            logger.error("OpenRouter embedding API call failed: %s", exc)
            return [[0.0] * self.dim for _ in texts]
        except json.JSONDecodeError as exc:
            logger.error("OpenRouter returned invalid JSON: %s", exc)
            return [[0.0] * self.dim for _ in texts]

        return _parse_embedding_response(body, len(texts), self.dim)

    def clear_cache(self) -> None:
        """Clear the in-memory embedding cache."""
        self._cache.clear()


def _parse_embedding_response(
    body: dict,
    expected_count: int,
    dim: int,
) -> list[list[float]]:
    """Parse the OpenRouter/OpenAI-compatible embedding response."""
    data = body.get("data", [])
    if not isinstance(data, list):
        logger.warning("Unexpected embedding response shape")
        return [[0.0] * dim for _ in range(expected_count)]

    # Sort by index to handle out-of-order responses
    sorted_data = sorted(data, key=lambda d: d.get("index", 0))

    vectors: list[list[float]] = []
    for entry in sorted_data:
        embedding = entry.get("embedding", [])
        if isinstance(embedding, list) and len(embedding) > 0:
            vectors.append([float(v) for v in embedding])
        else:
            vectors.append([0.0] * dim)

    # Pad if fewer vectors returned than expected
    while len(vectors) < expected_count:
        vectors.append([0.0] * dim)

    return vectors[:expected_count]
