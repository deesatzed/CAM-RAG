"""Ollama reranking backend using the local /api/chat endpoint.

Uses an LLM prompt to score relevance of evidence passages to a query.
The model evaluates (query, passage) pairs and returns relevance scores.

Requires a running Ollama instance (default: ``http://localhost:11434``).
No API key required for local inference.

The API is called via stdlib ``urllib`` -- no external HTTP library needed.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass

from cam_rag.reranking.prompt import build_rerank_prompt, parse_rerank_scores

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "qwen3:4b"
_DEFAULT_BASE_URL = "http://localhost:11434"


@dataclass(slots=True)
class OllamaRerankerBackend:
    """Reranking backend that calls a local Ollama ``/api/chat`` endpoint.

    Satisfies the ``RerankerBackend`` protocol:
    ``rerank(query, passages) -> list[float]``.
    """

    model: str = _DEFAULT_MODEL
    base_url: str = _DEFAULT_BASE_URL
    timeout: int = 120
    temperature: float = 0.0

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        """Score each passage's relevance to the query.

        Returns a list of scores in [0, 1], one per passage.
        Falls back to equal scores (0.5) on any failure.
        """
        if not passages:
            return []
        system_prompt, user_prompt = build_rerank_prompt(query, passages)
        response = self._call_api(system_prompt, user_prompt)
        if not response:
            return [0.5] * len(passages)
        return parse_rerank_scores(response, len(passages))

    def is_available(self) -> bool:
        """Check if the Ollama server is reachable."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        """Make a single API call to the Ollama chat endpoint."""
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": 512,
            },
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(
                req, timeout=self.timeout,
            ) as resp:
                body = json.loads(resp.read())
        except (urllib.error.URLError, OSError) as exc:
            logger.error("Ollama reranker API call failed: %s", exc)
            return ""
        except json.JSONDecodeError as exc:
            logger.error("Ollama returned invalid JSON: %s", exc)
            return ""

        message = body.get("message", {})
        if isinstance(message, dict):
            content = message.get("content", "")
            return content.strip() if isinstance(content, str) else ""
        return ""
