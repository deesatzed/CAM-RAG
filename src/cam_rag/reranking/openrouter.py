"""OpenRouter reranking backend using the chat/completions API.

Uses an LLM prompt to score relevance of evidence passages to a query.
The model evaluates (query, passage) pairs and returns relevance scores.

Requires ``OPENROUTER_API_KEY`` environment variable.

The API is called via stdlib ``urllib`` -- no external HTTP library needed.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

from cam_rag.reranking.prompt import build_rerank_prompt, parse_rerank_scores

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "anthropic/claude-haiku-4-5-20251001"
_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(slots=True)
class OpenRouterRerankerBackend:
    """Reranking backend that calls OpenRouter's ``/chat/completions`` endpoint.

    Satisfies the ``RerankerBackend`` protocol:
    ``rerank(query, passages) -> list[float]``.
    """

    model: str = _DEFAULT_MODEL
    api_key: str = ""
    base_url: str = _DEFAULT_BASE_URL
    timeout: int = 120
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY must be set in the environment "
                "or passed explicitly"
            )

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

    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        """Make a single API call to OpenRouter chat completions endpoint."""
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": 512,
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
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
            logger.error("OpenRouter reranker API call failed: %s", exc)
            return ""
        except json.JSONDecodeError as exc:
            logger.error("OpenRouter returned invalid JSON: %s", exc)
            return ""

        choices = body.get("choices", [])
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message", {})
                if isinstance(message, dict):
                    content = message.get("content", "")
                    return content.strip() if isinstance(content, str) else ""
        return ""
