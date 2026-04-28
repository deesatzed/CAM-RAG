"""Ollama generation backend using the local /api/chat endpoint.

Supports any Ollama-hosted chat/completion model including:
- ``qwen3:4b`` (fast, local)
- ``llama3.2:3b`` (fast, local)
- ``mistral:7b`` (moderate, local)

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

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "qwen3:4b"
_DEFAULT_BASE_URL = "http://localhost:11434"


@dataclass(slots=True)
class OllamaGenerationBackend:
    """Generation backend that calls a local Ollama ``/api/chat`` endpoint.

    Satisfies the ``GenerationBackend`` protocol: ``generate(system, user) -> str``.
    """

    model: str = _DEFAULT_MODEL
    base_url: str = _DEFAULT_BASE_URL
    timeout: int = 120
    temperature: float = 0.1
    max_tokens: int = 2048

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response given system and user prompts.

        Returns the generated text, or an empty string on failure.
        """
        return self._call_api(system_prompt, user_prompt)

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
                "num_predict": self.max_tokens,
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
            logger.error("Ollama chat API call failed: %s", exc)
            return ""
        except json.JSONDecodeError as exc:
            logger.error("Ollama returned invalid JSON: %s", exc)
            return ""

        return _parse_chat_response(body)


def _parse_chat_response(body: dict) -> str:
    """Parse the Ollama ``/api/chat`` response.

    Ollama returns ``{"message": {"role": "assistant", "content": "..."}, ...}``
    """
    message = body.get("message", {})
    if not isinstance(message, dict):
        logger.warning("Unexpected Ollama chat response shape")
        return ""
    content = message.get("content", "")
    if not isinstance(content, str):
        logger.warning("Ollama chat response content is not a string")
        return ""
    return content.strip()
