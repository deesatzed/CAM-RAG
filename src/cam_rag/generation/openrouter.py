"""OpenRouter generation backend using the chat/completions API.

Uses the OpenAI-compatible ``/chat/completions`` endpoint on OpenRouter.
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

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "anthropic/claude-sonnet-4"
_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(slots=True)
class OpenRouterGenerationBackend:
    """Generation backend that calls OpenRouter's ``/chat/completions`` endpoint.

    Satisfies the ``GenerationBackend`` protocol: ``generate(system, user) -> str``.
    """

    model: str = _DEFAULT_MODEL
    api_key: str = ""
    base_url: str = _DEFAULT_BASE_URL
    timeout: int = 120
    temperature: float = 0.1
    max_tokens: int = 2048

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY must be set in the environment "
                "or passed explicitly"
            )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response given system and user prompts.

        Returns the generated text, or an empty string on failure.
        """
        return self._call_api(system_prompt, user_prompt)

    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        """Make a single API call to OpenRouter chat completions endpoint."""
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
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
            logger.error("OpenRouter chat API call failed: %s", exc)
            return ""
        except json.JSONDecodeError as exc:
            logger.error("OpenRouter returned invalid JSON: %s", exc)
            return ""

        return _parse_chat_response(body)


def _parse_chat_response(body: dict) -> str:
    """Parse the OpenRouter/OpenAI-compatible chat completions response.

    Expected shape: ``{"choices": [{"message": {"content": "..."}}]}``
    """
    choices = body.get("choices", [])
    if not isinstance(choices, list) or not choices:
        logger.warning("Unexpected chat completions response shape")
        return ""
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        logger.warning("First choice is not a dict")
        return ""
    message = first_choice.get("message", {})
    if not isinstance(message, dict):
        logger.warning("Choice message is not a dict")
        return ""
    content = message.get("content", "")
    if not isinstance(content, str):
        logger.warning("Choice content is not a string")
        return ""
    return content.strip()
