"""Reranking prompt construction and score parsing.

Uses a structured LLM prompt to score relevance of each evidence passage
to the user query.  The LLM returns JSON scores that are parsed and
normalised to [0, 1].
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a relevance scoring engine. For each numbered passage, rate how \
relevant it is to the user's query on a scale from 0 to 10, where:
- 0 = completely irrelevant
- 5 = somewhat relevant
- 10 = perfectly answers the query

Return ONLY a JSON object mapping passage numbers to scores.
Example: {"1": 8, "2": 3, "3": 9}

Do not include any other text, explanation, or formatting.\
"""


def build_rerank_prompt(
    query: str,
    passages: list[str],
) -> tuple[str, str]:
    """Build a system + user prompt for LLM-based reranking.

    Returns ``(system_prompt, user_prompt)``.
    """
    passage_block = "\n\n".join(
        f"[{i + 1}] {text[:1000]}"
        for i, text in enumerate(passages)
    )

    user_prompt = f"""\
Query: {query}

Passages:
{passage_block}

Rate each passage's relevance to the query (0-10). Return only JSON."""

    return _SYSTEM_PROMPT, user_prompt


def parse_rerank_scores(
    response: str,
    expected_count: int,
) -> list[float]:
    """Parse LLM reranking response into normalised [0, 1] scores.

    Falls back to equal scores (0.5) on parse failure.
    """
    # Try to extract JSON from the response
    response = response.strip()

    # Handle markdown code blocks
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if json_match:
        response = json_match.group(1)

    # Try direct JSON parse
    try:
        scores_dict = json.loads(response)
    except json.JSONDecodeError:
        # Try to find a JSON object anywhere in the response
        obj_match = re.search(r"\{[^{}]*\}", response)
        if obj_match:
            try:
                scores_dict = json.loads(obj_match.group(0))
            except json.JSONDecodeError:
                logger.warning("Could not parse reranker scores from response")
                return [0.5] * expected_count
        else:
            logger.warning("No JSON object found in reranker response")
            return [0.5] * expected_count

    if not isinstance(scores_dict, dict):
        logger.warning("Reranker response is not a dict")
        return [0.5] * expected_count

    # Extract scores by passage number (1-indexed keys)
    scores: list[float] = []
    for i in range(1, expected_count + 1):
        raw = scores_dict.get(str(i), scores_dict.get(i, 5))
        try:
            score = float(raw)
            # Normalise from 0-10 to 0-1
            scores.append(max(0.0, min(1.0, score / 10.0)))
        except (TypeError, ValueError):
            scores.append(0.5)

    return scores
