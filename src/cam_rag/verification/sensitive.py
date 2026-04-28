"""Sensitive data detection and redaction for PHI/PII enforcement.

Two-phase detection:

Phase 1 (regex)
    Fast, deterministic pattern matching for structured PII/PHI formats
    (SSN, credit card, email, phone, MRN, DOB, prefixed patient names).
    Always runs when scanning is requested. Zero external dependencies.

Phase 2 (LLM NER)
    Contextual entity detection via a local healthcare-specific model
    (``pii-redaction-healthcare-qwen3-4b``). Catches names without
    prefixes, ages, addresses, geographic identifiers, and other HIPAA
    Safe Harbor categories that regex cannot reach. Opt-in via
    ``RAGPolicy.use_llm_redaction``.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phase 1 — Regex patterns
# ---------------------------------------------------------------------------

_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CREDIT_CARD_RE = re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"\b\d{3}-\d{3}-\d{4}\b")

PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "ssn": _SSN_RE,
    "credit_card": _CREDIT_CARD_RE,
    "email": _EMAIL_RE,
    "phone": _PHONE_RE,
}

_MRN_RE = re.compile(r"\bMRN[\s:]*\d{6,10}\b", re.IGNORECASE)
_DOB_RE = re.compile(
    r"\bDOB[\s:]*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", re.IGNORECASE,
)
_PATIENT_NAME_RE = re.compile(
    r"\bPatient[\s:]*[A-Z][a-z]+(?:\s[A-Z][a-z]+)+", re.UNICODE,
)

PHI_PATTERNS: dict[str, re.Pattern[str]] = {
    "mrn": _MRN_RE,
    "dob": _DOB_RE,
    "patient_name": _PATIENT_NAME_RE,
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SensitiveMatch:
    """A single detected sensitive-data span."""

    type: str          # "pii" or "phi"
    pattern_name: str  # key from PII_/PHI_PATTERNS or LLM entity type
    start: int
    end: int


# ---------------------------------------------------------------------------
# Phase 2 — LLM NER via ollama
# ---------------------------------------------------------------------------

_OLLAMA_BASE = "http://localhost:11434"
_OLLAMA_MODEL = "pii-redaction-healthcare"

_LLM_PROMPT_TEMPLATE = (
    "Identify all PII and PHI entities in this text and return ONLY a JSON "
    "array. Each element must have keys: type, value, start, end. "
    "Entity types: NAME, DATE, PHONE, EMAIL, SSN, MRN, ADDRESS, AGE, "
    "DEVICE_ID, URL, IP, ACCOUNT, LICENSE, VIN, BIOMETRIC, HEALTH_PLAN, "
    "OTHER_ID.\n\n{text}"
)

# Map LLM entity types to our type categories
_PHI_ENTITY_TYPES = frozenset({
    "NAME", "DATE", "MRN", "AGE", "HEALTH_PLAN",
    "BIOMETRIC", "DEVICE_ID", "VIN", "LICENSE",
})
_PII_ENTITY_TYPES = frozenset({
    "PHONE", "EMAIL", "SSN", "ADDRESS", "URL", "IP",
    "ACCOUNT", "OTHER_ID",
})


def _llm_scan(text: str) -> list[SensitiveMatch]:
    """Run Phase 2 LLM NER scan via ollama HTTP API.

    Returns detected entities as SensitiveMatch objects. Falls back to
    an empty list on any network or parsing error so that Phase 1 regex
    results still stand.
    """
    prompt = _LLM_PROMPT_TEMPLATE.format(text=text)
    payload = json.dumps({
        "model": _OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        f"{_OLLAMA_BASE}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        logger.warning("LLM PII scan unavailable: %s", exc)
        return []

    raw_response = body.get("response", "")
    return _parse_llm_entities(text, raw_response)


def _parse_llm_entities(
    source_text: str, raw_response: str,
) -> list[SensitiveMatch]:
    """Parse the LLM JSON response into SensitiveMatch objects.

    The LLM returns approximate character positions. We validate each
    entity by searching for its ``value`` in the source text near the
    reported position. If the value is found, we use the actual position;
    otherwise we fall back to a full-text search.
    """
    # Extract JSON array from response (may have leading/trailing text)
    start_idx = raw_response.find("[")
    end_idx = raw_response.rfind("]")
    if start_idx == -1 or end_idx == -1:
        logger.warning("LLM response did not contain a JSON array")
        return []

    try:
        entities = json.loads(raw_response[start_idx:end_idx + 1])
    except json.JSONDecodeError:
        logger.warning("LLM response contained invalid JSON")
        return []

    matches: list[SensitiveMatch] = []
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        value = str(entity.get("value", ""))
        entity_type = str(entity.get("type", "UNKNOWN")).upper()
        if not value:
            continue

        # Resolve actual position in source text
        actual_start = _find_entity_position(source_text, value, entity)
        if actual_start == -1:
            continue

        actual_end = actual_start + len(value)

        if entity_type in _PHI_ENTITY_TYPES:
            category = "phi"
        elif entity_type in _PII_ENTITY_TYPES:
            category = "pii"
        else:
            category = "phi"  # conservative: treat unknown as PHI

        matches.append(SensitiveMatch(
            type=category,
            pattern_name=f"llm_{entity_type.lower()}",
            start=actual_start,
            end=actual_end,
        ))

    return matches


def _find_entity_position(
    source_text: str, value: str, entity: dict,
) -> int:
    """Find the actual character offset of *value* in *source_text*.

    Checks near the LLM-reported position first (within +/- 20 chars),
    then falls back to a full-text search.
    """
    reported_start = int(entity.get("start", -1))

    # Try near the reported position first
    if reported_start >= 0:
        window_start = max(0, reported_start - 20)
        window_end = min(len(source_text), reported_start + len(value) + 20)
        window = source_text[window_start:window_end]
        local_idx = window.find(value)
        if local_idx != -1:
            return window_start + local_idx

    # Fall back to full-text search
    idx = source_text.find(value)
    return idx


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan_text(
    text: str,
    *,
    check_pii: bool = False,
    check_phi: bool = False,
    use_llm: bool = False,
) -> list[SensitiveMatch]:
    """Return all sensitive-data matches found in *text*.

    Parameters
    ----------
    text:
        The text to scan.
    check_pii:
        When ``True``, scan for PII patterns.
    check_phi:
        When ``True``, scan for PHI patterns.
    use_llm:
        When ``True``, also run Phase 2 LLM NER for contextual
        entity detection. Requires the ``pii-redaction-healthcare``
        model running in ollama.

    Returns
    -------
    list[SensitiveMatch]
        One entry per match, sorted by ``start`` position.
    """
    matches: list[SensitiveMatch] = []

    # Phase 1: regex
    if check_pii:
        for name, pattern in PII_PATTERNS.items():
            for m in pattern.finditer(text):
                matches.append(SensitiveMatch(
                    type="pii", pattern_name=name,
                    start=m.start(), end=m.end(),
                ))
    if check_phi:
        for name, pattern in PHI_PATTERNS.items():
            for m in pattern.finditer(text):
                matches.append(SensitiveMatch(
                    type="phi", pattern_name=name,
                    start=m.start(), end=m.end(),
                ))

    # Phase 2: LLM NER
    if use_llm and (check_pii or check_phi):
        llm_matches = _llm_scan(text)
        for lm in llm_matches:
            # Only include LLM matches for requested categories
            if lm.type == "pii" and not check_pii:
                continue
            if lm.type == "phi" and not check_phi:
                continue
            # Skip if regex already covered this span
            if not any(
                rm.start <= lm.start and rm.end >= lm.end
                for rm in matches
            ):
                matches.append(lm)

    matches.sort(key=lambda sm: sm.start)
    return matches


def redact_text(text: str, matches: list[SensitiveMatch]) -> str:
    """Replace every matched span in *text* with ``[REDACTED]``.

    Overlapping spans are handled by processing from rightmost to leftmost so
    that earlier indices remain valid.
    """
    if not matches:
        return text

    # Deduplicate and sort right-to-left so replacements don't shift indices.
    seen: set[tuple[int, int]] = set()
    unique: list[SensitiveMatch] = []
    for m in matches:
        key = (m.start, m.end)
        if key not in seen:
            seen.add(key)
            unique.append(m)
    unique.sort(key=lambda sm: sm.start, reverse=True)

    result = text
    for m in unique:
        result = result[: m.start] + "[REDACTED]" + result[m.end:]
    return result
