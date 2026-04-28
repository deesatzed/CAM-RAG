"""RAG prompt construction for LLM generation.

Builds a grounded-generation prompt from user query + retrieved evidence.
The prompt format is model-agnostic (plain text) and instructs the LLM to:
1. Answer using ONLY the provided evidence
2. Cite sources by number
3. Say "I don't know" when evidence is insufficient
"""

from __future__ import annotations

from cam_rag.rag.models import Evidence

_SYSTEM_PROMPT = """\
You are a precise, helpful research assistant. Answer the user's question \
using ONLY the evidence provided below. Follow these rules strictly:

1. Base your answer exclusively on the provided evidence passages.
2. Cite sources using [1], [2], etc. corresponding to the evidence numbers.
3. If the evidence does not contain enough information to answer fully, \
say so explicitly rather than guessing.
4. Be concise but thorough. Prefer direct answers over lengthy preambles.
5. If evidence passages conflict, note the discrepancy.
"""

_CONTRACT_LABELS: dict[str, str] = {
    "hard": "[HARD CONSTRAINT]",
    "medium": "[VERIFIED]",
    "soft": "[SUGGESTION]",
}

_CONTRACT_INSTRUCTIONS = """
6. Evidence marked [HARD CONSTRAINT] must be honored as authoritative. Do not contradict it.
7. Evidence marked [VERIFIED] is reliable and should be preferred.
8. Evidence marked [SUGGESTION] is advisory only.
"""

_MAX_EVIDENCE_CHARS = 12_000


def build_rag_prompt(
    query: str,
    evidence: list[Evidence],
    *,
    max_evidence_items: int = 10,
    max_evidence_chars: int = _MAX_EVIDENCE_CHARS,
    contract_aware: bool = False,
) -> tuple[str, str]:
    """Build a system + user prompt pair for grounded RAG generation.

    Returns ``(system_prompt, user_prompt)`` suitable for chat-style APIs.

    When *contract_aware* is True, evidence items with an
    ``accuracy_contract`` key in their chunk metadata will be prefixed with
    the corresponding contract label (e.g. ``[HARD CONSTRAINT]``).  The
    system prompt is also extended with contract-specific instructions when
    at least one contract label is present.
    """
    # Build evidence block
    evidence_lines: list[str] = []
    total_chars = 0
    any_contract_found = False

    for i, item in enumerate(evidence[:max_evidence_items], 1):
        source = item.chunk.title or item.chunk.source or f"Document {item.chunk.document_id}"
        heading = f" > {item.chunk.section_heading}" if item.chunk.section_heading else ""
        text = item.chunk.text
        # Truncate individual evidence if needed
        remaining = max_evidence_chars - total_chars
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[:remaining] + "..."

        # Determine contract label prefix
        label_prefix = ""
        if contract_aware:
            contract_value = item.chunk.metadata.get("accuracy_contract", "")
            if contract_value in _CONTRACT_LABELS:
                label_prefix = f"{_CONTRACT_LABELS[contract_value]} "
                any_contract_found = True

        evidence_lines.append(
            f"{label_prefix}[{i}] Source: {source}{heading}\n{text}"
        )
        total_chars += len(text)

    evidence_block = "\n\n".join(evidence_lines) if evidence_lines else "(No evidence retrieved)"

    # Build system prompt — extend with contract instructions when applicable
    system_prompt = _SYSTEM_PROMPT.strip()
    if contract_aware and any_contract_found:
        system_prompt = system_prompt + _CONTRACT_INSTRUCTIONS

    user_prompt = f"""\
Evidence:
{evidence_block}

Question: {query}

Answer the question using the evidence above. Cite sources with [1], [2], etc."""

    return system_prompt, user_prompt
