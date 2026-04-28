"""Tests for contract-aware generation prompt building.

Validates that the build_rag_prompt function correctly applies accuracy
contract labels to evidence items and extends the system prompt with
contract-specific instructions when applicable.

All data is real -- no mock, no simulation, no placeholders.
"""

from __future__ import annotations

import pytest

from cam_rag.generation.prompt import (
    _CONTRACT_INSTRUCTIONS,
    _CONTRACT_LABELS,
    build_rag_prompt,
)
from cam_rag.rag.models import Chunk, Evidence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_evidence(
    text: str,
    contract: str | None = None,
    *,
    title: str = "Test Source",
    source: str = "protocol.md",
    section_heading: str = "",
    doc_id: str = "doc1",
    chunk_id: str = "c1",
    score: float = 0.9,
) -> Evidence:
    """Create a single Evidence item with optional accuracy_contract metadata."""
    metadata: dict[str, object] = {}
    if contract is not None:
        metadata["accuracy_contract"] = contract
    return Evidence(
        chunk=Chunk(
            id=chunk_id,
            document_id=doc_id,
            text=text,
            title=title,
            source=source,
            section_heading=section_heading,
            metadata=metadata,
        ),
        score=score,
        retriever="hybrid_rrf",
    )


def _make_evidence_list(n: int = 3) -> list[Evidence]:
    """Create a list of evidence items without contracts (backward compat)."""
    texts = [
        "Sepsis requires blood cultures drawn within 45 minutes of presentation.",
        "Empiric antibiotics must be administered within 1 hour of sepsis recognition.",
        "Lactate levels should be measured and remeasured within 6 hours if elevated.",
    ]
    return [
        _make_evidence(
            text=texts[i % len(texts)],
            title=f"Sepsis Protocol Part {i + 1}",
            chunk_id=f"chunk-{i}",
            doc_id=f"doc-{i}",
            score=0.95 - (i * 0.05),
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# 1. Without contracts (contract_aware=False) -- output unchanged
# ---------------------------------------------------------------------------


class TestContractAwareDisabled:
    """When contract_aware=False, output is unchanged from baseline behavior."""

    def test_default_false_produces_standard_output(self):
        """contract_aware defaults to False, prompt structure is standard."""
        evidence = _make_evidence_list(2)
        system, user = build_rag_prompt("What is sepsis?", evidence)
        # Standard system prompt -- no contract instructions
        assert "[HARD CONSTRAINT]" not in system
        assert "[VERIFIED]" not in system
        assert "[SUGGESTION]" not in system
        assert _CONTRACT_INSTRUCTIONS not in system
        # Standard evidence numbering
        assert "[1] Source:" in user
        assert "[2] Source:" in user
        assert "What is sepsis?" in user

    def test_contract_metadata_ignored_when_disabled(self):
        """Even if chunks have contract metadata, labels are not added when disabled."""
        evidence = [
            _make_evidence("Amoxicillin dose is 500mg TID for adults.", contract="hard"),
            _make_evidence("Consider macrolide for atypical coverage.", contract="soft"),
        ]
        system, user = build_rag_prompt(
            "What antibiotic for pneumonia?",
            evidence,
            contract_aware=False,
        )
        assert "[HARD CONSTRAINT]" not in user
        assert "[SUGGESTION]" not in user
        assert _CONTRACT_INSTRUCTIONS not in system


# ---------------------------------------------------------------------------
# 2-4. Individual contract tiers
# ---------------------------------------------------------------------------


class TestIndividualContractLabels:
    """Each contract tier produces the correct label prefix."""

    def test_hard_contract_label(self):
        """Evidence with hard contract gets [HARD CONSTRAINT] prefix."""
        evidence = [
            _make_evidence(
                "Penicillin allergy contraindicates amoxicillin use.",
                contract="hard",
            )
        ]
        _, user = build_rag_prompt(
            "Can I give amoxicillin to penicillin-allergic patients?",
            evidence,
            contract_aware=True,
        )
        assert "[HARD CONSTRAINT] [1] Source:" in user

    def test_medium_contract_label(self):
        """Evidence with medium contract gets [VERIFIED] prefix."""
        evidence = [
            _make_evidence(
                "Vancomycin trough levels should target 15-20 mcg/mL for serious infections.",
                contract="medium",
            )
        ]
        _, user = build_rag_prompt(
            "What vancomycin levels should I target?",
            evidence,
            contract_aware=True,
        )
        assert "[VERIFIED] [1] Source:" in user

    def test_soft_contract_label(self):
        """Evidence with soft contract gets [SUGGESTION] prefix."""
        evidence = [
            _make_evidence(
                "Probiotics may reduce antibiotic-associated diarrhea incidence.",
                contract="soft",
            )
        ]
        _, user = build_rag_prompt(
            "Should I recommend probiotics?",
            evidence,
            contract_aware=True,
        )
        assert "[SUGGESTION] [1] Source:" in user


# ---------------------------------------------------------------------------
# 5. Mixed tiers
# ---------------------------------------------------------------------------


class TestMixedContractTiers:
    """Multiple contract tiers appear correctly in a single prompt."""

    def test_mixed_hard_medium_soft(self):
        """All three contract labels appear when evidence has mixed tiers."""
        evidence = [
            _make_evidence(
                "Beta-lactam antibiotics are first-line for community-acquired pneumonia.",
                contract="hard",
                chunk_id="c1",
                doc_id="d1",
                title="CAP Guidelines",
            ),
            _make_evidence(
                "Fluoroquinolones achieve reliable pulmonary tissue penetration.",
                contract="medium",
                chunk_id="c2",
                doc_id="d2",
                title="Pharmacokinetics Review",
            ),
            _make_evidence(
                "Zinc supplementation may shorten duration of respiratory infections.",
                contract="soft",
                chunk_id="c3",
                doc_id="d3",
                title="Complementary Therapies",
            ),
        ]
        system, user = build_rag_prompt(
            "How should I treat community-acquired pneumonia?",
            evidence,
            contract_aware=True,
        )
        assert "[HARD CONSTRAINT] [1] Source: CAP Guidelines" in user
        assert "[VERIFIED] [2] Source: Pharmacokinetics Review" in user
        assert "[SUGGESTION] [3] Source: Complementary Therapies" in user
        # System prompt extended with instructions
        assert "[HARD CONSTRAINT]" in system
        assert "[VERIFIED]" in system
        assert "[SUGGESTION]" in system


# ---------------------------------------------------------------------------
# 6-8. System prompt contract instructions behavior
# ---------------------------------------------------------------------------


class TestSystemPromptContractInstructions:
    """Contract instructions are appended to system prompt only when appropriate."""

    def test_system_prompt_includes_instructions_when_contracts_found(self):
        """System prompt has contract instructions when at least one contract present."""
        evidence = [
            _make_evidence("Ceftriaxone 2g IV daily for bacterial meningitis.", contract="hard"),
        ]
        system, _ = build_rag_prompt(
            "Meningitis treatment?",
            evidence,
            contract_aware=True,
        )
        assert "must be honored as authoritative" in system
        assert "reliable and should be preferred" in system
        assert "advisory only" in system

    def test_system_prompt_no_instructions_when_disabled(self):
        """System prompt has NO contract instructions when contract_aware=False."""
        evidence = [
            _make_evidence("Dexamethasone before antibiotics in meningitis.", contract="hard"),
        ]
        system, _ = build_rag_prompt(
            "Meningitis adjunct therapy?",
            evidence,
            contract_aware=False,
        )
        assert "must be honored as authoritative" not in system
        assert "advisory only" not in system

    def test_system_prompt_no_instructions_when_no_contracts_in_metadata(self):
        """System prompt has NO contract instructions when chunks lack contract metadata."""
        evidence = _make_evidence_list(3)
        system, _ = build_rag_prompt(
            "Sepsis management?",
            evidence,
            contract_aware=True,
        )
        # contract_aware is True but none of the evidence has contracts
        assert "must be honored as authoritative" not in system
        assert "advisory only" not in system


# ---------------------------------------------------------------------------
# 9-10. Graceful fallback for missing/unknown contract values
# ---------------------------------------------------------------------------


class TestContractFallback:
    """Graceful handling of missing or unknown accuracy_contract values."""

    def test_missing_metadata_key_no_label(self):
        """Evidence without accuracy_contract key gets no label prefix."""
        evidence = [
            _make_evidence("Standard dose amoxicillin is 500mg three times daily."),
        ]
        _, user = build_rag_prompt(
            "Amoxicillin dosing?",
            evidence,
            contract_aware=True,
        )
        # No label prefix -- line starts with [1]
        lines = user.split("\n")
        evidence_line = next(l for l in lines if "[1] Source:" in l)
        assert evidence_line.startswith("[1] Source:")

    def test_unknown_contract_value_no_label(self):
        """Evidence with unrecognized contract value gets no label prefix."""
        evidence = [
            _make_evidence(
                "Gentamicin requires therapeutic drug monitoring.",
                contract="ultra_high",
            ),
        ]
        system, user = build_rag_prompt(
            "Gentamicin monitoring?",
            evidence,
            contract_aware=True,
        )
        lines = user.split("\n")
        evidence_line = next(l for l in lines if "[1] Source:" in l)
        assert evidence_line.startswith("[1] Source:")
        # No contract instructions added for unrecognized values
        assert "must be honored as authoritative" not in system


# ---------------------------------------------------------------------------
# 11. Truncation still works with contract labels
# ---------------------------------------------------------------------------


class TestTruncationWithContracts:
    """Evidence truncation logic is unaffected by contract label prefixes."""

    def test_long_evidence_truncated_with_contract(self):
        """Long evidence text is still truncated when contract labels are present."""
        long_text = "Antibiotic resistance mechanisms include enzymatic degradation " * 500
        evidence = [
            _make_evidence(long_text, contract="hard"),
        ]
        _, user = build_rag_prompt(
            "Antibiotic resistance?",
            evidence,
            contract_aware=True,
            max_evidence_chars=200,
        )
        assert "[HARD CONSTRAINT] [1] Source:" in user
        assert "..." in user
        # Total user prompt should be well under the full long_text length
        assert len(user) < len(long_text)


# ---------------------------------------------------------------------------
# 12. contract_aware=True but no chunks have contracts
# ---------------------------------------------------------------------------


class TestContractAwareNoContracts:
    """contract_aware=True with no actual contracts in evidence."""

    def test_no_instructions_added_when_no_contracts_present(self):
        """No contract instructions appended when evidence lacks contract metadata."""
        evidence = [
            _make_evidence("Blood cultures should be drawn before antibiotic administration."),
            _make_evidence(
                "Broad-spectrum coverage is recommended for undifferentiated sepsis.",
                chunk_id="c2",
                doc_id="d2",
            ),
        ]
        system, user = build_rag_prompt(
            "Sepsis initial management?",
            evidence,
            contract_aware=True,
        )
        # No labels in user prompt
        assert "[HARD CONSTRAINT]" not in user
        assert "[VERIFIED]" not in user
        assert "[SUGGESTION]" not in user
        # No instructions in system prompt
        assert _CONTRACT_INSTRUCTIONS not in system
        assert "must be honored" not in system


# ---------------------------------------------------------------------------
# 13. Evidence ordering preserved
# ---------------------------------------------------------------------------


class TestEvidenceOrderingPreserved:
    """Evidence order is preserved with contract labels."""

    def test_ordering_matches_input_sequence(self):
        """Evidence items appear in the same order as the input list."""
        evidence = [
            _make_evidence(
                "First-line therapy for MRSA bacteremia is vancomycin.",
                contract="hard",
                chunk_id="c1",
                doc_id="d1",
                title="MRSA Guidelines",
            ),
            _make_evidence(
                "Daptomycin is an alternative for vancomycin-resistant organisms.",
                contract="medium",
                chunk_id="c2",
                doc_id="d2",
                title="Alternative Agents",
            ),
            _make_evidence(
                "Linezolid has oral bioavailability equivalent to IV formulation.",
                contract="soft",
                chunk_id="c3",
                doc_id="d3",
                title="Oral Step-down",
            ),
        ]
        _, user = build_rag_prompt(
            "MRSA treatment options?",
            evidence,
            contract_aware=True,
        )
        # Verify ordering via position of labels in output
        pos_hard = user.index("[HARD CONSTRAINT] [1]")
        pos_verified = user.index("[VERIFIED] [2]")
        pos_suggestion = user.index("[SUGGESTION] [3]")
        assert pos_hard < pos_verified < pos_suggestion


# ---------------------------------------------------------------------------
# 14. Real clinical text with hard constraint
# ---------------------------------------------------------------------------


class TestClinicalHardConstraint:
    """Real clinical scenario with hard constraint applied."""

    def test_clinical_contraindication_as_hard_constraint(self):
        """A clinical contraindication marked as hard constraint appears correctly."""
        evidence = [
            _make_evidence(
                text=(
                    "Metformin is absolutely contraindicated in patients with "
                    "estimated GFR below 30 mL/min/1.73m2 due to the risk of "
                    "lactic acidosis. This applies to all formulations including "
                    "extended-release preparations."
                ),
                contract="hard",
                title="Renal Dosing Guidelines",
                section_heading="Contraindications",
            ),
            _make_evidence(
                text=(
                    "For patients with GFR 30-45 mL/min/1.73m2, metformin dose "
                    "should be reduced to a maximum of 1000mg daily with close "
                    "monitoring of renal function every 3 months."
                ),
                contract="medium",
                chunk_id="c2",
                doc_id="d2",
                title="Renal Dosing Guidelines",
                section_heading="Dose Adjustments",
            ),
        ]
        system, user = build_rag_prompt(
            "Can I prescribe metformin for a patient with GFR 25?",
            evidence,
            contract_aware=True,
        )
        # Hard constraint applied to contraindication
        assert "[HARD CONSTRAINT] [1] Source: Renal Dosing Guidelines > Contraindications" in user
        # Verified applied to dose adjustment guidance
        assert "[VERIFIED] [2] Source: Renal Dosing Guidelines > Dose Adjustments" in user
        # System prompt extended
        assert "must be honored as authoritative" in system
        # Clinical content preserved
        assert "lactic acidosis" in user
        assert "GFR 30-45" in user


# ---------------------------------------------------------------------------
# 15. _CONTRACT_LABELS dict structure
# ---------------------------------------------------------------------------


class TestContractLabelsDict:
    """Validate the _CONTRACT_LABELS constant structure."""

    def test_expected_keys_present(self):
        """_CONTRACT_LABELS has exactly hard, medium, soft keys."""
        assert set(_CONTRACT_LABELS.keys()) == {"hard", "medium", "soft"}

    def test_label_values_are_bracketed(self):
        """All label values are wrapped in square brackets."""
        for key, label in _CONTRACT_LABELS.items():
            assert label.startswith("["), f"Label for {key!r} missing opening bracket"
            assert label.endswith("]"), f"Label for {key!r} missing closing bracket"
