"""Tests for TrainingDataGenerator using real Chunk objects and real BM25."""

from __future__ import annotations

import pytest

from cam_rag.finetuning.data import TrainingDataGenerator
from cam_rag.rag.models import Chunk


# ---------------------------------------------------------------------------
# Fixtures: realistic multi-document corpus
# ---------------------------------------------------------------------------

CARDIOLOGY_CHUNKS = [
    Chunk(
        id="card1",
        document_id="doc_cardiology",
        text=(
            "Acute ST-elevation myocardial infarction requires emergent percutaneous "
            "coronary intervention within 90 minutes of first medical contact."
        ),
        section_heading="STEMI Management",
        position=0,
    ),
    Chunk(
        id="card2",
        document_id="doc_cardiology",
        text=(
            "Dual antiplatelet therapy with aspirin and a P2Y12 inhibitor such as "
            "ticagrelor or prasugrel is standard of care post-PCI for at least 12 months."
        ),
        section_heading="Antiplatelet Therapy",
        position=1,
    ),
    Chunk(
        id="card3",
        document_id="doc_cardiology",
        text=(
            "Heart failure with reduced ejection fraction is managed with ACE inhibitors, "
            "beta-blockers, mineralocorticoid receptor antagonists, and SGLT2 inhibitors."
        ),
        section_heading="Heart Failure Treatment",
        position=2,
    ),
]

NEPHROLOGY_CHUNKS = [
    Chunk(
        id="neph1",
        document_id="doc_nephrology",
        text=(
            "Chronic kidney disease is staged by estimated glomerular filtration rate. "
            "Stage 3a corresponds to eGFR 45-59 mL/min/1.73m2."
        ),
        section_heading="CKD Staging",
        position=0,
    ),
    Chunk(
        id="neph2",
        document_id="doc_nephrology",
        text=(
            "Renal replacement therapy options include hemodialysis, peritoneal dialysis, "
            "and kidney transplantation. The choice depends on patient factors."
        ),
        section_heading="Renal Replacement Therapy",
        position=1,
    ),
]

ALL_CHUNKS = CARDIOLOGY_CHUNKS + NEPHROLOGY_CHUNKS


@pytest.fixture
def generator() -> TrainingDataGenerator:
    return TrainingDataGenerator(ALL_CHUNKS, seed=42)


# ---------------------------------------------------------------------------
# generate() tests
# ---------------------------------------------------------------------------


def test_generate_returns_triplets(generator: TrainingDataGenerator) -> None:
    triplets = generator.generate()
    assert len(triplets) > 0
    for t in triplets:
        assert "anchor" in t
        assert "positive" in t
        assert "negative" in t


def test_triplets_have_nonempty_fields(generator: TrainingDataGenerator) -> None:
    for t in generator.generate():
        assert len(t["anchor"].strip()) > 0
        assert len(t["positive"].strip()) > 0
        assert len(t["negative"].strip()) > 0


def test_title_body_pairs_use_section_heading(generator: TrainingDataGenerator) -> None:
    """At least some anchors should be section headings."""
    triplets = generator.generate()
    headings = {c.section_heading for c in ALL_CHUNKS if c.section_heading}
    anchors = {t["anchor"] for t in triplets}
    assert headings & anchors, "Expected some section headings as anchors"


def test_adjacent_pairs_generated(generator: TrainingDataGenerator) -> None:
    """Adjacent chunks from the same doc should produce positives."""
    triplets = generator.generate()
    # card1 and card2 are adjacent in doc_cardiology; card1.text should pair with card2.text
    cardiology_texts = {c.text for c in CARDIOLOGY_CHUNKS}
    pair_found = False
    for t in triplets:
        if t["anchor"] in cardiology_texts and t["positive"] in cardiology_texts:
            pair_found = True
            break
    assert pair_found, "Expected adjacent pair from cardiology document"


def test_negatives_from_different_document(generator: TrainingDataGenerator) -> None:
    """Hard negatives should come from a different document."""
    # Build text-to-doc mapping
    text_to_doc: dict[str, str] = {}
    for c in ALL_CHUNKS:
        text_to_doc[c.text] = c.document_id
        if c.section_heading:
            text_to_doc[c.section_heading] = c.document_id

    triplets = generator.generate()
    for t in triplets:
        anchor_doc = text_to_doc.get(t["anchor"])
        neg_doc = text_to_doc.get(t["negative"])
        if anchor_doc and neg_doc:
            assert anchor_doc != neg_doc, (
                f"Negative should be from a different doc: anchor={anchor_doc}, neg={neg_doc}"
            )


def test_deterministic_with_same_seed() -> None:
    g1 = TrainingDataGenerator(ALL_CHUNKS, seed=123)
    g2 = TrainingDataGenerator(ALL_CHUNKS, seed=123)
    assert g1.generate() == g2.generate()


def test_different_seed_may_differ() -> None:
    g1 = TrainingDataGenerator(ALL_CHUNKS, seed=1)
    g2 = TrainingDataGenerator(ALL_CHUNKS, seed=999)
    # Order should differ (shuffled)
    t1 = g1.generate()
    t2 = g2.generate()
    assert len(t1) == len(t2)
    # Content is the same but order may differ
    assert set(t["anchor"] for t in t1) == set(t["anchor"] for t in t2)


# ---------------------------------------------------------------------------
# generate_dataset() tests
# ---------------------------------------------------------------------------


def test_generate_dataset_returns_hf_dataset(generator: TrainingDataGenerator) -> None:
    ds = generator.generate_dataset()
    assert "anchor" in ds.column_names
    assert "positive" in ds.column_names
    assert "negative" in ds.column_names
    assert len(ds) > 0


def test_dataset_columns_match_triplets(generator: TrainingDataGenerator) -> None:
    triplets = generator.generate()
    ds = generator.generate_dataset()
    # Dataset is shuffled with same seed, so lengths should match
    assert len(ds) == len(triplets)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_single_document_no_negatives() -> None:
    """With chunks from only one document, hard negatives are unavailable."""
    single_doc = [
        Chunk(id="s1", document_id="only_doc", text="Some text here about topic A.", position=0),
        Chunk(id="s2", document_id="only_doc", text="More text here about topic B.", position=1),
    ]
    gen = TrainingDataGenerator(single_doc, seed=42)
    triplets = gen.generate()
    # Adjacent pairs exist but no cross-doc negatives → no triplets
    assert len(triplets) == 0


def test_empty_corpus() -> None:
    gen = TrainingDataGenerator([], seed=42)
    triplets = gen.generate()
    assert triplets == []
    ds = gen.generate_dataset()
    assert len(ds) == 0
