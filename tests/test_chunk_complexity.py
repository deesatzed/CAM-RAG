"""Tests for chunk-level complexity scoring.

All test data uses real clinical / medical protocol text or clearly
constructed prose -- no mocks, no placeholders, no simulations.
"""

from __future__ import annotations

from cam_rag.documents.chunking import chunk_documents
from cam_rag.documents.complexity import (
    _composite_score,
    _label_from_score,
    score_chunk_complexity,
    score_chunks,
)
from cam_rag.rag.models import Chunk, CorpusDocument


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(text: str, chunk_id: str = "c0", doc_id: str = "d0") -> Chunk:
    return Chunk(id=chunk_id, document_id=doc_id, text=text, source="test.md")


def _make_doc(text: str, doc_id: str = "doc_test") -> CorpusDocument:
    return CorpusDocument(id=doc_id, text=text, source="test.md", title="Test")


# ---------------------------------------------------------------------------
# Real clinical text fixtures
# ---------------------------------------------------------------------------

SIMPLE_TEXT = "Hello world. This is simple. Nothing complex here."

MEDIUM_TEXT = (
    "The patient presented to the emergency department with a 3-day history "
    "of productive cough, fevers to 38.9C, and progressive dyspnea on exertion. "
    "Chest radiograph demonstrated a right lower lobe consolidation consistent "
    "with community-acquired pneumonia. Blood cultures were obtained and the "
    "patient was started on ceftriaxone 1g IV daily and azithromycin 500mg IV "
    "daily per the hospital antibiotic stewardship protocol. Supplemental "
    "oxygen was administered at 2L/min via nasal cannula to maintain SpO2 above "
    "92%. The respiratory therapy team was consulted for incentive spirometry "
    "and chest physiotherapy. Serial chest radiographs were ordered to monitor "
    "for clinical improvement over the subsequent 48-72 hours."
)

COMPLEX_CLINICAL_TEXT = (
    "The patient was intubated with a 7.5mm endotracheal tube following rapid "
    "sequence induction with propofol 200mg IV and succinylcholine 100mg IV. "
    "Initial ventilator settings: FiO2 1.0, tidal volume 6mL/kg IBW, PEEP "
    "10cmH2O, respiratory rate 16/min. Post-intubation ABG showed pH 7.22, "
    "PaCO2 58mmHg, PaO2 68mmHg, HCO3 18mEq/L, lactate 4.2mmol/L. GCS was "
    "3T (E1V1TM1). SpO2 improved from 78% to 96% after intubation. "
    "Norepinephrine infusion was initiated at 0.1mcg/kg/min and titrated to "
    "MAP >65mmHg. CT angiography revealed bilateral pulmonary emboli in the "
    "segmental and subsegmental branches [1]. Heparin drip was started per "
    "protocol with aPTT target 60-80 seconds (Smith et al., 2019). "
    "APACHE II score calculated at 28, predicting mortality of approximately "
    "55%. SOFA score was 12 on admission."
)

CITATION_HEAVY_TEXT = (
    "Recent meta-analyses have demonstrated the efficacy of early goal-directed "
    "therapy in sepsis management [1]. The ARISE trial (Peake et al., 2014) "
    "found no significant difference in 90-day mortality between EGDT and "
    "usual care [2]. Similarly, the ProCESS trial (Yealy et al., 2014) "
    "reported comparable outcomes across three resuscitation strategies [3]. "
    "The ProMISe trial (Mouncey et al., 2015) confirmed these findings in "
    "the UK context [4]. A subsequent patient-level meta-analysis by the "
    "PRISM investigators (Rowan et al., 2017) pooled data from all three "
    "trials and found no benefit for protocolized care [5]."
)

BULLET_LIST_TEXT = (
    "Sepsis Bundle Checklist:\n"
    "- Measure lactate level within 1 hour\n"
    "- Obtain blood cultures before antibiotics\n"
    "- Administer broad-spectrum antibiotics within 1 hour\n"
    "- Begin rapid administration of 30mL/kg crystalloid for hypotension\n"
    "- Apply vasopressors if hypotensive during or after fluid resuscitation\n"
    "- Reassess volume status and tissue perfusion within 6 hours\n"
    "- Re-measure lactate if initial lactate was elevated"
)

NUMERIC_HEAVY_TEXT = (
    "Lab results: WBC 18.4 x10^9/L, Hgb 9.2 g/dL, Plt 89 x10^9/L, "
    "INR 1.8, PTT 42s, fibrinogen 180 mg/dL. BMP: Na 131 mEq/L, "
    "K 5.8 mEq/L, Cl 98 mEq/L, CO2 14 mEq/L, BUN 48 mg/dL, "
    "Cr 2.4 mg/dL, glucose 224 mg/dL. Troponin-I 0.84 ng/mL "
    "(reference <0.04). Pro-BNP 12,400 pg/mL. Procalcitonin 18.6 ng/mL."
)


# ===================================================================
# Test: Simple text scores "low"
# ===================================================================

def test_simple_text_scores_low():
    """Plain short text with no technical terms should be low complexity."""
    chunk = _make_chunk(SIMPLE_TEXT)
    label = score_chunk_complexity(chunk)
    assert label == "low"
    assert chunk.metadata["complexity"] == "low"
    assert chunk.metadata["complexity_score"] <= 0.33


# ===================================================================
# Test: Complex clinical text scores "high"
# ===================================================================

def test_complex_clinical_text_scores_high():
    """Dense ICU protocol text with medications, vitals, and citations."""
    chunk = _make_chunk(COMPLEX_CLINICAL_TEXT)
    label = score_chunk_complexity(chunk)
    assert label == "high"
    assert chunk.metadata["complexity"] == "high"
    assert chunk.metadata["complexity_score"] > 0.66


# ===================================================================
# Test: Medium complexity text scores "medium"
# ===================================================================

def test_medium_text_scores_medium():
    """Moderately complex clinical note with some medications and measurements
    should score medium -- between simple prose and dense ICU charting."""
    chunk = _make_chunk(MEDIUM_TEXT)
    label = score_chunk_complexity(chunk)
    assert label == "medium"
    assert chunk.metadata["complexity"] == "medium"
    assert 0.33 < chunk.metadata["complexity_score"] <= 0.66


# ===================================================================
# Test: Empty text does not crash
# ===================================================================

def test_empty_text_does_not_crash():
    """Empty chunk text should produce a valid score without errors."""
    chunk = _make_chunk("")
    label = score_chunk_complexity(chunk)
    assert label == "low"
    assert chunk.metadata["complexity_score"] == 0.0


# ===================================================================
# Test: Whitespace-only text does not crash
# ===================================================================

def test_whitespace_only_text():
    """Whitespace-only text should behave like empty text."""
    chunk = _make_chunk("   \n\t  ")
    label = score_chunk_complexity(chunk)
    assert label == "low"
    assert chunk.metadata["complexity_score"] == 0.0


# ===================================================================
# Test: Single word text
# ===================================================================

def test_single_word_text():
    """A single word should score low and not crash."""
    chunk = _make_chunk("Hello")
    label = score_chunk_complexity(chunk)
    assert label == "low"
    assert 0.0 <= chunk.metadata["complexity_score"] <= 1.0


# ===================================================================
# Test: Very long text
# ===================================================================

def test_very_long_text():
    """A very long text with many sentences should not crash."""
    long_text = ". ".join(
        [f"Sentence number {i} discusses the clinical presentation of the patient"
         for i in range(1, 51)]
    ) + "."
    chunk = _make_chunk(long_text)
    label = score_chunk_complexity(chunk)
    assert label in ("low", "medium", "high")
    assert 0.0 <= chunk.metadata["complexity_score"] <= 1.0


# ===================================================================
# Test: Text with citations/references
# ===================================================================

def test_citation_heavy_text():
    """Text dense with academic citations should score higher than simple."""
    chunk = _make_chunk(CITATION_HEAVY_TEXT)
    label = score_chunk_complexity(chunk)
    assert label in ("medium", "high")
    assert chunk.metadata["complexity_score"] > 0.33


# ===================================================================
# Test: Text with bullet lists
# ===================================================================

def test_bullet_list_text():
    """A structured checklist with medical terms should score above low."""
    chunk = _make_chunk(BULLET_LIST_TEXT)
    label = score_chunk_complexity(chunk)
    assert label in ("medium", "high")
    assert chunk.metadata["complexity_score"] > 0.33


# ===================================================================
# Test: Text with numbers/measurements (lab results)
# ===================================================================

def test_numeric_heavy_text_above_simple():
    """Lab results dense with numeric values should score well above simple text."""
    chunk = _make_chunk(NUMERIC_HEAVY_TEXT)
    label = score_chunk_complexity(chunk)
    # Lab results are data-dense but structurally straightforward (short
    # sentences, lists of values), so medium is a valid classification.
    assert label in ("medium", "high")
    assert chunk.metadata["complexity_score"] > 0.33


# ===================================================================
# Test: Numeric text scores higher than simple text
# ===================================================================

def test_numeric_text_higher_than_simple():
    """Lab results should score strictly higher than plain prose."""
    chunk_simple = _make_chunk(SIMPLE_TEXT, chunk_id="s")
    chunk_numeric = _make_chunk(NUMERIC_HEAVY_TEXT, chunk_id="n")
    score_chunk_complexity(chunk_simple)
    score_chunk_complexity(chunk_numeric)
    assert (
        chunk_simple.metadata["complexity_score"]
        < chunk_numeric.metadata["complexity_score"]
    )


# ===================================================================
# Test: Integration -- chunks from chunk_documents have complexity
# ===================================================================

def test_chunk_documents_adds_complexity_metadata():
    """chunk_documents() should automatically score each chunk."""
    doc = _make_doc(
        "Simple intro paragraph.\n\n"
        + COMPLEX_CLINICAL_TEXT
    )
    chunks = chunk_documents([doc])
    assert len(chunks) >= 1
    for chunk in chunks:
        assert "complexity" in chunk.metadata
        assert "complexity_score" in chunk.metadata
        assert chunk.metadata["complexity"] in ("low", "medium", "high")
        assert 0.0 <= chunk.metadata["complexity_score"] <= 1.0


# ===================================================================
# Test: Score is reproducible (same text = same score)
# ===================================================================

def test_score_is_reproducible():
    """Identical text must produce identical scores every time."""
    text = COMPLEX_CLINICAL_TEXT
    chunk_a = _make_chunk(text, chunk_id="a")
    chunk_b = _make_chunk(text, chunk_id="b")
    score_chunk_complexity(chunk_a)
    score_chunk_complexity(chunk_b)
    assert chunk_a.metadata["complexity_score"] == chunk_b.metadata["complexity_score"]
    assert chunk_a.metadata["complexity"] == chunk_b.metadata["complexity"]


# ===================================================================
# Test: All chunks get scored (no chunks without complexity)
# ===================================================================

def test_all_chunks_scored():
    """score_chunks must annotate every single chunk, leaving none behind."""
    chunks = [
        _make_chunk("Short.", chunk_id="c0"),
        _make_chunk(MEDIUM_TEXT, chunk_id="c1"),
        _make_chunk(COMPLEX_CLINICAL_TEXT, chunk_id="c2"),
    ]
    result = score_chunks(chunks)
    assert result is chunks  # mutates in place and returns same list
    for chunk in result:
        assert "complexity" in chunk.metadata
        assert "complexity_score" in chunk.metadata


# ===================================================================
# Test: Label bucketing boundaries
# ===================================================================

def test_label_low_boundary():
    """Score of exactly 0.33 should be 'low'."""
    assert _label_from_score(0.33) == "low"


def test_label_medium_boundary():
    """Score of exactly 0.66 should be 'medium'."""
    assert _label_from_score(0.66) == "medium"


def test_label_high_boundary():
    """Score of 0.67 should be 'high'."""
    assert _label_from_score(0.67) == "high"


def test_label_zero():
    """Score of 0.0 should be 'low'."""
    assert _label_from_score(0.0) == "low"


def test_label_one():
    """Score of 1.0 should be 'high'."""
    assert _label_from_score(1.0) == "high"


# ===================================================================
# Test: Composite score range
# ===================================================================

def test_composite_score_always_in_range():
    """_composite_score should always return a value in [0.0, 1.0]."""
    texts = [
        "",
        "x",
        SIMPLE_TEXT,
        MEDIUM_TEXT,
        COMPLEX_CLINICAL_TEXT,
        CITATION_HEAVY_TEXT,
        BULLET_LIST_TEXT,
        NUMERIC_HEAVY_TEXT,
        "a " * 10000,  # very long repetitive text
    ]
    for text in texts:
        score = _composite_score(text)
        assert 0.0 <= score <= 1.0, f"Out of range for text: {text[:60]!r}"


# ===================================================================
# Test: Metadata does not overwrite existing keys
# ===================================================================

def test_existing_metadata_preserved():
    """Scoring should add keys without destroying existing metadata."""
    chunk = _make_chunk(SIMPLE_TEXT)
    chunk.metadata["author"] = "Dr. Jones"
    chunk.metadata["department"] = "Cardiology"
    score_chunk_complexity(chunk)
    assert chunk.metadata["author"] == "Dr. Jones"
    assert chunk.metadata["department"] == "Cardiology"
    assert "complexity" in chunk.metadata
    assert "complexity_score" in chunk.metadata


# ===================================================================
# Test: Score ordering -- simple < complex
# ===================================================================

def test_simple_lower_than_complex():
    """Simple text must have a strictly lower score than complex text."""
    chunk_simple = _make_chunk(SIMPLE_TEXT, chunk_id="s")
    chunk_complex = _make_chunk(COMPLEX_CLINICAL_TEXT, chunk_id="c")
    score_chunk_complexity(chunk_simple)
    score_chunk_complexity(chunk_complex)
    assert (
        chunk_simple.metadata["complexity_score"]
        < chunk_complex.metadata["complexity_score"]
    )


# ===================================================================
# Test: Relative ordering -- medium between simple and complex
# ===================================================================

def test_medium_between_simple_and_complex():
    """Medium text score should fall between simple and complex."""
    chunk_simple = _make_chunk(SIMPLE_TEXT, chunk_id="s")
    chunk_medium = _make_chunk(MEDIUM_TEXT, chunk_id="m")
    chunk_complex = _make_chunk(COMPLEX_CLINICAL_TEXT, chunk_id="c")
    score_chunk_complexity(chunk_simple)
    score_chunk_complexity(chunk_medium)
    score_chunk_complexity(chunk_complex)
    assert (
        chunk_simple.metadata["complexity_score"]
        < chunk_medium.metadata["complexity_score"]
        < chunk_complex.metadata["complexity_score"]
    )


# ===================================================================
# Test: score_chunks returns the same list object
# ===================================================================

def test_score_chunks_returns_same_list():
    """score_chunks should mutate and return the same list, not a copy."""
    chunks = [_make_chunk("Test text.", chunk_id=f"c{i}") for i in range(5)]
    result = score_chunks(chunks)
    assert result is chunks


# ===================================================================
# Test: Technical term detection in clinical abbreviations
# ===================================================================

def test_technical_term_detection():
    """Text with clinical abbreviations like SpO2, PaCO2, GCS should score higher."""
    plain = "The patient was stable and awake and oriented."
    technical = "SpO2 was 92%, PaCO2 was 45mmHg, and GCS was 14."
    chunk_plain = _make_chunk(plain, chunk_id="p")
    chunk_tech = _make_chunk(technical, chunk_id="t")
    score_chunk_complexity(chunk_plain)
    score_chunk_complexity(chunk_tech)
    assert (
        chunk_plain.metadata["complexity_score"]
        < chunk_tech.metadata["complexity_score"]
    )


# ===================================================================
# Test: chunk_documents with multiple documents
# ===================================================================

def test_chunk_documents_multiple_docs_all_scored():
    """All chunks from multiple documents should be scored."""
    docs = [
        _make_doc(SIMPLE_TEXT, doc_id="doc_simple"),
        _make_doc(COMPLEX_CLINICAL_TEXT, doc_id="doc_complex"),
    ]
    chunks = chunk_documents(docs)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert "complexity" in chunk.metadata
        assert "complexity_score" in chunk.metadata
        assert chunk.metadata["complexity"] in ("low", "medium", "high")


# ===================================================================
# Test: score_chunk_complexity return value matches metadata
# ===================================================================

def test_return_value_matches_metadata():
    """The returned label must match what's stored in metadata."""
    chunk = _make_chunk(MEDIUM_TEXT)
    returned_label = score_chunk_complexity(chunk)
    assert returned_label == chunk.metadata["complexity"]


# ===================================================================
# Test: Raw score is a float with 4 decimal places
# ===================================================================

def test_raw_score_precision():
    """The stored raw score should be rounded to 4 decimal places."""
    chunk = _make_chunk(COMPLEX_CLINICAL_TEXT)
    score_chunk_complexity(chunk)
    score = chunk.metadata["complexity_score"]
    assert isinstance(score, float)
    # Check rounding: string representation should have at most 4 decimal digits
    decimal_part = str(score).split(".")[-1] if "." in str(score) else ""
    assert len(decimal_part) <= 4


# ===================================================================
# Test: Re-scoring overwrites previous values
# ===================================================================

def test_rescoring_overwrites():
    """Scoring a chunk twice should overwrite the previous values."""
    chunk = _make_chunk(SIMPLE_TEXT)
    score_chunk_complexity(chunk)
    first_score = chunk.metadata["complexity_score"]

    # Change text and re-score
    chunk.text = COMPLEX_CLINICAL_TEXT
    score_chunk_complexity(chunk)
    assert chunk.metadata["complexity_score"] != first_score
    assert chunk.metadata["complexity_score"] > first_score


# ===================================================================
# Test: score_chunks on empty list
# ===================================================================

def test_score_chunks_empty_list():
    """score_chunks on an empty list should return an empty list."""
    result = score_chunks([])
    assert result == []


# ===================================================================
# Test: Dosage-heavy clinical text
# ===================================================================

def test_dosage_heavy_text():
    """Medication dosages with units and numeric values should push score up."""
    text = (
        "Administer vancomycin 1.5g IV q12h, piperacillin-tazobactam 4.5g IV "
        "q8h, and dexamethasone 10mg IV q6h. Adjust vancomycin dose to achieve "
        "trough level of 15-20 mcg/mL. Renal dosing required if CrCl <30 mL/min. "
        "Amiodarone 150mg IV bolus over 10 minutes, then 1mg/min infusion for 6 "
        "hours, then 0.5mg/min maintenance. Monitor QTc interval; hold if QTc >500ms."
    )
    chunk = _make_chunk(text)
    label = score_chunk_complexity(chunk)
    assert label in ("medium", "high")
    assert chunk.metadata["complexity_score"] > 0.33


# ===================================================================
# Test: Highly repetitive text scores low vocabulary diversity
# ===================================================================

def test_repetitive_text_low_diversity():
    """Text that repeats the same words should score lower than diverse text."""
    repetitive = "The cat sat. The cat sat. The cat sat. The cat sat. The cat sat."
    diverse = (
        "Electrocardiographic monitoring revealed supraventricular tachycardia "
        "with aberrant conduction. Cardioversion was attempted with adenosine "
        "6mg IV push followed by a subsequent 12mg dose administered rapidly."
    )
    chunk_rep = _make_chunk(repetitive, chunk_id="r")
    chunk_div = _make_chunk(diverse, chunk_id="d")
    score_chunk_complexity(chunk_rep)
    score_chunk_complexity(chunk_div)
    assert (
        chunk_rep.metadata["complexity_score"]
        < chunk_div.metadata["complexity_score"]
    )


# ===================================================================
# Test: Chunk with only punctuation
# ===================================================================

def test_only_punctuation():
    """Text that is only punctuation should not crash and score low.

    Punctuation-only strings tokenize to zero words but the sentence
    splitter may produce non-empty sentence fragments, yielding a tiny
    nonzero sentence-count contribution.  The key invariant is that the
    label is "low".
    """
    chunk = _make_chunk("... --- !!! ???")
    label = score_chunk_complexity(chunk)
    assert label == "low"
    assert chunk.metadata["complexity_score"] < 0.1


# ===================================================================
# Test: Clinical workflow text with mixed features
# ===================================================================

def test_mixed_clinical_workflow():
    """A real clinical workflow combining prose, numbers, and structure."""
    text = (
        "Acute Stroke Protocol:\n"
        "1. Establish IV access with two 18-gauge peripheral lines\n"
        "2. Draw STAT labs: CBC, BMP, coagulation studies, troponin, glucose\n"
        "3. Obtain non-contrast CT head within 25 minutes of arrival\n"
        "4. Calculate NIHSS score at bedside\n"
        "5. If ischemic stroke confirmed and within 4.5-hour window, "
        "administer alteplase 0.9mg/kg IV (max 90mg), with 10% as bolus "
        "over 1 minute and remainder infused over 60 minutes\n"
        "6. Monitor blood pressure every 15 minutes for 2 hours post-tPA, "
        "then every 30 minutes for 6 hours, then hourly for 16 hours\n"
        "7. Target SBP <180mmHg and DBP <105mmHg during and after infusion"
    )
    chunk = _make_chunk(text)
    label = score_chunk_complexity(chunk)
    assert label in ("medium", "high")
    assert chunk.metadata["complexity_score"] > 0.33
