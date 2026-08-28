"""
Query type classification and adaptive level weighting.

Merged from v2 (comprehensive keywords + positional checks)
and v3 (profile fallback for ambiguous queries).
"""

from __future__ import annotations

from .profile import DocumentProfile

BASE_DERIV_WEIGHT = 0.12


def classify_query_type(query: str, profile: DocumentProfile | None = None) -> str:
    """Classify query into specification / summary / logic / synthesis.

    Uses multi-signal scoring: each type accumulates evidence from keywords,
    query structure, and length. The highest-scoring type wins. Ties are broken
    by priority order: synthesis > summary > logic > specification.
    """
    import re

    q = query.lower().strip()
    words = q.split()
    n_words = len(words)

    scores = {"specification": 0.0, "summary": 0.0, "logic": 0.0, "synthesis": 0.0}

    # ---- Detect single-study references (used by multiple categories) ----
    has_study_ref = bool(re.search(
        r"\b(?:the paper|the study|the review|the trial|the systematic review"
        r"|the randomized trial|the overview of)\b", q
    ))

    # ---- Synthesis signals (multi-domain, cross-cutting) ----
    # Strong: explicit cross-domain language
    for kw in ["both", "versus", "work together", "across different",
               "differ between", "and clinical", "and emergency",
               "and pharmacovigilance", "as well as",
               "themes across", "overall themes"]:
        if kw in q:
            scores["synthesis"] += 3.0

    # Compare: synthesis for long cross-domain comparisons, logic for short/direct
    if re.search(r"\bcompare\b", q) and not has_study_ref:
        if n_words > 10:
            scores["synthesis"] += 2.5
        scores["logic"] += 2.0

    # Medium: "how do [plural]" spanning multiple concepts (long queries only)
    if re.search(r"\bhow do\b", q) and n_words > 14:
        scores["synthesis"] += 2.0

    # "across" with enumerated items (diagnosis, triage, and treatment)
    if re.search(r"\bacross\b", q) and q.count(",") >= 1:
        scores["synthesis"] += 2.5

    # Weak: long queries with "and" connecting domains
    if n_words > 14 and q.count(" and ") >= 1:
        scores["synthesis"] += 1.0

    # ---- Summary signals (overview, evidence, applications) ----
    for kw in ["summarize", "overview", "main points", "high level", "briefly"]:
        if kw in q:
            scores["summary"] += 3.0

    # Domain overview patterns
    for kw in ["what approaches", "what are the perspectives",
               "what role does", "what are recent advances",
               "current evidence", "the evidence on"]:
        if kw in q:
            scores["summary"] += 2.5

    # "How is/are [topic] being [applied/used/developed]" = overview, not causal
    if re.search(r"\bhow (?:is|are) .+ being\b", q):
        scores["summary"] += 2.5

    # Evidence/application aggregation patterns
    for kw in ["applications", "advances in", "what is the current",
               "perspectives on", "evidence for", "evidence on"]:
        if kw in q:
            scores["summary"] += 1.5

    # Very short "What is [noun]?" = definition/summary (3-5 words)
    if re.search(r"^what is \w+\??$", q) and n_words <= 5:
        scores["summary"] += 2.5

    # Very short queries with "explain/describe [noun]" = summary
    if n_words <= 4 and words[0:1] and words[0] in ("explain", "describe"):
        scores["summary"] += 2.5

    # ---- Logic / Reasoning signals (causal, mechanistic) ----
    # Strong: causal why
    if re.search(r"\bwhy\b", q):
        scores["logic"] += 3.0

    # "How does X [verb] Y" = causal mechanism
    if re.search(r"\bhow does \w+ \w+\b", q) and n_words <= 14:
        scores["logic"] += 2.5

    # "How do [models/methods] [verb]" = mechanistic (shorter queries)
    how_do_mech = (
        r"\bhow do \w+ \w+ "
        r"(?:predict|detect|identify|classify|improve|assist|enhance)\b"
    )
    if re.search(how_do_mech, q):
        scores["logic"] += 2.5

    # "How are [methods] applied/used to [verb]" without "being" = mechanistic
    if re.search(r"\bhow are .+ (?:applied|used) to\b", q) and "being" not in q:
        scores["logic"] += 2.5

    for kw in ["cause", "what if", "explain the relationship",
               "preventing", "obstacles", "barriers", "challenges",
               "benefits and challenges", "effective for",
               "difference", "connected"]:
        if kw in q:
            scores["logic"] += 2.0

    # "effect" as a standalone word (not "effectiveness")
    if re.search(r"\beffect\b", q):
        scores["logic"] += 2.0

    # "How does" without "being" is typically causal
    if re.search(r"\bhow does\b", q) and "being" not in q:
        scores["logic"] += 1.5

    # Short "how are" without "being" = relational/mechanistic
    if re.search(r"\bhow are\b", q) and "being" not in q and n_words <= 8:
        scores["logic"] += 2.0

    # ---- Specification signals (single fact, single document) ----
    for kw in ["list", "exact", "how many", "name the"]:
        if kw in q:
            scores["specification"] += 3.0

    # "What [specific-noun] does the [paper/study/review]" = asking for one fact
    if re.search(r"\bwhat (?:specific |type of |types of |retinal |brain |clinical |data )", q):
        scores["specification"] += 2.5

    # Single study reference = strong spec signal (asking about ONE paper)
    if has_study_ref:
        scores["specification"] += 2.5

    # "In what [setting/specialty]" = asking for a specific answer
    if re.search(r"\bin what\b", q):
        scores["specification"] += 2.0

    # "What is the [noun]?" = asking for a specific thing (moderate signal)
    if re.search(r"^what is the \w+", q) and n_words <= 10:
        scores["specification"] += 1.5

    for kw in ["specific", "detail", "clause", "section", "policy",
               "procedure", "code", "version"]:
        if kw in q:
            scores["specification"] += 1.5

    # Short factual questions starting with "what" (no overview language)
    if words[0:1] == ["what"] and n_words < 14 and scores["summary"] == 0:
        scores["specification"] += 1.0

    # ---- Profile fallback ----
    if profile and profile.likely_question_types:
        for t in profile.likely_question_types:
            if t in scores:
                scores[t] += 0.5

    # ---- Pick winner (ties broken by priority: synthesis > summary > logic > spec) ----
    priority = ["synthesis", "summary", "logic", "specification"]
    best_type = "synthesis"
    best_score = scores["synthesis"]
    for t in priority:
        if scores[t] > best_score:
            best_score = scores[t]
            best_type = t

    return best_type


def get_type_weights(query_type: str, profile: DocumentProfile | None = None) -> dict:
    """Adaptive level weights + derivative multiplier per query type.

    Level weights: [doc_weight, para_weight, sent_weight] — must sum to ~1.0.
    deriv_mult: multiplier on derivative bonus (higher = more emphasis on relationships).
    k_per_level: how many results to return per level.
    """
    deriv_mult_factor = 1.0
    if profile:
        config = profile.to_config()
        deriv_mult_factor = config["priority_bias"]["deriv"]

    base = {
        "specification": {"levels": [0.15, 0.30, 0.55], "deriv_mult": 1.8, "k_per_level": 4},
        "summary":       {"levels": [0.55, 0.30, 0.15], "deriv_mult": 0.5, "k_per_level": 2},
        "logic":         {"levels": [0.25, 0.40, 0.35], "deriv_mult": 1.5, "k_per_level": 3},
        "synthesis":     {"levels": [0.35, 0.40, 0.25], "deriv_mult": 1.0, "k_per_level": 3},
    }[query_type]

    base["deriv_mult"] *= deriv_mult_factor

    if profile:
        base["k_per_level"] = {"low": 2, "medium": 3, "high": 4}[profile.complexity_level]

    return base


# Domain keyword mapping (from corpus domain names)
_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "ai_radiology_pathology": [
        "radiology", "imaging", "radiograph", "ct scan", "mri", "x-ray",
        "pathology", "histopathology", "scan", "retinal", "fundus",
        "medical imaging", "diagnostic imaging", "computer vision",
    ],
    "ai_clinical_decision": [
        "clinical decision", "clinical support", "triage", "diagnosis",
        "patient outcome", "ehr", "electronic health record", "emr",
        "clinical workflow", "patient safety",
    ],
    "ai_drug_discovery_pharma": [
        "drug", "pharmaceutical", "pharmacovigilance", "medication",
        "drug discovery", "adverse drug", "pharmacogenomics",
        "drug metabolism", "cytochrome", "therapeutic drug",
    ],
    "nlp_medical": [
        "nlp", "natural language", "text mining", "clinical notes",
        "named entity", "medical text", "information extraction",
        "clinical text", "language model", "bert",
    ],
    "ai_surgery_robotics": [
        "surgery", "surgical", "robotic", "robot-assisted",
        "laparoscopic", "teleoperation", "haptic", "operative",
    ],
    "ai_public_health_tic": [
        "public health", "telemedicine", "telehealth", "remote monitoring",
        "epidemic", "pandemic", "surveillance", "health informatics",
        "population health", "wearable",
    ],
}


def extract_domain_hints(query: str) -> dict | None:
    """Detect domain keywords in query and return metadata boost config.

    Returns None if no domain signal detected (no boost applied).
    If one or two domains detected, returns a domain_boost config.
    """
    q = query.lower()
    domain_scores: dict[str, int] = {}

    for domain, keywords in _DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in q)
        if score > 0:
            domain_scores[domain] = score

    if not domain_scores:
        return None

    # Sort by score descending
    sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
    top_domain, top_score = sorted_domains[0]

    if top_score >= 2 or (top_score >= 1 and len(sorted_domains) == 1):
        return {
            "domain_boost": 0.05,
            "domain_target": top_domain,
        }

    # If 2+ domains detected (synthesis-like), boost all detected
    if len(sorted_domains) >= 2:
        targets = [d for d, _ in sorted_domains]
        return {
            "domain_boost": 0.03,
            "domain_target": targets,
        }

    return None
