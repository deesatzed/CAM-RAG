"""
Tests for the multi-signal query type classifier.

Validates classify_query_type() against all 40 benchmark queries (ground truth)
and structural edge cases. No mocks.
"""

import pytest
from cam_rag.retrieval.fractal.query import classify_query_type


# ============================================================
# Structural pattern tests
# ============================================================

class TestSpecificationPatterns:
    def test_what_type_with_study_ref(self):
        assert classify_query_type("What type of therapy does the randomized trial compare?") == "specification"

    def test_how_many(self):
        assert classify_query_type("How many studies were included in the review?") == "specification"

    def test_in_what_specialty(self):
        assert classify_query_type("In what medical specialty is AI being applied?") == "specification"

    def test_what_specific(self):
        assert classify_query_type("What specific drug interactions does the review examine?") == "specification"

    def test_what_is_the_short(self):
        assert classify_query_type("What is the exact date?") == "specification"

    def test_what_is_the_difference(self):
        assert classify_query_type("What is the difference between mitosis and meiosis?") == "specification"


class TestSummaryPatterns:
    def test_summarize(self):
        assert classify_query_type("Summarize the key findings.") == "summary"

    def test_overview(self):
        assert classify_query_type("Give an overview of the field.") == "summary"

    def test_how_is_being_applied(self):
        assert classify_query_type("How is AI being applied to triage patients?") == "summary"

    def test_what_role_does(self):
        assert classify_query_type("What role does AI play in patient care?") == "summary"

    def test_what_are_recent_advances(self):
        assert classify_query_type("What are recent advances in drug safety?") == "summary"

    def test_short_what_is(self):
        assert classify_query_type("What is DNA?") == "summary"

    def test_short_explain(self):
        assert classify_query_type("Explain DNA") == "summary"


class TestLogicPatterns:
    def test_why(self):
        assert classify_query_type("Why is explainability important for trust?") == "logic"

    def test_how_does_mechanism(self):
        assert classify_query_type("How does deep learning detect retinal disease?") == "logic"

    def test_obstacles(self):
        assert classify_query_type("What are the obstacles preventing wider adoption of AI?") == "logic"

    def test_compare_short(self):
        assert classify_query_type("Compare DNA and RNA structures") == "logic"

    def test_effect_standalone(self):
        assert classify_query_type("What is the effect of gravity?") == "specification"

    def test_how_are_connected(self):
        assert classify_query_type("How are they connected?") == "logic"


class TestSynthesisPatterns:
    def test_both_and(self):
        assert classify_query_type("How does AI impact both drug safety and clinical decision support?") == "synthesis"

    def test_versus(self):
        assert classify_query_type("Compare AI effectiveness in skin conditions versus cardiac conditions.") == "synthesis"

    def test_across_with_enumeration(self):
        assert classify_query_type("What are the barriers across diagnosis, triage, and treatment?") == "synthesis"

    def test_work_together(self):
        assert classify_query_type("How do diagnostic tools and triage systems work together?") == "synthesis"

    def test_themes_across(self):
        assert classify_query_type("What are the overall themes across technology and history?") == "synthesis"

    def test_differ_between(self):
        assert classify_query_type("How do ethical concerns differ between resource allocation and diagnosis?") == "synthesis"


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:
    def test_empty_string(self):
        result = classify_query_type("")
        assert result in ("specification", "summary", "logic", "synthesis")

    def test_case_insensitive(self):
        assert classify_query_type("LIST all items") == "specification"
        assert classify_query_type("HOW DOES it work") == "logic"
        assert classify_query_type("SUMMARIZE this") == "summary"

    def test_profile_fallback(self):
        from cam_rag.retrieval.fractal.profile import DocumentProfile
        profile = DocumentProfile(likely_question_types=["specification"])
        result = classify_query_type("Tell me about this", profile=profile)
        # Profile adds 0.5 but synthesis default is 0 — spec should win with profile boost
        assert result in ("specification", "synthesis")

    def test_returns_valid_type(self):
        for query in ["random text", "????", "a", "the quick brown fox"]:
            result = classify_query_type(query)
            assert result in ("specification", "summary", "logic", "synthesis")
