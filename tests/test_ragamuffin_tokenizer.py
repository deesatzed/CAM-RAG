import sys
from pathlib import Path

import pytest


@pytest.fixture()
def ragamuffin(monkeypatch):
    app_path = Path(__file__).parents[1] / "apps" / "ragamuffin"
    monkeypatch.syspath_prepend(str(app_path))
    for module_name in (
        "ragamuffin_app",
        "ragamuffin_app.app",
        "ragamuffin_app.tokenizer",
    ):
        sys.modules.pop(module_name, None)

    import ragamuffin_app

    return ragamuffin_app


def test_medical_tokenize_expands_abbreviations_and_preserves_original(ragamuffin):
    assert ragamuffin.medical_tokenize("ICU ED BBP") == [
        "icu",
        "intensive",
        "care",
        "unit",
        "ed",
        "emergency",
        "department",
        "bbp",
        "bloodborne",
        "pathogen",
    ]


def test_medical_tokenize_removes_stopwords_and_expansion_stopwords(ragamuffin):
    assert ragamuffin.medical_tokenize("Explain NPO and PRN orders") == [
        "npo",
        "nothing",
        "mouth",
        "prn",
        "needed",
        "orders",
    ]


def test_medical_tokenize_handles_case_punctuation_and_numeric_units(ragamuffin):
    assert ragamuffin.medical_tokenize("SpO2 is 98%; give 24mg O2.") == [
        "spo2",
        "oxygen",
        "saturation",
        "98",
        "give",
        "24mg",
        "o2",
        "oxygen",
    ]


def test_ragamuffin_spec_uses_app_owned_medical_tokenizer(ragamuffin):
    spec = ragamuffin.ragamuffin_spec()

    assert spec.tokenizer is ragamuffin.medical_tokenize
    assert "electrocardiogram" in spec.tokenize("ECG")
