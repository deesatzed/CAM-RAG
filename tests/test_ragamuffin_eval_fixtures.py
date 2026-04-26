import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
RAGAMUFFIN_ROOT = REPO_ROOT / "apps" / "ragamuffin"
SAMPLE_ROOT = RAGAMUFFIN_ROOT / "eval" / "document_folder_sample"
SAMPLE_DOCS = SAMPLE_ROOT / "docs"
GOLDEN_QUESTIONS = SAMPLE_ROOT / "golden_questions.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_ragamuffin_eval_sample_files_are_present():
    assert SAMPLE_DOCS.is_dir()
    assert GOLDEN_QUESTIONS.is_file()
    assert sorted(path.name for path in SAMPLE_DOCS.iterdir()) == [
        "ed_triage.md",
        "icu_escalation.txt",
        "outpatient_followup.jsonl",
    ]


def test_ragamuffin_eval_goldens_have_stable_shape():
    rows = _load_jsonl(GOLDEN_QUESTIONS)

    assert len(rows) == 3
    assert {row["id"] for row in rows} == {
        "discharge_followup_timing",
        "ed_triage_chest_pain",
        "icu_transfer_criteria",
    }
    for row in rows:
        assert set(row) == {
            "answer",
            "expected_sources",
            "id",
            "must_include",
            "question",
        }
        assert isinstance(row["id"], str) and row["id"]
        assert isinstance(row["question"], str) and row["question"].endswith("?")
        assert isinstance(row["answer"], str) and row["answer"]
        assert isinstance(row["expected_sources"], list) and row["expected_sources"]
        assert isinstance(row["must_include"], list) and row["must_include"]
        assert all(isinstance(source, str) and source for source in row["expected_sources"])
        assert all(isinstance(term, str) and term for term in row["must_include"])


def test_ragamuffin_eval_goldens_reference_loadable_sample_documents(monkeypatch):
    monkeypatch.syspath_prepend(str(RAGAMUFFIN_ROOT))
    for module_name in ("ragamuffin_app", "ragamuffin_app.app"):
        sys.modules.pop(module_name, None)

    from ragamuffin_app import load_documents

    documents = load_documents(SAMPLE_DOCS)
    document_sources = {document.source for document in documents}
    expected_sources = {
        source
        for row in _load_jsonl(GOLDEN_QUESTIONS)
        for source in row["expected_sources"]
    }

    assert len(documents) == 3
    assert document_sources == {
        "ed_triage.md",
        "icu_escalation.txt",
        "outpatient_followup.jsonl#0",
    }
    assert expected_sources <= document_sources
    assert all(document.text.strip() for document in documents)


def test_ragamuffin_eval_goldens_are_ready_for_future_eval_api(monkeypatch):
    monkeypatch.syspath_prepend(str(RAGAMUFFIN_ROOT))

    from ragamuffin_eval import load_eval_fixture

    fixture = load_eval_fixture(SAMPLE_ROOT)

    assert fixture.docs_dir == SAMPLE_DOCS
    assert [case.id for case in fixture.cases] == [
        "ed_triage_chest_pain",
        "icu_transfer_criteria",
        "discharge_followup_timing",
    ]
