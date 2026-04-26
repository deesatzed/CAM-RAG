import sys
from pathlib import Path

from cam_rag.documents import read_document_folder
from cam_rag.rag import RAGAppSpec


def test_platform_reads_text_and_json_documents(tmp_path: Path):
    (tmp_path / "protocol.md").write_text("# ED Protocol\nUse triage first.", encoding="utf-8")
    (tmp_path / "export.jsonl").write_text(
        '{"content":"ICU transfer criteria.", "metadata":{"unit":"icu"}}\n',
        encoding="utf-8",
    )
    spec = RAGAppSpec(name="docs", supported_extensions=(".md", ".jsonl"))

    documents = read_document_folder(tmp_path, spec)

    assert len(documents) == 2
    assert {document.title for document in documents} == {"ED Protocol", "export"}
    assert all(document.id.startswith("doc_") for document in documents)


def test_ragamuffin_app_uses_platform_loader(tmp_path: Path):
    app_path = Path(__file__).parents[1] / "apps" / "ragamuffin"
    sys.path.insert(0, str(app_path))
    try:
        from ragamuffin_app import load_documents, ragamuffin_spec
    finally:
        sys.path.remove(str(app_path))

    (tmp_path / "icu.txt").write_text("ICU escalation workflow", encoding="utf-8")

    spec = ragamuffin_spec()
    documents = load_documents(tmp_path)

    assert spec.policy.enforce_phi is True
    assert "intensive" in spec.tokenize("ICU")
    assert len(documents) == 1
    assert documents[0].source == "icu.txt"
