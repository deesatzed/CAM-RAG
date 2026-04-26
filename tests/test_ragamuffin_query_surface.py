import sys
import types
from pathlib import Path

import pytest


@pytest.fixture()
def ragamuffin(monkeypatch):
    app_path = Path(__file__).parents[1] / "apps" / "ragamuffin"
    monkeypatch.syspath_prepend(str(app_path))
    for module_name in ("ragamuffin_app", "ragamuffin_app.app"):
        sys.modules.pop(module_name, None)
    import ragamuffin_app

    return ragamuffin_app


def test_query_documents_delegates_to_platform_query(monkeypatch, tmp_path: Path, ragamuffin):
    (tmp_path / "icu.txt").write_text("ICU escalation workflow", encoding="utf-8")
    calls = []

    platform_module = types.ModuleType("cam_rag.query")

    def query(**kwargs):
        calls.append(kwargs)
        return {"answer": "use the escalation workflow"}

    platform_module.query = query
    monkeypatch.setitem(sys.modules, "cam_rag.query", platform_module)

    result = ragamuffin.query_documents("How should we escalate?", tmp_path)

    assert result == {"answer": "use the escalation workflow"}
    assert len(calls) == 1
    assert calls[0]["question"] == "How should we escalate?"
    assert calls[0]["spec"].name == "ragamuffin"
    assert calls[0]["documents"][0].source == "icu.txt"


def test_query_documents_uses_real_platform_query(tmp_path: Path, ragamuffin):
    (tmp_path / "protocol.md").write_text("# Protocol\nUse triage first.", encoding="utf-8")

    result = ragamuffin.query_documents("triage", tmp_path)

    assert result.grounded is True
    assert result.citations[0].source == "protocol.md"


def test_cli_calls_query_surface(monkeypatch, tmp_path: Path, capsys, ragamuffin):
    (tmp_path / "note.txt").write_text("Clinical note", encoding="utf-8")
    monkeypatch.setattr(
        ragamuffin.app,
        "query_documents",
        lambda question, docs_dir: f"{Path(docs_dir).name}: {question}",
    )

    exit_code = ragamuffin.main([str(tmp_path), "What is documented?"])

    assert exit_code == 0
    assert capsys.readouterr().out == f"{tmp_path.name}: What is documented?\n"
