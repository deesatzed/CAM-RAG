import importlib
import sys
import tomllib
from pathlib import Path

import pytest


@pytest.fixture()
def app_path(monkeypatch):
    path = Path(__file__).parents[1] / "apps" / "ragamuffin"
    monkeypatch.syspath_prepend(str(path))
    for module_name in (
        "ragamuffin_app",
        "ragamuffin_app.__main__",
        "ragamuffin_app.app",
    ):
        sys.modules.pop(module_name, None)
    return path


def test_ragamuffin_pyproject_exposes_console_script(app_path: Path):
    pyproject = tomllib.loads((app_path / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["name"] == "ragamuffin"
    assert pyproject["project"]["requires-python"] == ">=3.11"
    assert pyproject["project"]["scripts"]["ragamuffin"] == "ragamuffin_app:main"
    assert pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        "ragamuffin_app"
    ]


def test_console_script_target_resolves_without_installing(app_path: Path):
    target = tomllib.loads((app_path / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["scripts"]["ragamuffin"]
    module_name, function_name = target.split(":", maxsplit=1)

    module = importlib.import_module(module_name)

    assert getattr(module, function_name) is module.main
    assert callable(module.main)


def test_python_m_entrypoint_delegates_to_cli_main(monkeypatch, tmp_path: Path, capsys, app_path):
    (tmp_path / "note.txt").write_text("Clinical note", encoding="utf-8")
    app = importlib.import_module("ragamuffin_app.app")
    monkeypatch.setattr(
        app,
        "query_documents",
        lambda question, docs_dir: f"{Path(docs_dir).name}: {question}",
    )

    module_main = importlib.import_module("ragamuffin_app.__main__")
    exit_code = module_main.main([str(tmp_path), "What is documented?"])

    assert exit_code == 0
    assert capsys.readouterr().out == f"{tmp_path.name}: What is documented?\n"
