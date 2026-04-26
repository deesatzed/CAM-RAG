from pathlib import Path

from cam_rag.methodologies.miner import mine_repo
from cam_rag.methodologies.seed import seed_store
from cam_rag.methodologies.store import JsonStore


def test_store_round_trip(tmp_path: Path):
    path = tmp_path / "store.json"
    store = seed_store()
    store.save(path)

    loaded = JsonStore.load(path)

    assert len(loaded.methodologies) == 3
    assert len(loaded.families) == 3
    assert len(loaded.memberships) == 3


def test_mine_repo_adds_families(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "cli_health.py").write_text(
        "def health():\n    return {'status': 'ok'}\n",
        encoding="utf-8",
    )
    store = JsonStore()

    report = mine_repo(repo, store, source="demo")

    assert report["files_mined"] == 1
    assert len(store.methodologies) == 1
    assert len(store.families) == 1
    assert store.families[next(iter(store.families))].category == "cli_ux"
