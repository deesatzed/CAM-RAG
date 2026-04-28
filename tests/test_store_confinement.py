"""Tests for S5-07: JsonStore path confinement."""

from __future__ import annotations

from pathlib import Path

import pytest

from cam_rag.methodologies.store import JsonStore


def test_load_rejects_symlink(tmp_path: Path):
    """load() raises ValueError for symlink paths."""
    real_file = tmp_path / "real.json"
    real_file.write_text("{}", encoding="utf-8")
    link_path = tmp_path / "link.json"
    link_path.symlink_to(real_file)

    with pytest.raises(ValueError, match="symlink"):
        JsonStore.load(link_path)


def test_save_rejects_symlink(tmp_path: Path):
    """save() raises ValueError for symlink paths."""
    real_dir = tmp_path / "real_dir"
    real_dir.mkdir()
    link_path = tmp_path / "link_dir"
    link_path.symlink_to(real_dir)
    # Create the symlink at the file level to test directly
    real_file = tmp_path / "actual.json"
    real_file.write_text("{}", encoding="utf-8")
    file_link = tmp_path / "file_link.json"
    file_link.symlink_to(real_file)

    store = JsonStore()
    with pytest.raises(ValueError, match="symlink"):
        store.save(file_link)


def test_load_rejects_dotdot_segments(tmp_path: Path):
    """load() raises ValueError for paths containing '..' segments."""
    # Construct a path that contains ".." even though it might resolve fine
    path_with_dotdot = tmp_path / "sub" / ".." / "store.json"

    with pytest.raises(ValueError, match=r"\.\."):
        JsonStore.load(path_with_dotdot)


def test_save_rejects_dotdot_segments(tmp_path: Path):
    """save() raises ValueError for paths containing '..' segments."""
    path_with_dotdot = tmp_path / "sub" / ".." / "store.json"

    store = JsonStore()
    with pytest.raises(ValueError, match=r"\.\."):
        store.save(path_with_dotdot)


def test_load_normal_path_works(tmp_path: Path):
    """load() works fine with a normal, non-symlink, no-dotdot path."""
    store_path = tmp_path / "store.json"
    # File doesn't exist -- load should return an empty store
    loaded = JsonStore.load(store_path)
    assert len(loaded.methodologies) == 0
    assert len(loaded.families) == 0
    assert len(loaded.memberships) == 0


def test_save_normal_path_works(tmp_path: Path):
    """save() works fine with a normal, non-symlink, no-dotdot path."""
    store_path = tmp_path / "store.json"
    store = JsonStore()
    store.save(store_path)
    assert store_path.exists()
    # Verify it's valid JSON
    import json

    data = json.loads(store_path.read_text(encoding="utf-8"))
    assert "methodologies" in data
    assert "families" in data
    assert "memberships" in data


def test_round_trip_with_confinement(tmp_path: Path):
    """Validate save-then-load still works with confinement checks active."""
    from cam_rag.methodologies.models import Family, Membership, Methodology

    store = JsonStore()
    meth = Methodology(problem="test problem", solution="test solution", id="meth_01")
    fam = Family(name="test family", barcode="TF001", id="fam_01")
    mem = Membership(family_id="fam_01", methodology_id="meth_01")
    store.add_methodology(meth)
    store.add_family(fam)
    store.add_membership(mem)

    store_path = tmp_path / "roundtrip.json"
    store.save(store_path)
    loaded = JsonStore.load(store_path)

    assert len(loaded.methodologies) == 1
    assert loaded.methodologies["meth_01"].problem == "test problem"
    assert len(loaded.families) == 1
    assert loaded.families["fam_01"].name == "test family"
    assert len(loaded.memberships) == 1


def test_load_rejects_deeply_nested_dotdot(tmp_path: Path):
    """Deeply nested '..' paths are also rejected."""
    deep_path = tmp_path / "a" / "b" / ".." / ".." / "store.json"
    with pytest.raises(ValueError, match=r"\.\."):
        JsonStore.load(deep_path)


def test_save_rejects_deeply_nested_dotdot(tmp_path: Path):
    """Deeply nested '..' paths are also rejected on save."""
    deep_path = tmp_path / "a" / "b" / ".." / ".." / "store.json"
    store = JsonStore()
    with pytest.raises(ValueError, match=r"\.\."):
        store.save(deep_path)


def test_load_nonexistent_symlink_rejects(tmp_path: Path):
    """A dangling symlink is still rejected."""
    dangling = tmp_path / "dangling.json"
    dangling.symlink_to(tmp_path / "does_not_exist.json")

    with pytest.raises(ValueError, match="symlink"):
        JsonStore.load(dangling)
