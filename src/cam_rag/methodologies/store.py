"""JSON-backed methodology store for local and test deployments."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from cam_rag.methodologies.models import Family, Membership, Methodology


def _validate_store_path(store_path: Path) -> None:
    """Validate a store path for safety.

    Rejects symlinks, paths containing ``..`` segments, and resolved paths
    that escape the expected parent directory.
    """
    if store_path.is_symlink():
        raise ValueError("symlink store paths are not allowed")

    # Reject any path containing ".." segments
    for part in store_path.parts:
        if part == "..":
            raise ValueError("store paths containing '..' segments are not allowed")

    # Ensure the resolved path stays within the parent directory
    resolved = store_path.resolve()
    parent_resolved = store_path.parent.resolve()
    if (
        not str(resolved).startswith(str(parent_resolved) + "/")
        and resolved != parent_resolved
        and resolved.parent != parent_resolved
    ):
        raise ValueError(
            f"store path resolves outside its parent directory: {resolved}"
        )


@dataclass(slots=True)
class JsonStore:
    methodologies: dict[str, Methodology] = field(default_factory=dict)
    families: dict[str, Family] = field(default_factory=dict)
    memberships: list[Membership] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> JsonStore:
        store_path = Path(path)
        _validate_store_path(store_path)
        if not store_path.exists():
            return cls()
        data = json.loads(store_path.read_text(encoding="utf-8"))
        return cls(
            methodologies={
                item["id"]: Methodology.from_dict(item)
                for item in data.get("methodologies", [])
            },
            families={
                item["id"]: Family.from_dict(item)
                for item in data.get("families", [])
            },
            memberships=[
                Membership.from_dict(item)
                for item in data.get("memberships", [])
            ],
        )

    def save(self, path: str | Path) -> None:
        store_path = Path(path)
        _validate_store_path(store_path)
        store_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "methodologies": [item.to_dict() for item in self.methodologies.values()],
            "families": [item.to_dict() for item in self.families.values()],
            "memberships": [item.to_dict() for item in self.memberships],
        }
        store_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def add_methodology(self, methodology: Methodology) -> Methodology:
        self.methodologies[methodology.id] = methodology
        return methodology

    def add_family(self, family: Family) -> Family:
        self.families[family.id] = family
        return family

    def add_membership(self, membership: Membership) -> Membership:
        self.memberships.append(membership)
        return membership

    def memberships_for_methodology(self, methodology_id: str) -> list[tuple[Family, Membership]]:
        rows: list[tuple[Family, Membership]] = []
        for membership in self.memberships:
            if membership.methodology_id != methodology_id:
                continue
            family = self.families.get(membership.family_id)
            if family is not None:
                rows.append((family, membership))
        return rows

    def methodologies_for_family(self, family_id: str) -> list[tuple[Methodology, Membership]]:
        rows: list[tuple[Methodology, Membership]] = []
        for membership in self.memberships:
            if membership.family_id != family_id:
                continue
            methodology = self.methodologies.get(membership.methodology_id)
            if methodology is not None:
                rows.append((methodology, membership))
        return rows
