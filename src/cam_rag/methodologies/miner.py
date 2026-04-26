"""Lightweight local repository miner for methodology-family retrieval."""

from __future__ import annotations

from pathlib import Path

from cam_rag.methodologies.models import Family, Membership, Methodology
from cam_rag.methodologies.store import JsonStore


CODE_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".md": "markdown",
}

SKIP_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "vendor",
}


def mine_repo(
    repo_path: str | Path,
    store: JsonStore,
    *,
    source: str | None = None,
    max_files: int = 80,
) -> dict[str, int]:
    """Mine local files into deterministic methodology families."""

    root = Path(repo_path)
    if not root.exists():
        raise FileNotFoundError(f"repo path does not exist: {root}")
    source_name = source or root.name
    mined = 0
    for path in _iter_candidate_files(root):
        if mined >= max_files:
            break
        language = CODE_EXTENSIONS.get(path.suffix.lower(), "text")
        relative_path = path.relative_to(root)
        category = _category_for(relative_path)
        methodology = store.add_methodology(
            Methodology(
                problem=f"{category.replace('_', ' ')} methodology from {relative_path}",
                solution=_snippet(path),
                language=language,
                source=source_name,
                tags=[
                    f"source:{source_name}",
                    f"category:{category}",
                    f"brain:{language}",
                ],
                notes=f"Mined from {relative_path}",
            )
        )
        family = store.add_family(
            Family(
                name=f"{category.replace('_', ' ')}: {path.stem.replace('_', ' ')}",
                barcode=f"repofrax:{language}:{category}:{_slug(path.stem)}",
                status="approved",
                category=category,
                language=language,
                source_repos=[source_name],
            )
        )
        store.add_membership(
            Membership(
                family_id=family.id,
                methodology_id=methodology.id,
                confidence=0.75,
                review_status="approved",
            )
        )
        mined += 1
    return {
        "files_mined": mined,
        "methodologies": len(store.methodologies),
        "families": len(store.families),
        "memberships": len(store.memberships),
    }


def _iter_candidate_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in CODE_EXTENSIONS:
            continue
        yield path


def _snippet(path: Path, max_chars: int = 1400) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lines.append(stripped)
        if sum(len(item) for item in lines) > max_chars:
            break
    return "\n".join(lines)[:max_chars]


def _category_for(path: Path) -> str:
    lower = "/".join(part.lower() for part in path.parts)
    if "test" in lower or "spec" in lower:
        return "testing"
    if "cli" in lower or "command" in lower:
        return "cli_ux"
    if "api" in lower or "server" in lower or "route" in lower:
        return "api"
    if "db" in lower or "store" in lower or "schema" in lower:
        return "persistence"
    if "retriev" in lower or "search" in lower or "rag" in lower:
        return "retrieval"
    if "config" in lower or "settings" in lower:
        return "configuration"
    return "implementation"


def _slug(value: str) -> str:
    chars = [ch.lower() if ch.isalnum() else "-" for ch in value]
    slug = "".join(chars).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "method"
