"""Deterministic methodology fixture data."""

from __future__ import annotations

from cam_rag.methodologies.models import Family, Membership, Methodology
from cam_rag.methodologies.store import JsonStore


def seed_store() -> JsonStore:
    store = JsonStore()
    health = store.add_methodology(
        Methodology(
            problem="Silent-on-success health watch command",
            solution="Return no output on success and actionable diagnostics on failure.",
            language="python",
            source="Agent-Reach",
            tags=["source:agent-reach", "category:cli_ux", "brain:python"],
            notes="CLI health command should be quiet unless intervention is needed.",
        )
    )
    fuzzy = store.add_methodology(
        Methodology(
            problem="Fuzzy filter validation with actionable suggestions",
            solution="Validate fuzzy filters and suggest valid nearby names when input is wrong.",
            language="python",
            source="OneDoTenv",
            tags=["source:onedotenv", "category:cli_ux", "brain:python"],
            notes="CLI validation should guide the user to the nearest valid option.",
        )
    )
    retention = store.add_methodology(
        Methodology(
            problem="Tiered time-series data retention with resolution-aware compaction",
            solution="Compact old samples into lower-resolution buckets by retention tier.",
            language="python",
            source="AI-Trader",
            tags=["source:ai-trader", "category:data_processing", "brain:python"],
            notes="Retention logic should preserve recent detail and summarize old history.",
        )
    )

    health_family = store.add_family(
        Family(
            name="silent health watch command",
            barcode="repofrax:python:cli_ux:health-watch",
            category="cli_ux",
            language="python",
            source_repos=["Agent-Reach"],
        )
    )
    fuzzy_family = store.add_family(
        Family(
            name="fuzzy filter validation",
            barcode="repofrax:python:cli_ux:fuzzy-filter",
            category="cli_ux",
            language="python",
            source_repos=["OneDoTenv"],
        )
    )
    retention_family = store.add_family(
        Family(
            name="tiered time-series retention",
            barcode="repofrax:python:data_processing:retention",
            category="data_processing",
            language="python",
            source_repos=["AI-Trader"],
        )
    )
    for family, methodology in (
        (health_family, health),
        (fuzzy_family, fuzzy),
        (retention_family, retention),
    ):
        store.add_membership(
            Membership(
                family_id=family.id,
                methodology_id=methodology.id,
                confidence=0.9,
                review_status="approved",
            )
        )
    return store
