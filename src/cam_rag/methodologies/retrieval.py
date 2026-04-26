"""Family-first methodology retrieval with graph expansion."""

from __future__ import annotations

from dataclasses import dataclass

from cam_rag.methodologies.models import Family, FamilyResult, Methodology, RowResult
from cam_rag.methodologies.store import JsonStore
from cam_rag.methodologies.text import overlap_score, tokens


def search_rows(store: JsonStore, query: str, limit: int = 15) -> list[RowResult]:
    rows: list[RowResult] = []
    for methodology in store.methodologies.values():
        document = " ".join(
            [
                methodology.problem,
                methodology.solution,
                methodology.notes,
                methodology.language,
                " ".join(methodology.tags),
                methodology.source,
            ]
        )
        score = overlap_score(query, document)
        if score > 0:
            rows.append(RowResult(methodology=methodology, score=score))
    return sorted(rows, key=lambda item: item.score, reverse=True)[:limit]


def search_families(
    store: JsonStore,
    query: str,
    *,
    limit: int = 5,
    approved_only: bool = True,
    graph_expand: bool = True,
    row_candidate_multiplier: int = 3,
) -> list[FamilyResult]:
    row_results = search_rows(store, query, limit=limit * max(1, row_candidate_multiplier))
    buckets: dict[str, FamilyResult] = {}

    for row in row_results:
        family_rows = store.memberships_for_methodology(row.methodology.id)
        if approved_only:
            family_rows = [
                (family, membership)
                for family, membership in family_rows
                if family.status == "approved" and membership.review_status == "approved"
            ]
        if not family_rows:
            key = f"methodology:{row.methodology.id}"
            buckets[key] = FamilyResult(
                key=key,
                score=row.score,
                family=None,
                representative=row.methodology,
                methodology_ids=[row.methodology.id],
            )
            continue

        for family, membership in family_rows:
            key = f"family:{family.id}"
            score = row.score + max(0.0, min(1.0, membership.confidence)) * 0.05
            existing = buckets.get(key)
            members = [
                methodology.id
                for methodology, _member in store.methodologies_for_family(family.id)
            ]
            if existing is None or score > existing.score:
                buckets[key] = FamilyResult(
                    key=key,
                    score=score,
                    family=family,
                    representative=row.methodology,
                    methodology_ids=members or [row.methodology.id],
                )

    if graph_expand:
        _add_graph_neighbors(store, buckets, approved_only=approved_only, limit=limit)

    return sorted(buckets.values(), key=lambda item: item.score, reverse=True)[:limit]


@dataclass(frozen=True, slots=True)
class _Profile:
    family: Family
    representative: Methodology | None
    methodology_ids: list[str]
    text_tokens: set[str]
    categories: set[str]
    sources: set[str]
    languages: set[str]
    split_origins: set[str]


def _add_graph_neighbors(
    store: JsonStore,
    buckets: dict[str, FamilyResult],
    *,
    approved_only: bool,
    limit: int,
) -> None:
    seeds = [
        result
        for result in buckets.values()
        if result.family is not None and result.source == "direct"
    ]
    if not seeds:
        return

    profiles = {
        family.id: _profile(store, family)
        for family in store.families.values()
        if not approved_only or family.status == "approved"
    }
    scored: dict[str, FamilyResult] = {}

    for seed in seeds:
        if seed.family is None:
            continue
        seed_profile = profiles.get(seed.family.id)
        if seed_profile is None:
            continue
        category_only_edges = 0
        for candidate_id, candidate_profile in profiles.items():
            if candidate_id == seed.family.id:
                continue
            key = f"family:{candidate_id}"
            edge_weight, reasons = _edge_weight(seed_profile, candidate_profile)
            if edge_weight < 0.30:
                continue
            category_only = (
                "shared_category" in reasons
                and not any(
                    reason.startswith("lexical_overlap")
                    or reason in {"same_split_origin", "shared_source"}
                    for reason in reasons
                )
            )
            if category_only:
                category_only_edges += 1
                if category_only_edges > 1:
                    continue
            graph_score = max(0.0, seed.score - (0.09 - min(edge_weight, 1.0) * 0.06))
            reasons = [f"from:{seed.family.id}", *reasons]

            existing = buckets.get(key)
            if existing is not None:
                if graph_score > existing.score:
                    existing.score = graph_score
                    existing.source = "direct+graph"
                    existing.graph_reasons = reasons
                continue

            representative = candidate_profile.representative
            if representative is None:
                continue
            previous = scored.get(key)
            if previous is not None and previous.score >= graph_score:
                continue
            scored[key] = FamilyResult(
                key=key,
                score=graph_score,
                family=candidate_profile.family,
                representative=representative,
                methodology_ids=candidate_profile.methodology_ids,
                source="graph",
                graph_reasons=reasons,
            )

    for result in sorted(scored.values(), key=lambda item: item.score, reverse=True)[:limit]:
        buckets[result.key] = result


def _profile(store: JsonStore, family: Family) -> _Profile:
    members = store.methodologies_for_family(family.id)
    methodologies = [methodology for methodology, _member in members]
    text = " ".join(
        [
            family.name,
            family.barcode.replace(":", " ").replace("-", " "),
            family.category,
            family.language,
            " ".join(family.source_repos),
            *[
                " ".join(
                    [
                        methodology.problem,
                        methodology.notes,
                        methodology.language,
                        " ".join(methodology.tags),
                        methodology.source,
                    ]
                )
                for methodology in methodologies
            ],
        ]
    )
    categories = {family.category.lower()} if family.category else set()
    sources = {source.lower() for source in family.source_repos}
    languages = {family.language.lower()} if family.language else set()
    for methodology in methodologies:
        if methodology.source:
            sources.add(methodology.source.lower())
        if methodology.language:
            languages.add(methodology.language.lower())
        for tag in methodology.tags:
            if tag.startswith("category:"):
                categories.add(tag.split(":", 1)[1].lower())
            elif tag.startswith("source:"):
                sources.add(tag.split(":", 1)[1].lower())
            elif tag.startswith("brain:"):
                languages.add(tag.split(":", 1)[1].lower())
    return _Profile(
        family=family,
        representative=methodologies[0] if methodologies else None,
        methodology_ids=[methodology.id for methodology in methodologies],
        text_tokens=tokens(text),
        categories=categories,
        sources=sources,
        languages=languages,
        split_origins=_split_origins(family),
    )


def _edge_weight(source: _Profile, candidate: _Profile) -> tuple[float, list[str]]:
    reasons: list[str] = []
    weight = 0.0
    if source.split_origins.intersection(candidate.split_origins):
        weight += 0.95
        reasons.append("same_split_origin")
    if source.sources.intersection(candidate.sources):
        weight += 0.55
        reasons.append("shared_source")
    if source.categories.intersection(candidate.categories):
        weight += 0.58
        reasons.append("shared_category")
    if source.languages.intersection(candidate.languages):
        weight += 0.05
        reasons.append("shared_language")
    overlap = source.text_tokens.intersection(candidate.text_tokens)
    if overlap:
        lexical = len(overlap) / max(1, min(len(source.text_tokens), len(candidate.text_tokens)))
        weight += min(0.65, 0.22 + lexical * 1.4)
        reasons.append(f"lexical_overlap:{','.join(sorted(overlap)[:5])}")
    return min(1.0, weight), reasons


def _split_origins(family: Family) -> set[str]:
    origins: set[str] = set()
    for key in ("split_from_family_id", "split_from_family_barcode"):
        value = family.provenance.get(key)
        if isinstance(value, str) and value:
            origins.add(value)
    for event_key in ("split_events", "merge_events"):
        events = family.provenance.get(event_key)
        if not isinstance(events, list):
            continue
        for event in events:
            if not isinstance(event, dict):
                continue
            for key in ("split_from_family_id", "split_from_family_barcode"):
                value = event.get(key)
                if isinstance(value, str) and value:
                    origins.add(value)
    return origins
