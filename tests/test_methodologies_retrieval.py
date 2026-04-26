from cam_rag.methodologies.retrieval import search_families
from cam_rag.methodologies.seed import seed_store


def test_graph_expansion_finds_nearby_family():
    store = seed_store()

    direct = search_families(
        store,
        "silent health watch command",
        limit=3,
        graph_expand=False,
    )
    expanded = search_families(
        store,
        "silent health watch command",
        limit=3,
        graph_expand=True,
    )

    assert len(direct) == 1
    assert len(expanded) >= 2
    graph_results = [result for result in expanded if result.source == "graph"]
    assert graph_results
    assert graph_results[0].family is not None
    assert graph_results[0].family.category == "cli_ux"
    assert "shared_category" in graph_results[0].graph_reasons


def test_graph_expansion_does_not_flood_unrelated_category():
    store = seed_store()

    results = search_families(
        store,
        "silent health watch command",
        limit=3,
        graph_expand=True,
    )

    categories = [result.family.category for result in results if result.family]
    assert "data_processing" not in categories
