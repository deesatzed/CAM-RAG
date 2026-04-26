from cam_rag.retrieval import RetrievalDocument, SparseBM25Retriever, bm25_search, tokenize


def test_tokenize_keeps_lightweight_technical_terms():
    assert tokenize("BM25-style CAM_RAG v2.0!") == ["bm25-style", "cam_rag", "v2", "0"]


def test_sparse_bm25_retrieves_lexical_match_with_metadata():
    retriever = SparseBM25Retriever(
        [
            RetrievalDocument("doc-1", "graph retrieval and reciprocal rank fusion"),
            RetrievalDocument("doc-2", "document ingestion parses folders", {"source": "ingest"}),
            RetrievalDocument(
                "doc-3",
                "lexical sparse retrieval uses bm25 scoring",
                {"source": "search"},
            ),
        ]
    )

    results = retriever.retrieve("sparse bm25 retrieval", k=2)

    assert [result.doc_id for result in results] == ["doc-3", "doc-1"]
    assert results[0].rank == 1
    assert results[0].metadata == {"source": "search"}
    assert results[0].score > results[1].score


def test_bm25_search_accepts_tuple_documents_and_ignores_empty_queries():
    documents = [
        ("a", "alpha beta beta"),
        ("b", "gamma delta"),
    ]

    assert bm25_search(documents, "beta", k=1)[0].doc_id == "a"
    assert bm25_search(documents, "!!!", k=5) == []
