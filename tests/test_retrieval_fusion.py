import pytest

from cam_rag.retrieval import RetrievalResult, rrf_fuse


def test_rrf_fuse_boosts_documents_seen_by_multiple_sources():
    dense = [
        RetrievalResult("a", "dense a", 0.9, rank=1),
        RetrievalResult("b", "dense b", 0.8, rank=2),
    ]
    sparse = [
        RetrievalResult("c", "sparse c", 3.0, rank=1),
        RetrievalResult("a", "sparse a", 2.0, rank=2),
    ]

    fused = rrf_fuse(dense, sparse, names=["dense", "sparse"])

    assert [result.doc_id for result in fused] == ["a", "c", "b"]
    assert fused[0].source_ranks == {"dense": 1, "sparse": 2}
    assert fused[0].source_scores == {"dense": 0.9, "sparse": 2.0}
    assert fused[0].text == "dense a"


def test_rrf_fuse_uses_weights_and_fallback_ranks():
    dense = [
        RetrievalResult("a", "a", 1.0),
        RetrievalResult("b", "b", 0.5),
    ]
    sparse = [RetrievalResult("b", "b", 2.0)]

    fused = rrf_fuse(dense, sparse, names=["dense", "sparse"], weights={"sparse": 4.0})

    assert fused[0].doc_id == "b"
    assert fused[0].source_ranks == {"dense": 2, "sparse": 1}


def test_rrf_fuse_validates_configuration():
    with pytest.raises(ValueError, match="names"):
        rrf_fuse([], names=["dense", "sparse"])

    with pytest.raises(ValueError, match="rrf_k"):
        rrf_fuse([], rrf_k=0)
