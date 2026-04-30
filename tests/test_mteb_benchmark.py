from __future__ import annotations

from types import SimpleNamespace

from cam_rag.benchmarks.mteb import CamRAGMTEBModel, _coerce_texts, _load_model, build_arg_parser, main
from cam_rag.retrieval import HashEmbeddingBackend, OllamaEmbeddingBackend, OpenRouterEmbeddingBackend


def test_cam_rag_mteb_model_encodes_plain_strings() -> None:
    model = CamRAGMTEBModel(HashEmbeddingBackend(dim=16))

    embeddings = model.encode(["alpha protocol", "beta protocol"])

    assert len(embeddings) == 2
    assert all(len(embedding) == 16 for embedding in embeddings)
    assert not (embeddings[0] == embeddings[1]).all()


def test_coerce_texts_handles_loader_style_batches() -> None:
    inputs = [
        ["plain sentence", {"text": "dict sentence"}],
        [{"query": "query sentence", "document": "ignored"}],
    ]

    assert _coerce_texts(inputs) == [
        "plain sentence",
        "dict sentence",
        "query sentence",
    ]


def test_main_runs_hash_model_with_fake_mteb(monkeypatch, tmp_path) -> None:
    calls: dict[str, object] = {}
    fake_tasks = [
        SimpleNamespace(metadata=SimpleNamespace(name="SciFactRetrieval", type="Retrieval"))
    ]

    class FakeMTEB:
        def get_tasks(self, *, tasks, languages=None):
            calls["get_tasks"] = {"tasks": tasks, "languages": languages}
            return fake_tasks

        def evaluate(self, model, **kwargs):
            calls["model"] = model
            calls["evaluate"] = kwargs
            return [{"main_score": 0.0}]

    monkeypatch.setattr("cam_rag.benchmarks.mteb._load_mteb", lambda: FakeMTEB())

    exit_code = main(
        [
            "--model",
            "hash",
            "--hash-dim",
            "8",
            "--tasks",
            "SciFactRetrieval",
            "--output-folder",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert isinstance(calls["model"], CamRAGMTEBModel)
    assert calls["get_tasks"] == {"tasks": ["SciFactRetrieval"], "languages": None}
    assert calls["evaluate"]["encode_kwargs"] == {"batch_size": 32}
    assert calls["evaluate"]["overwrite_strategy"] == "always"
    assert (tmp_path / "cam_rag_mteb_summary.json").exists()


def test_main_filters_benchmark_tasks_with_fake_mteb(monkeypatch, tmp_path) -> None:
    retrieval_task = SimpleNamespace(
        metadata=SimpleNamespace(name="NFCorpusRetrieval", type="Retrieval")
    )
    classification_task = SimpleNamespace(
        metadata=SimpleNamespace(name="Banking77Classification", type="Classification")
    )
    calls: dict[str, object] = {}

    class FakeMTEB:
        def get_model(self, model_name):
            calls["get_model"] = model_name
            return SimpleNamespace(name=model_name)

        def get_benchmark(self, benchmark_name):
            calls["get_benchmark"] = benchmark_name
            return SimpleNamespace(tasks=[retrieval_task, classification_task])

        def filter_tasks(self, benchmark, *, task_types):
            calls["filter_tasks"] = task_types
            return [
                task
                for task in benchmark.tasks
                if task.metadata.type in set(task_types)
            ]

        def evaluate(self, model, **kwargs):
            calls["model"] = model
            calls["evaluate"] = kwargs
            return [{"main_score": 1.0}]

    monkeypatch.setattr("cam_rag.benchmarks.mteb._load_mteb", lambda: FakeMTEB())

    exit_code = main(
        [
            "--model",
            "intfloat/e5-base-v2",
            "--benchmark",
            "MTEB(eng, v2)",
            "--task-type",
            "Retrieval",
            "--output-folder",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert calls["get_model"] == "intfloat/e5-base-v2"
    assert calls["get_benchmark"] == "MTEB(eng, v2)"
    assert calls["filter_tasks"] == ["Retrieval"]
    assert calls["evaluate"]["tasks"] == [retrieval_task]


# ---------------------------------------------------------------------------
# OpenRouter / Ollama prefix loading tests
# ---------------------------------------------------------------------------


def test_load_model_openrouter_prefix(monkeypatch) -> None:
    """openrouter: prefix instantiates OpenRouterEmbeddingBackend."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-for-unit-test")
    parser = build_arg_parser()
    args = parser.parse_args([
        "--model", "openrouter:perplexity/pplx-embed-v1-4b",
    ])
    mteb_stub = SimpleNamespace()  # not used for openrouter path

    model = _load_model(args, mteb_stub)

    assert isinstance(model, CamRAGMTEBModel)
    assert isinstance(model.backend, OpenRouterEmbeddingBackend)
    assert model.backend.model == "perplexity/pplx-embed-v1-4b"
    assert model.backend.dim == 2560  # default dim for openrouter
    assert model.mteb_model_meta["name"] == "openrouter-perplexity/pplx-embed-v1-4b"


def test_load_model_ollama_prefix() -> None:
    """ollama: prefix instantiates OllamaEmbeddingBackend."""
    parser = build_arg_parser()
    args = parser.parse_args([
        "--model", "ollama:bge-m3",
    ])
    mteb_stub = SimpleNamespace()

    model = _load_model(args, mteb_stub)

    assert isinstance(model, CamRAGMTEBModel)
    assert isinstance(model.backend, OllamaEmbeddingBackend)
    assert model.backend.model == "bge-m3"
    assert model.backend.dim == 1024  # default dim for ollama
    assert model.mteb_model_meta["name"] == "ollama-bge-m3"


def test_embedding_dim_arg_passed_to_backend(monkeypatch) -> None:
    """--embedding-dim overrides the default dimension for openrouter backend."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-for-unit-test")
    parser = build_arg_parser()
    args = parser.parse_args([
        "--model", "openrouter:custom/model",
        "--embedding-dim", "768",
    ])
    mteb_stub = SimpleNamespace()

    model = _load_model(args, mteb_stub)

    assert isinstance(model.backend, OpenRouterEmbeddingBackend)
    assert model.backend.dim == 768


def test_embedding_dim_arg_passed_to_ollama_backend() -> None:
    """--embedding-dim overrides the default dimension for ollama backend."""
    parser = build_arg_parser()
    args = parser.parse_args([
        "--model", "ollama:nomic-embed-text",
        "--embedding-dim", "512",
    ])
    mteb_stub = SimpleNamespace()

    model = _load_model(args, mteb_stub)

    assert isinstance(model.backend, OllamaEmbeddingBackend)
    assert model.backend.dim == 512


def test_encode_uses_batch_when_available() -> None:
    """CamRAGMTEBModel.encode() calls embed_batch() when the backend has it."""
    calls: list[str] = []

    class BatchBackend:
        dim = 4

        def embed(self, text: str) -> list[float]:
            calls.append(f"embed:{text}")
            return [1.0] * self.dim

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            calls.append(f"batch:{len(texts)}")
            return [[1.0] * self.dim for _ in texts]

    model = CamRAGMTEBModel(BatchBackend(), name="test-batch")
    result = model.encode(["alpha", "beta", "gamma"])

    assert len(result) == 3
    assert calls == ["batch:3"]


def test_encode_falls_back_to_embed_without_batch() -> None:
    """CamRAGMTEBModel.encode() falls back to per-text embed() calls."""
    model = CamRAGMTEBModel(HashEmbeddingBackend(dim=8))
    result = model.encode(["alpha", "beta"])

    assert len(result) == 2
    assert all(len(v) == 8 for v in result)


# ---------------------------------------------------------------------------
# local: prefix loading test
# ---------------------------------------------------------------------------


def test_load_model_local_prefix() -> None:
    """local: prefix instantiates LocalSentenceTransformerBackend."""
    from cam_rag.finetuning.backend import LocalSentenceTransformerBackend

    parser = build_arg_parser()
    args = parser.parse_args(["--model", "local:all-MiniLM-L6-v2"])
    mteb_stub = SimpleNamespace()

    model = _load_model(args, mteb_stub)

    assert isinstance(model, CamRAGMTEBModel)
    assert isinstance(model.backend, LocalSentenceTransformerBackend)
    assert model.backend.dim == 384
    assert model.mteb_model_meta["name"] == "local-all-MiniLM-L6-v2"
