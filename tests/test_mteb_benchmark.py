from __future__ import annotations

from types import SimpleNamespace

from cam_rag.benchmarks.mteb import CamRAGMTEBModel, _coerce_texts, main
from cam_rag.retrieval import HashEmbeddingBackend


def test_cam_rag_mteb_model_encodes_plain_strings() -> None:
    model = CamRAGMTEBModel(HashEmbeddingBackend(dim=16))

    embeddings = model.encode(["alpha protocol", "beta protocol"])

    assert len(embeddings) == 2
    assert all(len(embedding) == 16 for embedding in embeddings)
    assert embeddings[0] != embeddings[1]


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
    assert calls["evaluate"]["output_folder"] == str(tmp_path)
    assert calls["evaluate"]["encode_kwargs"] == {"batch_size": 32}
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
