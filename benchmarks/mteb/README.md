# MTEB Benchmarks

This folder benchmarks the embedding layer used by `cam_rag`. It does not
measure the full RAG answer pipeline, citation quality, grounding, or
no-evidence behavior. Use `cam_rag.evaluate` for those platform-level metrics.

Install optional benchmark dependencies:

```bash
python -m pip install -e ".[benchmark]"
```

Run the local hash backend as a wiring baseline:

```bash
python benchmarks/mteb/run_retrieval.py \
  --model hash \
  --tasks SciFactRetrieval NFCorpusRetrieval \
  --output-folder benchmarks/mteb/results/hash
```

Run a real embedding model:

```bash
python benchmarks/mteb/run_retrieval.py \
  --model intfloat/e5-base-v2 \
  --tasks SciFactRetrieval NFCorpusRetrieval \
  --output-folder benchmarks/mteb/results/e5-base-v2
```

Run retrieval tasks from the English benchmark:

```bash
python benchmarks/mteb/run_retrieval.py \
  --model intfloat/e5-base-v2 \
  --benchmark "MTEB(eng, v2)" \
  --task-type Retrieval \
  --output-folder benchmarks/mteb/results/e5-base-v2-eng-retrieval
```

The `hash` model is expected to score poorly. Its purpose is to validate the
benchmark harness before running model-backed embeddings.
