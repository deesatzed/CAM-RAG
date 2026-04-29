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
  --tasks SciFact NFCorpus \
  --output-folder benchmarks/mteb/results/hash
```

Run a real embedding model:

```bash
python benchmarks/mteb/run_retrieval.py \
  --model intfloat/e5-base-v2 \
  --tasks SciFact NFCorpus \
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

## OpenRouter embeddings

Run with an OpenRouter-hosted embedding model (requires `OPENROUTER_API_KEY`):

```bash
python benchmarks/mteb/run_retrieval.py \
  --model openrouter:qwen/qwen3-embedding-8b \
  --tasks SciFact NFCorpus \
  --output-folder benchmarks/mteb/results/openrouter-qwen3-embed-8b \
  --batch-size 16
```

With PseudoRAG spec overrides for labelled comparison:

```bash
python benchmarks/mteb/run_retrieval.py \
  --model openrouter:qwen/qwen3-embedding-8b \
  --tasks SciFact NFCorpus \
  --output-folder benchmarks/mteb/results/openrouter-qwen3-embed-8b-pseudorag \
  --spec-overrides '{"moe_scoring_enabled": true, "accuracy_contracts_enabled": true}' \
  --batch-size 16
```

Override the default embedding dimension with `--embedding-dim`:

```bash
python benchmarks/mteb/run_retrieval.py \
  --model openrouter:custom/model-name \
  --embedding-dim 768 \
  --tasks SciFact \
  --output-folder benchmarks/mteb/results/custom
```

## Baseline scores

Baselines are stored in `baselines/` for regression detection.

| Model | SciFact nDCG@10 | NFCorpus nDCG@10 |
|-------|----------------|-----------------|
| `qwen/qwen3-embedding-8b` (4096d) | 0.768 | 0.384 |
| `hash` (256d) | ~0.02 | ~0.02 |

## Ollama embeddings

Run with a local Ollama embedding model (requires a running Ollama instance):

```bash
python benchmarks/mteb/run_retrieval.py \
  --model ollama:bge-m3 \
  --tasks SciFact NFCorpus \
  --output-folder benchmarks/mteb/results/ollama-bge-m3 \
  --batch-size 16
```
