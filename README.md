# CAM RAG Platform

Clean-room repository for the merged SOTA RAG platform work from CAM-Pulse,
repofrax, and Ragamuffin.

## Purpose

This repo has two deliverables:

1. `cam_rag`: a reusable RAG platform for many applications.
2. `apps/ragamuffin`: the first example application, focused on document-folder
   RAG for clinical/protocol documents.

## Source Roles

- **CAM-Pulse** is the base architecture: memory, verification, attribution,
  agent integration, lifecycle, governance, and benchmark discipline.
- **repofrax** contributes methodology-family retrieval and explainable graph
  expansion.
- **ragamuffin** contributes the concrete document-folder RAG application and
  reusable retrieval techniques such as dense/sparse hybrid search, RRF,
  confidence gating, grounding checks, and policy hooks.

## Target Layout

```text
src/cam_rag/
  rag/              # domain-neutral platform contracts and pipeline pieces
  methodologies/    # repofrax-derived family retrieval
  documents/        # ingestion, parsing, chunking, metadata
  retrieval/        # dense, sparse, fusion, PRF, HyDE, reranking
  verification/     # grounding, citations, confidence, receipts
  policy/           # app-pluggable PHI/PII/RBAC hooks
apps/ragamuffin/    # document-folder app using cam_rag
docs/               # merge plan, architecture, benchmarks
tests/              # platform and app tests
```

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Benchmarking

MTEB benchmarking is optional and targets the embedding layer, not the complete
RAG answer pipeline.

```bash
python -m pip install -e ".[benchmark]"
python benchmarks/mteb/run_retrieval.py \
  --model hash \
  --tasks SciFactRetrieval NFCorpusRetrieval \
  --output-folder benchmarks/mteb/results/hash
```

Use `hash` only as a harness smoke test. For meaningful embedding quality,
benchmark model-backed embeddings such as `intfloat/e5-base-v2`, BGE, Jina,
Nomic, OpenAI, or Voyage adapters, then plug the winning backend into
`DenseVectorRetriever`.

## Current Status

Alpha platform scaffold with working document-folder ingestion, chunking,
methodology retrieval, BM25 sparse retrieval, hash dense retrieval, RRF fusion,
query expansion, confidence scoring, citation grounding, Ragamuffin app wiring,
and local evaluation fixtures. The current dense backend is deterministic and
dependency-free for tests; real model-backed embedding adapters are the next
step before claiming SOTA performance.
