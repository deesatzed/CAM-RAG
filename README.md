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

## Current Status

Initial clean scaffold. The next step is to port `repofrax` into
`cam_rag.methodologies` with its existing tests, then extract Ragamuffin's
document RAG pipeline behind the platform contracts.
