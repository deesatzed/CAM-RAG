# Merge Plan

## Deliverables

1. Build a reusable RAG platform for many applications.
2. Rebuild Ragamuffin as the first document-folder example app using that
   platform.

## Order

1. Port `repofrax` into `cam_rag.methodologies`.
2. Preserve repofrax tests inside this repo.
3. Extract Ragamuffin document models, ingestion, retrieval, confidence, and
   grounding behind neutral platform contracts.
4. Rebuild `apps/ragamuffin` as a thin app layer.
5. Add benchmark harnesses and ablations for SOTA claims.

## Non-Goals

- Do not merge all nested vendor repos wholesale.
- Do not hardcode clinical assumptions into the generic platform.
- Do not keep two independent RAG stacks alive.
- Do not rely on `sys.path` hacks for production imports.
