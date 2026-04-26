# Ragamuffin

Ragamuffin is the first application built on `cam_rag`: a document-folder RAG
app for clinical and protocol documents.

The app should own:

- document-folder UX
- clinical defaults
- medical tokenizer configuration
- PHI/PII/RBAC policy choices
- sample configs and fixtures

The app should not own duplicate platform logic for dense retrieval, BM25, RRF,
confidence gating, grounding, citations, or graph expansion. Those belong in
`src/cam_rag`.
