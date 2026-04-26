# Ragamuffin

Ragamuffin is the first application built on `cam_rag`: a document-folder RAG
app for clinical and protocol documents.

Run it from this package path with:

```sh
python -m ragamuffin_app <docs_dir> <question>
```

When installed as a package, it exposes the same CLI as:

```sh
ragamuffin <docs_dir> <question>
```

The app should own:

- document-folder UX
- clinical defaults
- medical tokenizer configuration
- PHI/PII/RBAC policy choices
- sample configs and fixtures

The app should not own duplicate platform logic for dense retrieval, BM25, RRF,
confidence gating, grounding, citations, or graph expansion. Those belong in
`src/cam_rag`.
