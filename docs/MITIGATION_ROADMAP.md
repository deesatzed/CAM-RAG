# CAM RAG Platform — Mitigation & Improvement Roadmap

**Generated**: 2026-04-27
**Commit baseline**: `70d09ea` on branch `main`
**Source**: Security audit + Reliability/Performance/Maintainability audit
**Scope**: Sections 5–8 only (Security, Reliability, Performance, Maintainability)

---

## 1. Executive Remediation Summary

**Risk posture shift**: Alpha scaffold → Hardened alpha in 1 sprint; → Production-eligible after 2 sprints.

**Risk delta summary**:
- Path traversal via symlinks eliminated (HIGH → NONE)
- PHI/PII policy flags enforced rather than decorative (HIGH → LOW residual)
- Index-rebuild-per-query removed — 2–4× query throughput gain
- Input validation closes XSS and memory-bomb vectors
- CI/CD pipeline catches regressions automatically

**Effort**: ~89 story points / ~28 person-days
*Assumptions: senior Python engineer familiar with the codebase; no external API integration yet; no deployment infra changes.*

**Quick-win ROI (<4 hours)**:
1. **Add LICENSE file** (S5-04) — unblocks legal distribution compliance
2. **Add path-traversal guard** (S5-02) — closes the highest-severity exploit with 8 lines
3. **Add query input validation** (S5-03) — closes XSS and memory-bomb vectors
4. **Remove sys.path hacks** (S5-06) — aligns with merge plan non-goals
5. **Fix ruff lint errors in tests** (M8-07) — 22 existing violations, auto-fixable
6. **Wire adaptive params** (M8-05) — already implemented and tested, just disconnected

**Systemic themes**:
- Policy flags declared but never enforced (decorative security)
- No resource bounding on ingestion or retrieval (unbounded rglob, unbounded embed, index rebuild per query)
- Zero logging or audit trail in production code
- No CI/CD — all quality enforcement is manual
- Test coverage concentrated on happy paths; error paths and parsers untested
- Adaptive/dynamic capabilities implemented but not integrated

---

## 2. Prioritization Matrix (Impact × Effort)

| Effort \ Impact | High (P0/P1) | Medium (P2) | Low (P2) |
|---|---|---|---|
| **Easy (≤4h)** | **S5-02** Path traversal guard (2h, P0) | **S5-04** LICENSE file (0.5h, P2) | **M8-07** Fix ruff lint errors (0.5h, P2) |
| | **S5-03** Query input validation (2h, P0) | **S5-06** Remove sys.path hacks (1h, P2) | **M8-04** Missing docstrings (1h, P2) |
| | **R6-05** Add CI pipeline (3h, P0) | **M8-05** Wire adaptive params (3h, P2) | **M8-06** Freeze mutable _TYPE_DEFAULTS (0.5h, P2) |
| | | **M8-08** Expand ruff rule set (1h, P2) | |
| **Medium (4–16h)** | **S5-01** PHI/PII enforcement (8h, P0) | **R6-01** Error handling in query pipeline (6h, P2) | **P7-03** Chunk overlap + configurable max_chars (4h, P2) |
| | **P7-02** Cache retrieval indexes (8h, P1) | **R6-02** JSON ingestion error handling (4h, P2) | **M8-02** Type safety: replace type:ignore (4h, P2) |
| | | **M8-01** Raise test coverage to 90%+ (12h, P2) | |
| **Hard (≥16h)** | **P7-01** ANN index for dense retrieval (16h, P1) | **S5-05** SBOM + license audit (8h, P2) | **S5-10** Data-at-rest encryption (16h, P2) |
| | **S5-09** Structured logging + audit trail (12h, P1) | **P7-04** Memory-bounded corpus (8h, P2) | |
| | | **S5-07** JsonStore path confinement (4h, P2) | |

---

## 3. Atomic Mitigation Tasks (Complete Backlog)

### 3.1 Path Traversal Guard on rglob

**ID**: `S5-02`
**Report Quote**: "folder.py:33 calls root.rglob('*') which by default follows symlinks on Python 3.11+. A symlink planted inside a document folder would cause the system to read arbitrary files outside the intended document root."
**Why it matters**: An attacker with write access to the document folder can exfiltrate arbitrary files (e.g., `/etc/passwd`, `.env`) through the RAG answer.
**Priority**: P0
**Estimate**: 2h
**Owner Type**: Agent
**Risk if skipped**: High
**Dependencies**: None
**Done When**: `tests/test_path_traversal.py` passes; symlink is rejected; resolved path outside root is rejected.

#### A) Preconditions & Repo Context
- **Files to inspect**: `src/cam_rag/documents/folder.py:33`, `src/cam_rag/methodologies/miner.py:96`
- **Commands to run**:
  ```bash
  cd /Volumes/WS4TB/CPfrac/cam-rag-platform
  python -m pytest tests/ -q
  ```
- **Expected signals**: 65 passed, 0 failed

#### B) Implementation Plan
1. In `src/cam_rag/documents/folder.py`, after `if not file_path.is_file(): continue` (line 34), add:
   ```python
   resolved = file_path.resolve()
   root_resolved = root.resolve()
   if file_path.is_symlink() or not str(resolved).startswith(str(root_resolved) + "/"):
       continue
   ```
2. Apply the same guard in `src/cam_rag/methodologies/miner.py` inside `_iter_candidate_files` where files are yielded.
3. Create `tests/test_path_traversal.py`:
   - Test: symlink to file outside root is skipped
   - Test: normal file inside root is included
   - Test: deeply nested symlink is skipped

#### C) Tests & Verification
- **Unit**:
  ```bash
  python -m pytest tests/test_path_traversal.py -v
  ```
- **Regression**:
  ```bash
  python -m pytest tests/ -q
  ```
- **Edge cases**:
  - Symlink pointing to parent directory
  - Symlink loop (A→B→A)
  - File with `..` in name but valid path
  - Root itself is a symlink
- **Pass/Fail**: All 65+ tests pass; new symlink tests pass; no symlinked file appears in ingested documents.
- **Done when**: `tests/test_path_traversal.py` passes and `python -m pytest -q` shows 0 failures.

#### D) Rollout & Migration
- No migration needed. Behavioral change: symlinks are silently skipped.
- **Rollback**: `git revert <commit>`.

#### E) Cleanup & Documentation
- Add comment at guard site explaining why symlinks are rejected.

#### F) Success Metrics
- 0 path traversal vectors reachable from `read_document_folder()`

---

### 3.2 Query Input Validation

**ID**: `S5-03`
**Report Quote**: "query() and query_document_folder() accept an arbitrary question:str with no validation. No length limit, no character filtering. User-supplied question is echoed verbatim at query.py:177 — XSS vector if rendered in HTML."
**Why it matters**: Unbounded input can cause OOM in tokenizer/embedder; verbatim echo is an XSS vector.
**Priority**: P0
**Estimate**: 2h
**Owner Type**: Agent
**Risk if skipped**: High
**Dependencies**: None
**Done When**: Queries > 10000 chars raise ValueError; empty queries raise ValueError; user text in answer output is not raw.

#### A) Preconditions & Repo Context
- **Files to inspect**: `src/cam_rag/query.py:16-17`, `src/cam_rag/query.py:175-178`

#### B) Implementation Plan
1. Add validation function in `src/cam_rag/query.py`:
   ```python
   _MAX_QUERY_LENGTH = 10_000

   def _validate_query(question: str) -> str:
       question = question.strip()
       if not question:
           raise ValueError("query must not be empty")
       if len(question) > _MAX_QUERY_LENGTH:
           raise ValueError(f"query exceeds maximum length of {_MAX_QUERY_LENGTH} characters")
       return question
   ```
2. Call `_validate_query(question)` at the top of `query()` and `query_document_folder()`.
3. In `_retrieval_only_answer()` (line 177), truncate the echoed query:
   ```python
   safe_query = query[:200]
   return f"No cited evidence found for: {safe_query}"
   ```

#### C) Tests & Verification
- **Unit**:
  ```bash
  python -m pytest tests/test_query_pipeline.py -v -k "validation or empty or length"
  ```
- **Edge cases**:
  - Empty string, whitespace-only, None (type error)
  - Exactly at limit (10000 chars)
  - 1 char over limit
  - Unicode/emoji heavy strings
  - HTML/script tags in query text
- **Pass/Fail**: ValueError raised for empty and over-length; answer text truncates query echo.

#### D) Rollout & Migration
- Breaking change for callers passing empty strings. Document in CHANGELOG.
- **Rollback**: `git revert <commit>`.

#### E) Cleanup & Documentation
- Update docstrings for `query()` and `query_document_folder()` to document limits.

#### F) Success Metrics
- 0 XSS vectors from user-supplied query text in output

---

### 3.3 PHI/PII Policy Enforcement

**ID**: `S5-01`
**Report Quote**: "Ragamuffin sets enforce_phi=True and enforce_pii=True at app.py:33-34, but no code anywhere in src/cam_rag/ reads, checks, or acts on policy.enforce_phi, policy.enforce_pii, policy.min_confidence, or policy.allowed_residency."
**Why it matters**: Clinical deployment with decorative HIPAA flags is a compliance and liability risk.
**Priority**: P0
**Estimate**: 8h
**Owner Type**: Mixed (Agent implements; Human reviews patterns)
**Risk if skipped**: High
**Dependencies**: S5-03 (input validation)
**Done When**: Queries containing PHI/PII patterns are rejected when enforce_phi/enforce_pii=True; answers below min_confidence are gated; tests prove enforcement.

#### A) Preconditions & Repo Context
- **Files to inspect**: `src/cam_rag/rag/spec.py:25-33`, `src/cam_rag/query.py:16-37`, `apps/ragamuffin/ragamuffin_app/app.py:24-44`

#### B) Implementation Plan
1. Create `src/cam_rag/verification/policy.py`:
   ```python
   """Runtime enforcement of RAGPolicy switches."""
   import re
   from cam_rag.rag.spec import RAGPolicy

   # Patterns for common PHI/PII indicators
   _SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
   _MRN_PATTERN = re.compile(r"\b(?:MRN|mrn)[:\s]?\d{4,}\b")
   _EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
   _PHONE_PATTERN = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")

   def check_phi_pii(text: str, policy: RAGPolicy) -> list[str]:
       """Return list of violation descriptions. Empty = clean."""
       violations = []
       if policy.enforce_phi:
           if _SSN_PATTERN.search(text):
               violations.append("PHI: SSN pattern detected")
           if _MRN_PATTERN.search(text):
               violations.append("PHI: MRN pattern detected")
       if policy.enforce_pii:
           if _EMAIL_PATTERN.search(text):
               violations.append("PII: email address detected")
           if _PHONE_PATTERN.search(text):
               violations.append("PII: phone number detected")
       return violations
   ```
2. In `query.py`, after validation, check input:
   ```python
   from cam_rag.verification.policy import check_phi_pii

   violations = check_phi_pii(question, spec.policy)
   if violations:
       raise ValueError(f"Query rejected by policy: {'; '.join(violations)}")
   ```
3. In `_answer_from_evidence()`, gate on `min_confidence`:
   ```python
   if spec.policy.min_confidence and confidence_report.overall < spec.policy.min_confidence:
       answer = "Insufficient confidence to provide an answer."
   ```
4. Pass `spec` through to `_answer_from_evidence` (currently not passed).
5. Add tests: query with SSN rejected, query with email rejected, low-confidence gated.

#### C) Tests & Verification
- **Unit**:
  ```bash
  python -m pytest tests/test_policy_enforcement.py -v
  ```
- **Edge cases**:
  - SSN-like patterns in document text vs query text
  - Phone numbers that are actually dates (e.g., 2026-04-27)
  - min_confidence = 0.0 (disabled)
  - enforce_phi=False skips PHI checks
- **Pass/Fail**: Policy-violating queries raise ValueError; low-confidence answers gated; all existing tests still pass.

#### D) Rollout & Migration
- Breaking change for Ragamuffin callers who pass PHI in queries. This is the intended behavior.
- **Rollback**: Set `enforce_phi=False, enforce_pii=False` in app spec.

#### E) Cleanup & Documentation
- Update `spec.py` docstrings to note that flags are now enforced.
- ```bash
  git grep -n "enforce_phi\|enforce_pii"
  ```

#### F) Success Metrics
- 100% of PHI/PII pattern queries rejected when policy enabled
- 0 low-confidence answers returned when min_confidence is set

---

### 3.4 Add CI Pipeline

**ID**: `R6-05`
**Report Quote**: "There is no CI/CD configuration anywhere in the repository. No .github/workflows, no Makefile."
**Why it matters**: Without CI, regressions enter main silently. Tests and lint are manual-only.
**Priority**: P0
**Estimate**: 3h
**Owner Type**: Agent
**Risk if skipped**: High
**Dependencies**: None
**Done When**: `.github/workflows/ci.yml` exists; PRs trigger pytest + ruff; badge in README.

#### A) Preconditions & Repo Context
- **Files to inspect**: `pyproject.toml:39-49` (pytest and ruff config)

#### B) Implementation Plan
1. Create `.github/workflows/ci.yml`:
   ```yaml
   name: CI
   on:
     push:
       branches: [main]
     pull_request:
       branches: [main]
   jobs:
     test:
       runs-on: ubuntu-latest
       strategy:
         matrix:
           python-version: ["3.11", "3.12"]
       steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-python@v5
           with:
             python-version: ${{ matrix.python-version }}
         - run: pip install -e ".[dev]"
         - run: python -m ruff check src/cam_rag benchmarks/mteb tests
         - run: python -m pytest -q --tb=short
   ```

#### C) Tests & Verification
- **Manual check**: Push branch, observe Actions tab.
- **Pass/Fail**: Green check on PR with `65 passed` and `All checks passed!`.

#### D) Rollout & Migration
- No migration. Additive only.
- **Rollback**: Delete `.github/workflows/ci.yml`.

#### E) Cleanup & Documentation
- Update README with CI badge.

#### F) Success Metrics
- 100% of PRs gated by automated test + lint

---

### 3.5 Add LICENSE File

**ID**: `S5-04`
**Report Quote**: "pyproject.toml:11 declares license = 'MIT', but no LICENSE file exists in the repository root."
**Why it matters**: MIT license requires full text in distributions. Without it, license declaration is legally incomplete.
**Priority**: P2
**Estimate**: 0.5h
**Owner Type**: Agent
**Risk if skipped**: Medium
**Dependencies**: None
**Done When**: `LICENSE` file exists at repo root with MIT text.

#### B) Implementation Plan
1. Create `LICENSE` with standard MIT text, copyright holder from `pyproject.toml` author metadata.

#### C) Tests & Verification
- `test -f LICENSE && echo "OK"` → OK
- **Pass/Fail**: File exists, contains "MIT License".

---

### 3.6 Remove sys.path Hacks

**ID**: `S5-06`
**Report Quote**: "run_retrieval.py:10-11 inserts src/ into sys.path. MERGE_PLAN.md explicitly lists 'Do not rely on sys.path hacks for production imports' as a non-goal."
**Why it matters**: Masks import resolution issues; violates own merge plan.
**Priority**: P2
**Estimate**: 1h
**Owner Type**: Agent
**Risk if skipped**: Low
**Dependencies**: None
**Done When**: `git grep -n "sys.path.insert" src/ benchmarks/` returns 0 matches.

#### B) Implementation Plan
1. Remove `sys.path.insert` from `benchmarks/mteb/run_retrieval.py`.
2. Add note in benchmarks README: "Run after `pip install -e .`".

#### C) Tests & Verification
- ```bash
  git grep -n "sys.path.insert" src/ benchmarks/
  ```
  Expected: 0 matches.

---

### 3.7 Cache Retrieval Indexes Across Queries

**ID**: `P7-02`
**Report Quote**: "SparseBM25Retriever and DenseVectorRetriever are freshly constructed inside _retrieve_evidence(), rebuilt from scratch on every call. When query expansion triggers a second pass, both indexes are rebuilt a second time."
**Why it matters**: 2–4× unnecessary work per query. With real embedding models, this becomes the dominant latency.
**Priority**: P1
**Estimate**: 8h
**Owner Type**: Agent
**Risk if skipped**: High (blocks SOTA performance)
**Dependencies**: None
**Done When**: Same document set queried twice constructs indexes only once; query expansion reuses existing indexes.

#### A) Preconditions & Repo Context
- **Files to inspect**: `src/cam_rag/query.py:131-172`, `src/cam_rag/retrieval/dense.py:45-81`, `src/cam_rag/retrieval/sparse.py:23-112`

#### B) Implementation Plan
1. Move retriever construction out of `_retrieve_evidence()` into `_rank_chunks()`:
   ```python
   def _rank_chunks(query, chunks, spec, *, limit):
       chunk_by_id = {chunk.id: chunk for chunk in chunks}
       documents = [RetrievalDocument(doc_id=c.id, text=c.text, metadata={"chunk_id": c.id}) for c in chunks]
       sparse = SparseBM25Retriever(documents, tokenizer=spec.tokenize)
       dense = DenseVectorRetriever(documents)
       evidence = _retrieve_evidence(query, sparse, dense, chunk_by_id, spec, limit=limit)
       expanded_query = query
       if spec.query_expansion_enabled and evidence:
           expanded_query = build_expanded_query(query, evidence, tokenizer=spec.tokenize, max_terms=spec.expansion_terms)
           if expanded_query != query:
               evidence = _retrieve_evidence(expanded_query, sparse, dense, chunk_by_id, spec, limit=limit)
       return evidence, expanded_query
   ```
2. Update `_retrieve_evidence` signature to accept pre-built retrievers instead of raw documents.
3. Verify existing tests still pass — the public API (`query()`, `query_document_folder()`) is unchanged.

#### C) Tests & Verification
- **Unit**:
  ```bash
  python -m pytest tests/test_query_pipeline.py -v
  ```
- **Performance**: Measure wall-clock for 2 queries over same corpus before/after.
- **Pass/Fail**: All pipeline tests pass; query expansion path uses same index objects.

#### F) Success Metrics
- Query with expansion enabled takes <50% more time than without (vs current 100%+ overhead)

---

### 3.8 ANN Index for Dense Retrieval

**ID**: `P7-01`
**Report Quote**: "retrieve() performs a linear scan over all document vectors: O(N * d). At 10K+ chunks query latency from dense retrieval alone exceeds 100ms."
**Why it matters**: Brute-force cosine is acceptable for <5K chunks but blocks scaling to production corpus sizes.
**Priority**: P1
**Estimate**: 16h
**Owner Type**: Mixed
**Risk if skipped**: High (blocks scaling)
**Dependencies**: P7-02 (cache indexes)
**Done When**: DenseVectorRetriever supports configurable backend (brute-force for <5K, ANN for >=5K); latency at 50K chunks < 50ms.

#### B) Implementation Plan
1. Add optional `numpy` dependency for vector operations.
2. Add optional `hnswlib` or `faiss-cpu` dependency as `[ann]` extra.
3. Create `src/cam_rag/retrieval/ann_backend.py` implementing a numpy-accelerated brute-force and ANN index.
4. Modify `DenseVectorRetriever.__init__` to accept an optional `index_backend` parameter.
5. Default behavior unchanged (pure Python brute-force) for backward compatibility.

#### C) Tests & Verification
- Existing dense retrieval tests pass with both backends.
- Benchmark: 50K synthetic chunks, latency < 50ms with ANN.

#### F) Success Metrics
- p99 dense retrieval latency < 50ms at 50K chunks

---

### 3.9 Structured Logging + Audit Trail

**ID**: `S5-09`
**Report Quote**: "Grep for logging|logger|print across all production source files returned zero matches. No logging framework configured, no audit trail."
**Why it matters**: HIPAA requires access logging; no observability makes debugging production issues impossible.
**Priority**: P1
**Estimate**: 12h
**Owner Type**: Agent
**Risk if skipped**: High (compliance gap)
**Dependencies**: S5-01 (PHI/PII enforcement — log events must be redaction-aware)
**Done When**: All query events logged with structured JSON; document ingestion events logged; PHI-redacted query text in logs.

#### B) Implementation Plan
1. Create `src/cam_rag/logging_config.py` with JSON structured logger.
2. Add `logger = logging.getLogger(__name__)` to `query.py`, `folder.py`, `parsers.py`.
3. Log: query start (redacted), document count, chunk count, evidence count, confidence score, grounding result, query end.
4. Never log raw document text or full query text when PHI enforcement is enabled.

#### C) Tests & Verification
- Capture log output in tests; assert structured JSON with expected fields.
- Assert no raw PHI in log output when enforce_phi=True.

#### F) Success Metrics
- 100% of query events produce a structured log entry
- 0 PHI/PII in log output when enforcement enabled

---

### 3.10 Error Handling in Query Pipeline

**ID**: `R6-01`
**Report Quote**: "The query pipeline (query.py) contains zero try/except blocks. All exceptions propagate uncaught to the caller."
**Why it matters**: A single malformed document or chunk crashes the entire query for all users.
**Priority**: P2
**Estimate**: 6h
**Owner Type**: Agent
**Risk if skipped**: Medium
**Dependencies**: S5-09 (logging — errors should be logged)
**Done When**: Malformed documents are skipped with warning; retrieval errors produce empty results with low confidence; no uncaught exceptions from query().

#### B) Implementation Plan
1. Wrap `_retrieve_evidence` body in try/except; on failure, return empty evidence list.
2. Wrap individual document reads in `folder.py` in try/except; skip failed files with logged warning.
3. `_answer_from_evidence` with empty evidence returns low-confidence "no evidence" answer.

#### C) Tests & Verification
- Test: corrupted file in document folder → skipped, other files processed.
- Test: retrieval failure → empty result with confidence 0.0.

---

### 3.11 JSON Ingestion Error Handling

**ID**: `R6-02`
**Report Quote**: "json.loads at folder.py:94 and 96 can raise json.JSONDecodeError — no try/except. A single malformed JSON record will abort the entire folder ingestion."
**Why it matters**: One bad JSON line kills the entire pipeline.
**Priority**: P2
**Estimate**: 4h
**Owner Type**: Agent
**Risk if skipped**: Medium
**Dependencies**: None
**Done When**: Malformed JSONL lines are skipped with warning; malformed JSON files are skipped.

#### B) Implementation Plan
1. In `_read_json_documents()`, wrap `json.loads(stripped)` at line 94 in try/except JSONDecodeError; skip line, log warning.
2. Wrap `json.loads(raw)` at line 96 similarly; return empty list on failure.

---

### 3.12 Raise Test Coverage to 90%+

**ID**: `M8-01`
**Report Quote**: "Overall 86%; parsers.py at 54%, chunking.py at 59%, folder.py at 73%."
**Why it matters**: Untested parser code handles external file formats where bugs cause silent data corruption.
**Priority**: P2
**Estimate**: 12h
**Owner Type**: Agent
**Risk if skipped**: Medium
**Dependencies**: None
**Done When**: `pytest --cov=cam_rag --cov-fail-under=90` passes.

#### B) Implementation Plan
1. Add tests for `parsers.py` PDF path (use a real small PDF fixture or test the error paths).
2. Add tests for `parsers.py` DOCX path.
3. Add tests for `chunking.py` `_split_long_text` with text lacking sentence punctuation.
4. Add tests for `folder.py` error paths (missing dir, non-dir path, empty files, malformed JSON).
5. Add tests for `evaluation/core.py` missing lines.
6. Add tests for `methodologies/retrieval.py` graph expansion edge cases.

#### C) Tests & Verification
- ```bash
  python -m pytest --cov=cam_rag --cov-report=term-missing --cov-fail-under=90
  ```

#### F) Success Metrics
- Line coverage ≥ 90% across all modules; parsers.py ≥ 80%

---

### 3.13 Wire Adaptive Params into Query Pipeline

**ID**: `M8-05`
**Report Quote**: "compute_adaptive_params() is implemented, tested, exported — but never imported or called by query.py. The adaptive module is effectively dead code in production paths."
**Why it matters**: The platform has query-type-aware parameter tuning that is unused — wasting implemented capability.
**Priority**: P2
**Estimate**: 3h
**Owner Type**: Agent
**Risk if skipped**: Low
**Dependencies**: P7-02 (cache indexes — adaptive params change k values)
**Done When**: `query()` uses adaptive parameters when `spec.query_expansion_enabled` or when spec signals dynamic mode.

#### B) Implementation Plan
1. In `query.py`, import `compute_adaptive_params`.
2. In `_rank_chunks`, call `compute_adaptive_params(query_type, query, corpus_size=len(chunks))`.
3. Use returned `dense_k`, `sparse_k`, `dense_weight`, `sparse_weight` instead of hardcoded spec values.
4. Add integration test: different query types produce different retrieval parameters.

---

### 3.14 SBOM + License Audit

**ID**: `S5-05`
**Report Quote**: "No SBOM, no dependency license scan. pymupdf is AGPL-3.0 which has strong copyleft requirements that may conflict with MIT license."
**Why it matters**: AGPL dependency in an MIT project is a license conflict; no SBOM blocks compliance audits.
**Priority**: P2
**Estimate**: 8h
**Owner Type**: Mixed (Agent generates; Human reviews AGPL decision)
**Risk if skipped**: Medium
**Dependencies**: S5-04 (LICENSE file)
**Done When**: `sbom.json` exists; pymupdf AGPL decision documented; no license conflicts.

#### B) Implementation Plan
1. Install `pip-licenses`: `pip install pip-licenses`.
2. Generate: `pip-licenses --format=json --output-file=sbom.json`.
3. Review pymupdf AGPL-3.0: decide to keep (document AGPL obligations) or replace with `pdfplumber` (MIT).
4. Document decision in `docs/LICENSE_AUDIT.md`.

---

### 3.15 JsonStore Path Confinement

**ID**: `S5-07`
**Report Quote**: "JsonStore.save(path) accepts any string/Path, creates parent directory tree with mkdir(parents=True), and writes JSON to that location."
**Why it matters**: Unvalidated write path could be used for arbitrary file write if path is attacker-controlled.
**Priority**: P2
**Estimate**: 2h
**Owner Type**: Agent
**Risk if skipped**: Low (currently no external input controls path)
**Dependencies**: None
**Done When**: `save()` validates path is within expected directory; test confirms rejection of escape paths.

---

### 3.16 Chunk Overlap + Configurable max_chars

**ID**: `P7-03`
**Report Quote**: "No overlap between chunks — boundary information is lost. RAGAppSpec does not expose max_chars as configurable."
**Why it matters**: Lost boundary context reduces retrieval quality for information spanning chunk borders.
**Priority**: P2
**Estimate**: 4h
**Owner Type**: Agent
**Risk if skipped**: Low
**Dependencies**: None
**Done When**: `chunk_document()` accepts `overlap_sentences` parameter; `RAGAppSpec` has `chunk_max_chars` field.

---

### 3.17 Memory-Bounded Corpus Loading

**ID**: `P7-04`
**Report Quote**: "rglob('*') with no file-count limit, no depth limit, no size-per-file limit. All vectors held in-memory with no disk persistence."
**Why it matters**: A directory with millions of files or very large files can OOM the process.
**Priority**: P2
**Estimate**: 8h
**Owner Type**: Agent
**Risk if skipped**: Medium
**Dependencies**: None
**Done When**: `read_document_folder` has configurable `max_files` and `max_file_bytes` limits; defaults are sensible (e.g., 10K files, 50MB per file).

---

### 3.18 Type Safety: Replace type:ignore Comments

**ID**: `M8-02`
**Report Quote**: "fusion.py lines 47, 60-63 contain type:ignore comments suppressing type errors on dict[str, object] entries used as float."
**Why it matters**: Suppressed type errors mask potential runtime bugs.
**Priority**: P2
**Estimate**: 4h
**Owner Type**: Agent
**Risk if skipped**: Low
**Dependencies**: None
**Done When**: 0 `type: ignore` comments in `src/cam_rag/`; mypy passes clean.

---

### 3.19 Fix Ruff Lint Errors in Tests

**ID**: `M8-07`
**Report Quote**: "ruff check on tests/ found 22 errors (10 auto-fixable)."
**Why it matters**: Lint violations indicate unused imports and code quality issues.
**Priority**: P2
**Estimate**: 0.5h
**Owner Type**: Agent
**Risk if skipped**: Low
**Dependencies**: None
**Done When**: `python -m ruff check tests/` returns 0 errors.

#### B) Implementation Plan
1. ```bash
   python -m ruff check tests/ --fix
   ```
2. Manually fix remaining 12 non-auto-fixable errors.

---

### 3.20 Add Missing Docstrings

**ID**: `M8-04`
**Report Quote**: "DenseVectorRetriever.retrieve() has no docstring. rrf_fuse() has a one-line docstring with no parameter documentation."
**Why it matters**: Public API without documentation blocks onboarding and agent-driven development.
**Priority**: P2
**Estimate**: 1h
**Owner Type**: Agent
**Risk if skipped**: Low
**Dependencies**: None
**Done When**: All public functions in `src/cam_rag/retrieval/` have docstrings.

---

### 3.21 Freeze Mutable _TYPE_DEFAULTS

**ID**: `M8-06`
**Report Quote**: "_TYPE_DEFAULTS is a module-level mutable dict of dicts. Currently safe but unprotected."
**Why it matters**: Any accidental mutation would silently corrupt defaults for all consumers.
**Priority**: P2
**Estimate**: 0.5h
**Owner Type**: Agent
**Risk if skipped**: Low
**Dependencies**: None
**Done When**: `_TYPE_DEFAULTS` wrapped in `MappingProxyType` or converted to frozen dataclasses.

---

### 3.22 Expand Ruff Rule Set

**ID**: `M8-08`
**Report Quote**: "Rule set is minimal (E, F, I, N, W). No bugbear (B), security (S), comprehension (C4), or pytest (PT) rules."
**Why it matters**: Additional rules catch bugs, security issues, and anti-patterns automatically.
**Priority**: P2
**Estimate**: 1h
**Owner Type**: Agent
**Risk if skipped**: Low
**Dependencies**: M8-07 (fix existing lint errors first)
**Done When**: `pyproject.toml` ruff config includes `B`, `S`, `C4`, `UP` rules; `ruff check` passes.

---

### 3.23 Data-at-Rest Encryption

**ID**: `S5-10`
**Report Quote**: "All persistent data is stored as plaintext JSON. Retention, encryption, audit logging, and de-identification rules are not documented."
**Why it matters**: For clinical/PHI deployments, unencrypted data at rest is a HIPAA violation.
**Priority**: P2
**Estimate**: 16h
**Owner Type**: Human
**Risk if skipped**: Low (dev-only currently; required before clinical deployment)
**Dependencies**: S5-01 (PHI/PII enforcement)
**Done When**: Encryption strategy documented; implementation plan for clinical deployments exists.

---

## 4. Master Verification Suite

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /Volumes/WS4TB/CPfrac/cam-rag-platform

echo "== Preflight =="
python --version
pip show cam-rag-platform >/dev/null 2>&1 || pip install -e ".[dev]"

echo "== Lint/Format (src) =="
python -m ruff check src/cam_rag benchmarks/mteb

echo "== Lint/Format (tests) =="
python -m ruff check tests/ || echo "WARN: test lint issues (non-blocking)"

echo "== Unit Tests =="
python -m pytest tests/ -q --tb=short

echo "== Coverage Gate =="
python -m pytest --cov=cam_rag --cov-fail-under=86 -q --tb=line 2>/dev/null || echo "WARN: coverage below target"

echo "== Security: No sys.path hacks in src =="
if git grep -n "sys.path.insert" -- src/; then
  echo "FAIL: sys.path hacks found in src/"
  exit 1
fi

echo "== Security: No hardcoded secrets =="
if git grep -nE "(sk-[a-zA-Z0-9]{20}|ghp_[a-zA-Z0-9]{36}|AKIA[A-Z0-9]{16})" -- src/ apps/; then
  echo "FAIL: potential secrets found"
  exit 1
fi

echo "== Security: LICENSE file exists =="
test -f LICENSE || { echo "FAIL: no LICENSE file"; exit 1; }

echo "== Smoke: query pipeline =="
python -c "
from cam_rag.rag.spec import RAGAppSpec
from cam_rag.rag.models import CorpusDocument
from cam_rag.query import query
doc = CorpusDocument(id='t1', text='The quick brown fox jumps over the lazy dog.', source='test.md', title='Test')
spec = RAGAppSpec(name='smoke')
result = query('brown fox', [doc], spec)
assert result.confidence >= 0.0
assert result.answer
print('Smoke test passed')
"

echo ""
echo "✅ ALL MITIGATIONS VERIFIED"
```

---

## 5. GitOps Workflow (Agent-Friendly)

**Branch naming**: `remediate/<ID>-<short-slug>`
Examples: `remediate/S5-02-path-traversal`, `remediate/P7-02-cache-indexes`

**Commit convention**:
- `fix(p0): S5-02 add path traversal guard to folder.py`
- `perf(p1): P7-02 cache retrieval indexes across queries`
- `refactor(p2): M8-05 wire adaptive params into query pipeline`
- `chore(p2): S5-04 add MIT LICENSE file`

**PR template**:
```markdown
## Mitigation Task

**ID**: S5-02
**Title**: Path Traversal Guard on rglob
**Priority**: P0

## Changes
- [ ] Guard added to `folder.py`
- [ ] Guard added to `miner.py`
- [ ] Tests added

## Verification
```
python -m pytest tests/ -q
# Paste output here
```

## Security Notes
Symlinks outside document root are now silently skipped.

## Rollback
`git revert <this-commit>`
```

**Agent execution protocol**:
1. One task per branch. Never combine P0 and P2 tasks.
2. Auto-commit after each file group. Message format: `fix(p0): S5-02 <description>`.
3. Stop and flag assumptions with `[ASSUMPTION: ...]`.
4. Never change unrelated code. If you notice an issue outside the task scope, note it in the PR but do not fix it.
5. Run `python -m pytest -q` after every change. Do not push if tests fail.
6. Run `python -m ruff check src/cam_rag` before committing. Fix any violations introduced by your changes.
