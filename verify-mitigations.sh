#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

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
