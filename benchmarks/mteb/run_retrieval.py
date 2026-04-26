"""Run MTEB retrieval benchmarks from a source checkout."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _main() -> int:
    from cam_rag.benchmarks.mteb import main

    return main()


if __name__ == "__main__":
    raise SystemExit(_main())
