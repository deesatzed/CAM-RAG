"""Run MTEB retrieval benchmarks from a source checkout."""

from __future__ import annotations


def _main() -> int:
    from cam_rag.benchmarks.mteb import main

    return main()


if __name__ == "__main__":
    raise SystemExit(_main())
