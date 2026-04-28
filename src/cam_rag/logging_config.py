"""Structured logging configuration for the CAM RAG platform.

Provides both human-readable and JSON-lines output formats using only
the standard library ``logging`` module.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any


class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=UTC
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Merge any structured extras supplied via StructuredLogger
        extras = getattr(record, "_structured_extras", None)
        if extras and isinstance(extras, dict):
            payload.update(extras)
        return json.dumps(payload, default=str)


_HUMAN_FORMAT = (
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)


def configure_logging(
    level: str = "INFO",
    json_format: bool = False,
) -> None:
    """Configure the root logger for the CAM RAG platform.

    Parameters
    ----------
    level:
        Log level name (``"DEBUG"``, ``"INFO"``, ``"WARNING"``).
    json_format:
        When ``True``, emit structured JSON lines to *stderr*.
        When ``False``, emit a human-readable format.
    """
    root = logging.getLogger()
    numeric_level = getattr(
        logging, level.upper(), logging.INFO
    )
    root.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicate output
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(numeric_level)

    if json_format:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(_HUMAN_FORMAT))

    root.addHandler(handler)


class StructuredLogger:
    """Thin wrapper around :class:`logging.Logger` with extras.

    Usage::

        slog = StructuredLogger("cam_rag.query")
        slog.info(
            "query received",
            query_type="synthesis",
            evidence_count=5,
        )

    In JSON mode keyword arguments appear as top-level JSON keys.
    In human mode they are appended as ``key=value`` pairs.
    """

    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)

    # -- public convenience properties --------------------------------

    @property
    def name(self) -> str:
        return self._logger.name

    # -- logging methods ----------------------------------------------

    def debug(self, msg: str, **extras: Any) -> None:
        self._log(logging.DEBUG, msg, extras)

    def info(self, msg: str, **extras: Any) -> None:
        self._log(logging.INFO, msg, extras)

    def warning(self, msg: str, **extras: Any) -> None:
        self._log(logging.WARNING, msg, extras)

    def error(self, msg: str, **extras: Any) -> None:
        self._log(logging.ERROR, msg, extras)

    # -- internal -----------------------------------------------------

    def _log(
        self,
        level: int,
        msg: str,
        extras: dict[str, Any],
    ) -> None:
        if not self._logger.isEnabledFor(level):
            return
        # Build human-readable suffix for non-JSON formatters
        suffix = (
            " "
            + " ".join(
                f"{k}={v}" for k, v in extras.items()
            )
            if extras
            else ""
        )
        record = self._logger.makeRecord(
            name=self._logger.name,
            level=level,
            fn="",
            lno=0,
            msg=msg + suffix,
            args=(),
            exc_info=None,
        )
        # Attach structured extras for the JSON formatter
        record._structured_extras = extras  # type: ignore[attr-defined]
        self._logger.handle(record)
