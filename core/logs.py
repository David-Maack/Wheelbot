"""Structured JSON logging setup.

Per spec §13 #24: structured JSON to file + console, with rotation. We rotate
in-process via TimedRotatingFileHandler (daily, 30 backups) so we don't depend
on an OS-level logrotate to be installed. `ops/logrotate.conf` exists for hosts
that prefer system rotation — both can coexist.

The `[CHECKPOINT]` line shape from `core/checkpoint.py` is preserved as the
log message; the JSON formatter wraps it with timestamp/level/logger fields so
`grep CHECKPOINT` still works on the file output.

Usage:

    from core.config import load_config
    from core.logs import setup_logging
    setup_logging(load_config())
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    """One JSON object per record. Stable key order for grep-friendliness."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Allow callers to attach arbitrary fields via `extra={"key": val, ...}`.
        for key, value in record.__dict__.items():
            if key in payload or key in _STDLIB_RECORD_FIELDS:
                continue
            try:
                json.dumps(value, default=str)
                payload[key] = value
            except Exception:
                payload[key] = repr(value)
        return json.dumps(payload, default=str, separators=(",", ":"))


_STDLIB_RECORD_FIELDS = {
    "args", "asctime", "created", "exc_info", "exc_text", "filename",
    "funcName", "levelname", "levelno", "lineno", "message", "module",
    "msecs", "msg", "name", "pathname", "process", "processName",
    "relativeCreated", "stack_info", "thread", "threadName", "taskName",
}


def setup_logging(config: dict[str, Any]) -> None:
    """Idempotent — clears existing root handlers before installing ours."""
    section = config.get("logging", {}) or {}
    level = str(section.get("level", "INFO")).upper()
    json_lines = bool(section.get("json", True))
    path = section.get("path")

    formatter: logging.Formatter
    if json_lines:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s | %(message)s"
        )

    root = logging.getLogger()
    root.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    if path:
        log_path = Path(path).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = TimedRotatingFileHandler(
            log_path,
            when="midnight",
            backupCount=30,
            encoding="utf-8",
            utc=True,
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    root.setLevel(level)
