"""core/logs — JSON formatter + setup_logging."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from core.logs import JsonFormatter, setup_logging


def test_json_formatter_emits_valid_one_line_json():
    record = logging.LogRecord(
        name="wheelbot.test",
        level=logging.INFO,
        pathname="x.py",
        lineno=1,
        msg="hello",
        args=None,
        exc_info=None,
    )
    out = JsonFormatter().format(record)
    payload = json.loads(out)
    assert payload["level"] == "INFO"
    assert payload["logger"] == "wheelbot.test"
    assert payload["msg"] == "hello"
    assert "\n" not in out


def test_json_formatter_includes_extra_fields():
    record = logging.LogRecord("a", logging.INFO, "x.py", 1, "msg", None, None)
    record.symbol = "F"  # extra
    record.qty = 3
    out = JsonFormatter().format(record)
    payload = json.loads(out)
    assert payload["symbol"] == "F"
    assert payload["qty"] == 3


def test_setup_logging_writes_to_file(tmp_path: Path):
    log_path = tmp_path / "wheelbot.log"
    setup_logging({"logging": {"level": "INFO", "json": True, "path": str(log_path)}})
    logger = logging.getLogger("wheelbot.test")
    logger.info("hello-from-test")
    for h in logging.getLogger().handlers:
        h.flush()
    text = log_path.read_text(encoding="utf-8").strip()
    assert text
    payload = json.loads(text.splitlines()[0])
    assert payload["msg"] == "hello-from-test"


def test_setup_logging_idempotent_clears_old_handlers(tmp_path: Path):
    log_path = tmp_path / "wheelbot.log"
    cfg = {"logging": {"level": "INFO", "json": True, "path": str(log_path)}}
    setup_logging(cfg)
    setup_logging(cfg)
    # Should have console + file = 2 handlers, not 4.
    handlers = logging.getLogger().handlers
    assert len([h for h in handlers if not isinstance(h, logging.NullHandler)]) == 2
