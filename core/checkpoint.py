"""Checkpoint logger.

Every meaningful step of the bot emits one line in this shape:

    [CHECKPOINT] step=<name> status=<ok|fail|start> [duration_ms=<n>] context={...}

The point is that when something goes wrong at 2am, `grep CHECKPOINT` on the log file
gives you the timeline. Same convention as PolyTrader (spec §16).

Two ways to use it:

    with checkpoint("place_csp", symbol="F", strike=10):
        ...                       # status=start logged on entry, ok/fail on exit

    log_checkpoint("reconcile_done", status="ok", n_orders=3)
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger("wheelbot.checkpoint")


def _format(step: str, status: str, context: dict[str, Any], duration_ms: float | None) -> str:
    parts = [f"[CHECKPOINT] step={step}", f"status={status}"]
    if duration_ms is not None:
        parts.append(f"duration_ms={duration_ms:.1f}")
    if context:
        parts.append(f"context={json.dumps(context, default=str, separators=(',', ':'))}")
    return " ".join(parts)


def log_checkpoint(step: str, status: str = "ok", **context: Any) -> None:
    """Emit a single checkpoint line. status is conventionally ok | fail | start."""
    line = _format(step, status, context, None)
    if status == "fail":
        logger.error(line)
    else:
        logger.info(line)


@contextmanager
def checkpoint(step: str, **context: Any) -> Iterator[dict[str, Any]]:
    """Context manager: emits status=start on entry, status=ok/fail with duration on exit.

    Mutate the yielded dict to add fields that will appear in the final line:

        with checkpoint("reconcile") as ctx:
            ctx["fills"] = 3
    """
    payload: dict[str, Any] = dict(context)
    logger.info(_format(step, "start", payload, None))
    started = time.perf_counter()
    try:
        yield payload
    except Exception as exc:
        duration_ms = (time.perf_counter() - started) * 1000
        payload.setdefault("error", f"{type(exc).__name__}: {exc}")
        logger.error(_format(step, "fail", payload, duration_ms))
        raise
    else:
        duration_ms = (time.perf_counter() - started) * 1000
        logger.info(_format(step, "ok", payload, duration_ms))


def configure_logging(level: str = "INFO", json_lines: bool = False) -> None:
    """Bare-bones logging setup. Replace with structlog or similar in Sprint 5."""
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler()
    if json_lines:
        fmt = '{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}'
    else:
        fmt = "%(asctime)s %(levelname)s %(name)s | %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(handler)
    root.setLevel(level.upper())
