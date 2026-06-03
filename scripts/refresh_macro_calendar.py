"""Daily macro-event calendar refresh — TICKET-007 cron entrypoint.

    docker exec wheelbot python -m scripts.refresh_macro_calendar

Suggested crontab (weekdays pre-market):

    0 7 * * 1-5  docker exec wheelbot python -m scripts.refresh_macro_calendar

Pulls events from Finnhub `/calendar/economic` when `FINNHUB_API_KEY` is set,
otherwise falls back to `config/macro_calendar.yaml`. Upserts via UNIQUE
(event_date, event_type) so running multiple times the same day is safe.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from core.checkpoint import configure_logging, log_checkpoint
from core.config import load_config
from core.logs import setup_logging
from data.macro_calendar import refresh_events
from db.repo import Database, Repos


async def main_async() -> int:
    config = load_config()
    setup_logging(config)
    db_path = Path(config.get("database", {}).get("path", "wheelbot.db")).expanduser()
    async with Database(db_path) as db:
        repos = Repos(db)
        distinct_rows, source = await refresh_events(repos, config)
    log_checkpoint(
        "run_refresh_macro_calendar_done",
        status="ok",
        distinct_rows=distinct_rows,
        source=source,
    )
    print(f"macro calendar refreshed: {distinct_rows} distinct rows ({source})")
    return 0


if __name__ == "__main__":
    configure_logging()
    sys.exit(asyncio.run(main_async()))
