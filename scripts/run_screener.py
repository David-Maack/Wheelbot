"""Daily screener — cron entrypoint.

    python -m scripts.run_screener

Cron line (LXC, weekdays pre-market):

    0 8 * * 1-5  /opt/wheelbot/.venv/bin/python -m scripts.run_screener
"""

from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

from core.broker_factory import make_broker
from core.checkpoint import configure_logging, log_checkpoint
from core.config import load_config
from core.logs import setup_logging
from data.ivr import IVRProvider
from db.repo import Database, Repos
from intelligence.anthropic_client import AnthropicClient
from intelligence.budget import BudgetTracker
from intelligence.news import make_news_source
from intelligence.screener import run_screener


async def main() -> int:
    config = load_config()
    setup_logging(config)
    db_path = Path(config.get("database", {}).get("path", "wheelbot.db")).expanduser()
    async with Database(db_path) as db:
        repos = Repos(db)
        broker = make_broker(config)
        ivr = IVRProvider(repos.iv_history)
        news = make_news_source(config)
        budget = BudgetTracker(repos.llm_decisions, config)
        anthropic = AnthropicClient(repos.llm_decisions, budget)
        result = await run_screener(
            broker=broker,
            repos=repos,
            ivr=ivr,
            news=news,
            anthropic=anthropic,
            config=config,
            run_date=datetime.now(UTC).date(),
        )
        if hasattr(broker, "aclose"):
            await broker.aclose()
    log_checkpoint("run_screener_done", status="ok", **{k: v for k, v in result.items()})
    return 0


if __name__ == "__main__":
    configure_logging()
    sys.exit(asyncio.run(main()))
