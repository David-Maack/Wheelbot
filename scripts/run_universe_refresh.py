"""Weekly universe refresh — cron entrypoint.

    python -m scripts.run_universe_refresh

Cron line (LXC is MDT; Saturday 07:00 MDT — Friday-close data, the operator
has the weekend to review the proposal before Monday's open):

    0 7 * * 6  /opt/wheelbot/.venv/bin/python -m scripts.run_universe_refresh

No-op (logged skip) unless `universe_refresh.enabled: true` in config.
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
from core.notify import make_notifier, set_dispatcher
from data.ivr import IVRProvider
from db.repo import Database, Repos
from intelligence.anthropic_client import AnthropicClient
from intelligence.budget import BudgetTracker
from intelligence.universe_refresh import run_universe_refresh


async def main() -> int:
    config = load_config()
    setup_logging(config)
    db_path = Path(config.get("database", {}).get("path", "wheelbot.db")).expanduser()
    async with Database(db_path) as db:
        repos = Repos(db)
        broker = make_broker(config)
        ivr = IVRProvider(repos.iv_history)
        # Discord notifier so the proposal lands where the operator will see it.
        set_dispatcher(make_notifier(config))
        # strict=True — an unknown model hard-fails the cron run rather than
        # silently bypassing the daily cap (same stance as run_screener).
        budget = BudgetTracker(repos.llm_decisions, config, strict=True)
        anthropic = AnthropicClient(repos.llm_decisions, budget)
        result = await run_universe_refresh(
            broker=broker,
            repos=repos,
            ivr=ivr,
            anthropic=anthropic,
            config=config,
            run_date=datetime.now(UTC).date(),
        )
        if hasattr(broker, "aclose"):
            await broker.aclose()
    log_checkpoint("run_universe_refresh_done", status="ok",
                   **{k: v for k, v in result.items() if k != "guardrail_notes"})
    return 0


if __name__ == "__main__":
    configure_logging()
    sys.exit(asyncio.run(main()))
