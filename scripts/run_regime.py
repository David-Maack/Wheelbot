"""Daily regime snapshot — cron entrypoint.

    python -m scripts.run_regime

Cron line (LXC, weekdays after market close):

    30 16 * * 1-5  /opt/wheelbot/.venv/bin/python -m scripts.run_regime
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from core.checkpoint import configure_logging, log_checkpoint
from core.config import load_config
from core.logs import setup_logging
from db.repo import Database, Repos
from risk.regime import run_regime


async def main() -> int:
    config = load_config()
    setup_logging(config)
    db_path = Path(config.get("database", {}).get("path", "wheelbot.db")).expanduser()
    async with Database(db_path) as db:
        repos = Repos(db)
        result = await run_regime(repos, config)
    log_checkpoint(
        "run_regime_done",
        status="ok" if result else "skip",
        regime=result.regime.value if result and hasattr(result.regime, "value") else (result.regime if result else None),
    )
    return 0 if result else 1


if __name__ == "__main__":
    configure_logging()
    sys.exit(asyncio.run(main()))
