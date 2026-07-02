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
        # 2026-07-01 AI audit item #2: daily LLM exception veto (reduce-only) —
        # stamps llm_risk_veto on today's snapshot; the bot halves entry sizes
        # for the day. Gated by config (ships false); EVERY failure fails open.
        if result is not None and bool(
            (config.get("intelligence", {}) or {}).get("llm_regime_veto_enabled", False)
        ):
            try:
                from intelligence.anthropic_client import AnthropicClient
                from intelligence.budget import BudgetTracker
                from intelligence.regime_veto import run_regime_veto

                budget = BudgetTracker(repos.llm_decisions, config, strict=True)
                anthropic = AnthropicClient(repos.llm_decisions, budget)
                await run_regime_veto(repos, config, anthropic)
            except Exception as exc:  # noqa: BLE001 — never block the snapshot
                log_checkpoint("regime_veto_wiring_fail", status="fail", error=str(exc))
    log_checkpoint(
        "run_regime_done",
        status="ok" if result else "skip",
        regime=result.regime.value if result and hasattr(result.regime, "value") else (result.regime if result else None),
    )
    return 0 if result else 1


if __name__ == "__main__":
    configure_logging()
    sys.exit(asyncio.run(main()))
