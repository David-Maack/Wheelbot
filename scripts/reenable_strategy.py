"""Manually re-enable a strategy that was soft-disabled by the drawdown
circuit breaker (Sprint 13 sub-sprint 1).

    docker exec wheelbot python -m scripts.reenable_strategy --list
    docker exec wheelbot python -m scripts.reenable_strategy --strategy bear_call_spread

The auto-disable feature stores runtime state in `strategy_runtime_state`.
This script clears the disable record so the bot resumes the strategy on
the next tick. The static `enabled` flag in config.yaml is NOT touched.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from core.config import load_config
from db.repo import Database, Repos


async def main_async(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="Show currently disabled strategies")
    group.add_argument("--strategy", help="Strategy id to re-enable (e.g. put_spread)")
    args = parser.parse_args(argv)

    config = load_config()
    db_path = Path(config.get("database", {}).get("path", "wheelbot.db")).expanduser()
    async with Database(db_path) as db:
        repos = Repos(db)

        if args.list:
            disabled = await repos.strategy_runtime.list_disabled()
            if not disabled:
                print("No strategies are currently auto-disabled.")
                return 0
            print(f"{'strategy':<24} {'disabled_until':<28} reason")
            print("-" * 100)
            for row in disabled:
                print(
                    f"{row['strategy_id']:<24} "
                    f"{row.get('disabled_until') or '—':<28} "
                    f"{row.get('disabled_reason') or '—'}"
                )
            return 0

        # --strategy path
        disabled, reason = await repos.strategy_runtime.is_disabled(args.strategy)
        if not disabled:
            print(f"Strategy {args.strategy!r} is not currently disabled — no-op.")
            return 0
        await repos.strategy_runtime.enable(args.strategy)
        print(f"Re-enabled {args.strategy!r}. (Previously disabled: {reason})")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async()))
