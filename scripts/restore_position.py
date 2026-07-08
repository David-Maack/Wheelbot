"""Restore a MANUAL_INTERVENTION position to a managed state.

The proper operator path out of a manual flag (replaces the raw-SQL ritual).
Restores to the from_state recorded in state_log when the position was
flagged, or an explicit --state. Writes its own state_log row.

Run on the box with the live DB:
    docker exec wheelbot python -m scripts.restore_position --list
    docker exec wheelbot python -m scripts.restore_position --symbol SOFI --strategy calendar
    # explicit target state (rare — default is the from_state at flag time):
    docker exec wheelbot python -m scripts.restore_position \
        --symbol SOFI --strategy calendar --state SPREAD_OPEN
"""

from __future__ import annotations

import argparse
import asyncio
import json

from core.config import load_config
from db.repo import Database, Repos
from risk.manual_flags import list_flagged, restore_position


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true", help="show flagged positions and exit")
    ap.add_argument("--symbol")
    ap.add_argument("--strategy")
    ap.add_argument("--state", default=None,
                    help="explicit target state (default: the from_state at flag time)")
    ap.add_argument("--reason", default="operator restore via scripts/restore_position")
    args = ap.parse_args()

    config = load_config()
    account_id = (config.get("account") or {}).get("id", "primary")
    db = Database(config["database"]["path"])
    repos = Repos(db)
    try:
        if args.list:
            flagged = await list_flagged(repos, account_id=account_id)
            if not flagged:
                print("no MANUAL_INTERVENTION positions")
            for row in flagged:
                print(json.dumps(row, indent=2))
            return
        if not args.symbol or not args.strategy:
            ap.error("--symbol and --strategy are required (or use --list)")
        result = await restore_position(
            repos, args.symbol, args.strategy,
            account_id=account_id, state=args.state, reason=args.reason,
        )
        print(json.dumps(result, indent=2))
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
