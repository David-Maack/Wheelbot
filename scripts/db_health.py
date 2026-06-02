"""Manual DB diagnostic — print journal mode, integrity, size, and last vacuum.

Not a cron. Run on demand when investigating contention / corruption:

    docker exec wheelbot python -m scripts.db_health

Output is plain text — pipe through `tee` or copy into an incident report.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

from core.config import load_config
from db.repo import Database


async def main_async() -> int:
    config = load_config()
    db_path = Path(config.get("database", {}).get("path", "wheelbot.db")).expanduser()
    print(f"WheelBot DB health — {datetime.now(UTC).isoformat()}")
    print(f"  path : {db_path}")
    if not db_path.exists():
        print("  ERROR: db file does not exist")
        return 1
    size_mb = db_path.stat().st_size / 1_048_576
    print(f"  size : {size_mb:.2f} MB")

    async with Database(db_path) as db:
        conn = await db.connect()

        async with conn.execute("PRAGMA journal_mode") as cur:
            row = await cur.fetchone()
            print(f"  journal_mode    : {row[0] if row else '?'}  (expected: wal)")

        async with conn.execute("PRAGMA synchronous") as cur:
            row = await cur.fetchone()
            sync_map = {0: "OFF", 1: "NORMAL", 2: "FULL", 3: "EXTRA"}
            v = row[0] if row else None
            print(f"  synchronous     : {sync_map.get(v, v)}  (expected: NORMAL)")

        async with conn.execute("PRAGMA busy_timeout") as cur:
            row = await cur.fetchone()
            print(f"  busy_timeout_ms : {row[0] if row else '?'}  (expected: 5000)")

        async with conn.execute("PRAGMA foreign_keys") as cur:
            row = await cur.fetchone()
            print(f"  foreign_keys    : {'ON' if row and row[0] else 'OFF'}  (expected: ON)")

        async with conn.execute("PRAGMA integrity_check") as cur:
            row = await cur.fetchone()
            integrity = row[0] if row else "?"
            print(f"  integrity_check : {integrity}  (expected: ok)")

        # Recent activity counters — handy heartbeat.
        async with conn.execute("SELECT COUNT(*) FROM orders") as cur:
            row = await cur.fetchone()
            print(f"  rows.orders     : {row[0] if row else 0}")
        async with conn.execute("SELECT COUNT(*) FROM positions") as cur:
            row = await cur.fetchone()
            print(f"  rows.positions  : {row[0] if row else 0}")
        async with conn.execute("SELECT COUNT(*) FROM wheel_cycles") as cur:
            row = await cur.fetchone()
            print(f"  rows.cycles     : {row[0] if row else 0}")

    return 0 if integrity == "ok" else 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async()))
