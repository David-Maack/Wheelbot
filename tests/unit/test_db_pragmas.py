"""TICKET-003: SQLite WAL + concurrency safety pragmas are applied on connect.

Multiple processes write to the same SQLite file (bot loop + screener cron +
regime cron + daily summary cron + dashboard readers). The pragmas in
Database.connect() are what prevents `database is locked` errors and keeps
the WAL contained.
"""

from __future__ import annotations

import pytest

from db.repo import Database


async def _pragma(conn, name: str):
    async with conn.execute(f"PRAGMA {name}") as cur:
        row = await cur.fetchone()
    return row[0] if row else None


@pytest.mark.asyncio
async def test_wal_mode_is_on_after_connect(tmp_path):
    """journal_mode must be WAL — required for the multi-process layout."""
    db = Database(tmp_path / "wal.db")
    async with db:
        conn = await db.connect()
        assert (await _pragma(conn, "journal_mode")).lower() == "wal"


@pytest.mark.asyncio
async def test_busy_timeout_is_set(tmp_path):
    """Without busy_timeout, a contending statement raises SQLITE_BUSY
    immediately instead of waiting briefly for the writer."""
    db = Database(tmp_path / "bt.db")
    async with db:
        conn = await db.connect()
        assert await _pragma(conn, "busy_timeout") == 5000


@pytest.mark.asyncio
async def test_synchronous_normal(tmp_path):
    """synchronous=NORMAL trades a tiny durability window for write throughput;
    safe with WAL — only a power-loss-mid-commit risks the last txn."""
    db = Database(tmp_path / "sync.db")
    async with db:
        conn = await db.connect()
        # 1 = NORMAL, 2 = FULL.
        assert await _pragma(conn, "synchronous") == 1


@pytest.mark.asyncio
async def test_foreign_keys_on(tmp_path):
    """Required for our cycle_id / position_id FK constraints to actually fire."""
    db = Database(tmp_path / "fk.db")
    async with db:
        conn = await db.connect()
        assert await _pragma(conn, "foreign_keys") == 1
