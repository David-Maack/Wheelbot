"""Shared fixtures for Sprint 4 unit tests that need a real SQLite repo layer."""

from __future__ import annotations

from pathlib import Path

import pytest_asyncio

from db.repo import Database, Repos

ROOT = Path(__file__).resolve().parent.parent.parent
SCHEMA = (ROOT / "db" / "schema.sql").read_text(encoding="utf-8")


@pytest_asyncio.fixture
async def db_repos(tmp_path):
    """Fresh tmp-file SQLite DB with the canonical schema applied."""
    db_path = tmp_path / "wheelbot_test.db"
    db = Database(db_path)
    conn = await db.connect()
    await conn.executescript(SCHEMA)
    await conn.commit()
    repos = Repos(db)
    yield repos
    await db.close()
