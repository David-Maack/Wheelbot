"""Shared fixtures for Sprint 4 unit tests that need a real SQLite repo layer."""

from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio

from db.repo import Database, Repos


@pytest.fixture(autouse=True)
def _entry_window_open(monkeypatch):
    """Default the router's entry-window gate to OPEN so order-placement tests
    aren't dependent on the wall-clock time they happen to run at. Tests that
    specifically exercise the gate override this with their own monkeypatch."""
    try:
        monkeypatch.setattr("execution.router.within_entry_window", lambda **k: True)
    except (ImportError, AttributeError):
        # router not imported in this test module — nothing to stub.
        pass


@pytest.fixture(autouse=True)
def _no_earnings_lookup(monkeypatch):
    """The wheel close path's earnings trigger looks the date up live
    (Finnhub / yfinance). Default it to "no data" so unit tests never touch
    the network; earnings-trigger tests inject `next_earnings_fn`."""
    from data.earnings import EarningsLookup

    monkeypatch.setattr(
        "strategies.wheel_close.default_next_earnings",
        lambda symbol, **k: EarningsLookup(symbol=symbol, next_date=None, source="none"),
    )


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
