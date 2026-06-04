"""TICKET-022 integration test — exercises scripts.parity_run against
real Alpaca paper + Tastytrade sandbox.

Auto-skips when required env vars are absent so `pytest tests/` stays
green on a fresh clone. Run explicitly with:

    pytest tests/integration/test_parity_run.py -m integration

Required env (set via secrets.env or directly):
    APCA_API_KEY_ID            (Alpaca paper)
    APCA_API_SECRET_KEY
    TASTYTRADE_PROVIDER_SECRET (Tastytrade sandbox)
    TASTYTRADE_REMEMBER_TOKEN
    TASTYTRADE_USE_SANDBOX=true

This test does NOT place orders. It only fetches chains and writes parity
rows to a temp SQLite DB, then verifies the row count and report file are
reasonable.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

# Auto-skip when SDKs absent.
pytest.importorskip("tastytrade", reason="tastytrade SDK not installed; install '.[broker]'")
pytest.importorskip("alpaca", reason="alpaca-py not installed; install '.[broker]'")

from core.config import load_secrets  # noqa: E402

load_secrets()

_required = (
    "APCA_API_KEY_ID", "APCA_API_SECRET_KEY",
    "TASTYTRADE_PROVIDER_SECRET", "TASTYTRADE_REMEMBER_TOKEN",
)
if not all(os.environ.get(k) for k in _required):
    pytest.skip(
        f"Skipping: missing one of {_required}. See docs/GO_LIVE_RUNBOOK.md §5 "
        "Part A for credential provisioning.",
        allow_module_level=True,
    )


@pytest.mark.asyncio
async def test_parity_run_against_real_sandbox(tmp_path):
    """End-to-end: real Alpaca paper + Tastytrade sandbox brokers,
    1-symbol universe, single run. Verifies rows land in the DB and the
    markdown report renders without error."""
    from scripts.parity_run import run

    class _T:
        def __init__(self, symbol, strategies):
            self.symbol = symbol
            self.strategies = strategies

    # Single-symbol universe to keep the integration run cheap. F is a
    # cheap mega-cap with weekly options on both brokers.
    universe = {"tickers": [_T("F", ["monthly_wheel"])]}
    config = {
        "account": {"id": "test", "broker": "alpaca_paper"},
        "database": {"path": str(tmp_path / "parity_integration.db")},
    }
    # Initialize DB with the canonical schema before run() opens it.
    import aiosqlite
    schema = (Path(__file__).resolve().parent.parent.parent / "db" / "schema.sql").read_text(encoding="utf-8")
    async with aiosqlite.connect(config["database"]["path"]) as c:
        await c.executescript(schema)
        await c.commit()

    result = await run(
        config=config,
        universe=universe,
        report_dir=tmp_path / "reports",
        now=datetime.now(UTC).replace(tzinfo=None),
    )
    # Sanity: at least some contracts came back from both brokers.
    assert result["symbols"] == 1
    assert result["rows"] > 0, "No parity rows written — both brokers returned no contracts?"
    report_path = Path(result["report_path"])
    assert report_path.exists()
    body = report_path.read_text(encoding="utf-8")
    assert "Broker parity report" in body
    assert "F" in body
