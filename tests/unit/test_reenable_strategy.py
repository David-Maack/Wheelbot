"""scripts/reenable_strategy — operator override for runtime gates (TICKET-009 refinements)."""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from datetime import UTC, datetime, timedelta

import pytest

from scripts.reenable_strategy import _format_row, main_async


@pytest.fixture
def _patched_db(monkeypatch, db_repos, tmp_path):
    """Stub load_config + Database open so the script targets our test repo."""
    db_path = tmp_path / "wheelbot_test.db"
    monkeypatch.setattr(
        "scripts.reenable_strategy.load_config",
        lambda: {"database": {"path": str(db_path)}},
    )
    # Override the Database context manager so it returns the live test
    # connection rather than opening a fresh one against the on-disk file.
    class _NoOpDb:
        async def __aenter__(self):
            return db_repos.db
        async def __aexit__(self, *exc):
            return None
    monkeypatch.setattr("scripts.reenable_strategy.Database", lambda _: _NoOpDb())
    # Ensure Repos(...) wraps our existing db not a fresh handle.
    monkeypatch.setattr("scripts.reenable_strategy.Repos", lambda _db: db_repos)
    return db_repos


# -- _format_row -----------------------------------------------------------


def test_format_row_shows_both_gates_when_both_set():
    row = {
        "drawdown_state": "DISABLED",
        "disabled_until": "2030-01-01T00:00:00",
        "disabled_reason": "drawdown trip",
        "pause_state": "LOW_WIN_RATE",
        "paused_at": "2025-01-01T00:00:00",
        "paused_reenable_eligible_at": "2025-01-15T00:00:00",
        "paused_reason": "win rate trip",
    }
    out = _format_row(row)
    assert "drawdown_state=DISABLED" in out
    assert "pause_state=LOW_WIN_RATE" in out
    assert "drawdown trip" in out
    assert "win rate trip" in out


def test_format_row_pause_only():
    row = {
        "pause_state": "LOW_WIN_RATE",
        "paused_at": "2025-01-01T00:00:00",
        "paused_reenable_eligible_at": "2025-01-15T00:00:00",
        "paused_reason": "low win rate",
    }
    out = _format_row(row)
    assert "drawdown_state" not in out
    assert "pause_state=LOW_WIN_RATE" in out


def test_format_row_empty():
    assert _format_row({}) == "(no active state)"


# -- script flow -----------------------------------------------------------


@pytest.mark.asyncio
async def test_list_shows_paused_and_disabled_merged(_patched_db):
    """--list output combines drawdown-disabled and win-rate-paused
    strategies into one merged view per strategy_id."""
    now = datetime.now(UTC).replace(tzinfo=None)
    await _patched_db.strategy_runtime.disable(
        "put_spread", until=now + timedelta(days=30), reason="drawdown", now=now,
    )
    await _patched_db.strategy_runtime.mark_paused(
        "bear_call_spread", reason="low win rate", now=now,
    )
    # One row with both gates set.
    await _patched_db.strategy_runtime.disable(
        "monthly_wheel", until=now + timedelta(days=30), reason="d2", now=now,
    )
    await _patched_db.strategy_runtime.mark_paused(
        "monthly_wheel", reason="p2", now=now,
    )

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = await main_async(["--list"])
    output = buf.getvalue()
    assert rc == 0
    assert "put_spread" in output
    assert "bear_call_spread" in output
    # The dual-state strategy appears once with both flags.
    assert "monthly_wheel" in output
    assert output.count("monthly_wheel") == 1
    assert "drawdown_state" in output and "pause_state" in output


@pytest.mark.asyncio
async def test_reenable_prints_prior_state_before_clearing(_patched_db):
    """The script must print the prior state BEFORE running enable()
    so the operator sees what's about to be cleared."""
    now = datetime.now(UTC).replace(tzinfo=None)
    await _patched_db.strategy_runtime.disable(
        "put_spread", until=now + timedelta(days=30),
        reason="drawdown trip detail", now=now,
    )
    await _patched_db.strategy_runtime.mark_paused(
        "put_spread", reason="low win rate detail", now=now,
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = await main_async(["--strategy", "put_spread"])
    output = buf.getvalue()
    assert rc == 0
    # Prior state surfaced before the success line — both gates shown.
    assert "prior:" in output
    assert "drawdown_state=DISABLED" in output
    assert "pause_state=LOW_WIN_RATE" in output
    assert "drawdown trip detail" in output
    assert "low win rate detail" in output
    # Row cleared.
    assert await _patched_db.strategy_runtime.get("put_spread") is None


@pytest.mark.asyncio
async def test_reenable_with_reason_emits_audit_checkpoint(_patched_db, monkeypatch):
    """--reason gets propagated into the strategy_manual_reenable checkpoint
    so post-hoc grep on logs surfaces operator rationale."""
    captured: list[dict] = []
    def _stub_checkpoint(event, **kw):
        captured.append({"event": event, **kw})
    monkeypatch.setattr("scripts.reenable_strategy.log_checkpoint", _stub_checkpoint)

    now = datetime.now(UTC).replace(tzinfo=None)
    await _patched_db.strategy_runtime.mark_paused(
        "put_spread", reason="loss streak", now=now,
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        await main_async([
            "--strategy", "put_spread",
            "--reason", "tightened universe; KMI sector rotation",
        ])
    audit = [c for c in captured if c["event"] == "strategy_manual_reenable"]
    assert len(audit) == 1
    assert audit[0]["reason"] == "tightened universe; KMI sector rotation"
    assert audit[0]["prior_pause_state"] == "LOW_WIN_RATE"
    assert audit[0]["prior_paused_reason"] == "loss streak"


@pytest.mark.asyncio
async def test_reenable_no_reason_still_logs_audit(_patched_db, monkeypatch):
    """No --reason flag → reason field is None in the audit but the
    checkpoint still fires (so we can grep for every reenable, even
    unattributed ones)."""
    captured: list[dict] = []
    monkeypatch.setattr(
        "scripts.reenable_strategy.log_checkpoint",
        lambda event, **kw: captured.append({"event": event, **kw}),
    )
    now = datetime.now(UTC).replace(tzinfo=None)
    await _patched_db.strategy_runtime.disable(
        "put_spread", until=now + timedelta(days=30), reason="d", now=now,
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        await main_async(["--strategy", "put_spread"])
    output = buf.getvalue()
    assert "consider passing --reason" in output
    audit = [c for c in captured if c["event"] == "strategy_manual_reenable"]
    assert len(audit) == 1
    assert audit[0]["reason"] is None


@pytest.mark.asyncio
async def test_reenable_noop_when_no_state(_patched_db):
    """Reenabling a strategy with no runtime state is a clean no-op."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = await main_async(["--strategy", "put_spread"])
    assert rc == 0
    assert "no active runtime state" in buf.getvalue()
