"""scripts/preflight_live — pre-go-live readiness checks."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from core.models import DailyState, Position, PositionState
from scripts.preflight_live import (
    CheckResult,
    PreflightReport,
    _check_data_signals,
    _check_db,
    _check_ops,
    _check_universe,
)


def _utc() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def test_preflight_report_ok_only_when_no_failures():
    r = PreflightReport(target_mode="sandbox")
    r.results.append(CheckResult("a", "pass"))
    r.results.append(CheckResult("b", "warn", "something"))
    assert r.ok is True
    r.results.append(CheckResult("c", "fail", "nope"))
    assert r.ok is False


def test_check_universe_uses_load_universe(monkeypatch):
    from core.models import UniverseEntry

    fake_universe = {
        "tickers": [
            UniverseEntry(symbol="F", name="Ford", tier=1, overrides={}),
            UniverseEntry(symbol="HOOD", name="Robinhood", tier=3, overrides={}),
        ],
        "banned": [],
        "banned_rules": [],
    }
    monkeypatch.setattr("scripts.preflight_live.load_universe", lambda: fake_universe)
    report = PreflightReport(target_mode="sandbox")
    _check_universe(report)
    statuses = {r.name: r.status for r in report.results}
    assert statuses["universe"] == "pass"


def test_check_universe_fails_when_no_tier1(monkeypatch):
    from core.models import UniverseEntry

    fake_universe = {
        "tickers": [
            UniverseEntry(symbol="HOOD", name="Robinhood", tier=3, overrides={}),
        ],
        "banned": [],
        "banned_rules": [],
    }
    monkeypatch.setattr("scripts.preflight_live.load_universe", lambda: fake_universe)
    report = PreflightReport(target_mode="sandbox")
    _check_universe(report)
    assert any(r.name == "universe" and r.status == "fail" for r in report.results)


@pytest.mark.asyncio
async def test_check_ops_warns_on_manual_intervention(db_repos):
    await db_repos.positions.insert(
        Position(
            account_id="primary",
            symbol="MYSTERY",
            state=PositionState.MANUAL_INTERVENTION,
            shares=0,
            state_changed_at=_utc(),
            state_change_reason="bad mojo",
        )
    )
    report = PreflightReport(target_mode="sandbox")
    await _check_ops(report, db_repos, {"account": {"id": "primary"}})
    statuses = {r.name: r.status for r in report.results}
    assert statuses["manual_intervention"] == "warn"


@pytest.mark.asyncio
async def test_check_ops_warns_on_armed_kill_switch(db_repos):
    await db_repos.daily_state.upsert(
        DailyState(
            account_id="primary",
            snapshot_date=datetime.now(UTC).date(),
            session_open_equity=10_000,
            kill_switch_armed=True,
            kill_switch_reason="drawdown",
        )
    )
    report = PreflightReport(target_mode="sandbox")
    await _check_ops(report, db_repos, {"account": {"id": "primary"}})
    statuses = {r.name: r.status for r in report.results}
    assert statuses["kill_switch"] == "warn"


@pytest.mark.asyncio
async def test_check_ops_warns_on_stop_file(db_repos, tmp_path):
    stop = tmp_path / "STOP"
    stop.write_text("halt")
    report = PreflightReport(target_mode="sandbox")
    await _check_ops(
        report,
        db_repos,
        {"account": {"id": "primary"}, "risk": {"stop_file_path": str(stop)}},
    )
    statuses = {r.name: r.status for r in report.results}
    assert statuses["stop_file"] == "warn"


@pytest.mark.asyncio
async def test_check_data_signals_warns_when_history_thin(db_repos):
    report = PreflightReport(target_mode="sandbox")
    await _check_data_signals(report, db_repos)
    statuses = {r.name: r.status for r in report.results}
    assert statuses["regime_snapshots"] == "warn"
    assert statuses["iv_history"] == "warn"
