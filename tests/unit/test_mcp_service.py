"""WheelBot Ops MCP service layer — read tools + guarded controls."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from core.models import Position, PositionState, Regime, RegimeSnapshot
from mcp_server.service import ControlsDisabled, WheelbotMcpService
from platforms.paper_broker import PaperBroker


def _config(tmp_path) -> dict:
    return {
        "account": {"id": "test", "broker": "paper", "max_concurrent_total": 4},
        "risk": {"stop_file_path": str(tmp_path / "STOP")},
    }


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _service(db_repos, tmp_path, *, controls_enabled=True, cash=20_000) -> WheelbotMcpService:
    return WheelbotMcpService(
        db_repos, PaperBroker(cash=cash), _config(tmp_path), controls_enabled=controls_enabled,
    )


@pytest.mark.asyncio
async def test_controls_disabled_raises(db_repos, tmp_path):
    svc = _service(db_repos, tmp_path, controls_enabled=False)
    with pytest.raises(ControlsDisabled):
        await svc.pause_strategy("put_spread")
    with pytest.raises(ControlsDisabled):
        await svc.engage_kill_switch()
    # read tools still work with controls off
    assert (await svc.get_positions())["count"] == 0


@pytest.mark.asyncio
async def test_pause_then_reenable_strategy(db_repos, tmp_path):
    svc = _service(db_repos, tmp_path)
    res = await svc.pause_strategy("put_spread", reason="test pause")
    assert res["paused"] is True
    rt = await db_repos.strategy_runtime.get("put_spread")
    assert rt is not None and rt.get("pause_state") == "LOW_WIN_RATE"

    res2 = await svc.reenable_strategy("put_spread")
    assert res2["cleared"] is True
    assert await db_repos.strategy_runtime.get("put_spread") is None


@pytest.mark.asyncio
async def test_kill_switch_engage_release(db_repos, tmp_path):
    svc = _service(db_repos, tmp_path)
    stop = tmp_path / "STOP"
    assert not stop.exists()
    eng = await svc.engage_kill_switch(reason="halt")
    assert eng["kill_switch"] == "ENGAGED" and stop.exists()
    rel = await svc.release_kill_switch()
    assert rel["was_engaged"] is True and not stop.exists()


@pytest.mark.asyncio
async def test_get_positions_reads_db(db_repos, tmp_path):
    await db_repos.positions.insert(Position(
        account_id="test", symbol="F", strategy_id="put_spread",
        state=PositionState.CSP_OPEN, shares=0, state_changed_at=_utcnow(),
    ))
    res = await _service(db_repos, tmp_path).get_positions()
    assert res["count"] == 1
    assert res["positions"][0]["symbol"] == "F"
    assert res["positions"][0]["state"] == "CSP_OPEN"


@pytest.mark.asyncio
async def test_get_account_risk_reports_cap(db_repos, tmp_path):
    res = await _service(db_repos, tmp_path, cash=20_000).get_account_risk()
    assert res["concurrent_cap"]["limit"] == 4
    assert res["concurrent_cap"]["used"] == 0
    assert "equity" in res and "net_position_value" in res


async def _seed_regime(db_repos) -> None:
    await db_repos.regime.insert(RegimeSnapshot(
        snapshot_date=date(2026, 6, 24),
        regime=Regime.BULL_TREND,
        csps_allowed=True,
        bear_calls_allowed=False,
        spy_above_sma=True,
        vix_close=17.3,
    ))


@pytest.mark.asyncio
async def test_get_regime_and_calendar_reads_snapshot(db_repos, tmp_path):
    # Regression: the service read `self._repos.regime_snapshots` (no such attr
    # on Repos — it's `.regime`), so this tool raised AttributeError in prod.
    await _seed_regime(db_repos)
    res = await _service(db_repos, tmp_path).get_regime_and_calendar(days=30)
    assert res["regime"] is not None
    assert res["regime"]["regime"] == "BULL_TREND"
    assert res["regime"]["csps_allowed"] is True
    assert res["regime"]["bear_calls_allowed"] is False
    assert "upcoming_events" in res


@pytest.mark.asyncio
async def test_diagnose_symbol_reads_regime(db_repos, tmp_path):
    # Regression: same `self._repos.regime_snapshots` bug also broke this tool.
    await _seed_regime(db_repos)
    res = await _service(db_repos, tmp_path).diagnose_symbol("spy")
    assert res["symbol"] == "SPY"
    assert res["regime"] is not None
    assert res["regime"]["regime"] == "BULL_TREND"
