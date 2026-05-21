"""scripts/daily_summary — metric aggregation for the Discord daily post."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from core.models import (
    CycleOutcome,
    Position,
    PositionState,
    Regime,
    WheelCycle,
)
from scripts.daily_summary import compute_metrics


def _config() -> dict:
    return {"account": {"id": "test"}}


async def _insert_cycle(
    db_repos,
    *,
    strategy_id: str,
    final_pnl: float | None = None,
    ended_at: datetime | None = None,
    started_at: datetime | None = None,
    symbol: str = "F",
) -> int:
    started_at = started_at or (datetime.now(UTC).replace(tzinfo=None) - timedelta(days=10))
    return await db_repos.cycles.insert(
        WheelCycle(
            account_id="test",
            symbol=symbol,
            strategy_id=strategy_id,
            started_at=started_at,
            ended_at=ended_at,
            final_pnl=final_pnl,
            cycle_outcome=(
                CycleOutcome.SPREAD_CLOSED_PROFIT
                if (final_pnl or 0) > 0
                else CycleOutcome.SPREAD_MAX_LOSS
            ) if final_pnl is not None else None,
        )
    )


async def _insert_position(
    db_repos, *, symbol: str, strategy_id: str, state: PositionState
) -> None:
    now = datetime.now(UTC).replace(tzinfo=None)
    await db_repos.positions.insert(
        Position(
            account_id="test",
            symbol=symbol,
            strategy_id=strategy_id,
            state=state,
            shares=0,
            state_changed_at=now,
        )
    )


# -- empty-state baseline ---------------------------------------------------


@pytest.mark.asyncio
async def test_metrics_with_no_data_returns_safe_zeros(db_repos):
    now = datetime.now(UTC).replace(tzinfo=None)
    m = await compute_metrics(db_repos, _config(), now=now)
    assert m["today_pnl"] == "$+0.00"
    assert m["today_closed"] == 0
    assert m["today_opened"] == 0
    assert m["week_pnl"] == "$+0.00"
    assert m["week_win_rate"] == "—"
    assert m["open_total"] == 0
    assert m["open_by_strategy"] == "none"
    assert m["regime"] == "—"
    assert m["auto_disabled"] == "none"


# -- today's metrics --------------------------------------------------------


@pytest.mark.asyncio
async def test_today_pnl_aggregates_cycles_closed_today_only(db_repos):
    now = datetime.now(UTC).replace(tzinfo=None)
    today_start = datetime.combine(now.date(), datetime.min.time())
    # Two closed today, one closed yesterday (should be excluded from today's PnL).
    await _insert_cycle(db_repos, strategy_id="put_spread",
                         final_pnl=40.0, ended_at=today_start + timedelta(hours=2))
    await _insert_cycle(db_repos, strategy_id="put_spread",
                         final_pnl=-15.0, ended_at=today_start + timedelta(hours=5))
    await _insert_cycle(db_repos, strategy_id="put_spread",
                         final_pnl=999.0, ended_at=today_start - timedelta(hours=2))

    m = await compute_metrics(db_repos, _config(), now=now)
    assert m["today_closed"] == 2
    assert m["today_pnl"] == "$+25.00"  # 40 - 15


@pytest.mark.asyncio
async def test_today_opened_counts_started_today(db_repos):
    now = datetime.now(UTC).replace(tzinfo=None)
    today_start = datetime.combine(now.date(), datetime.min.time())
    await _insert_cycle(db_repos, strategy_id="put_spread",
                         started_at=today_start + timedelta(hours=1))
    await _insert_cycle(db_repos, strategy_id="monthly_wheel",
                         started_at=today_start + timedelta(hours=4))
    # Yesterday's start — excluded
    await _insert_cycle(db_repos, strategy_id="weekly_wheel",
                         started_at=today_start - timedelta(hours=2))

    m = await compute_metrics(db_repos, _config(), now=now)
    assert m["today_opened"] == 2


# -- week-to-date metrics ---------------------------------------------------


@pytest.mark.asyncio
async def test_week_pnl_and_win_rate(db_repos):
    now = datetime.now(UTC).replace(tzinfo=None)
    # 3 winners + 1 loser within the rolling 7 days. 1 cycle outside the window.
    for pnl, days_ago in [(50.0, 1), (30.0, 3), (-20.0, 5), (10.0, 6), (1000.0, 10)]:
        await _insert_cycle(
            db_repos, strategy_id="put_spread",
            final_pnl=pnl, ended_at=now - timedelta(days=days_ago),
        )
    m = await compute_metrics(db_repos, _config(), now=now)
    assert m["week_closed"] == 4
    assert m["week_pnl"] == "$+70.00"  # 50 + 30 - 20 + 10
    assert m["week_win_rate"] == "75%"  # 3 of 4


# -- open positions ---------------------------------------------------------


@pytest.mark.asyncio
async def test_open_positions_grouped_by_strategy_excludes_idle(db_repos):
    await _insert_position(db_repos, symbol="F", strategy_id="monthly_wheel",
                            state=PositionState.CSP_OPEN)
    await _insert_position(db_repos, symbol="BAC", strategy_id="monthly_wheel",
                            state=PositionState.CSP_OPEN)
    await _insert_position(db_repos, symbol="MSFT", strategy_id="put_spread",
                            state=PositionState.SPREAD_OPEN)
    # IDLE — should be excluded
    await _insert_position(db_repos, symbol="ZZZ", strategy_id="put_spread",
                            state=PositionState.IDLE)

    m = await compute_metrics(db_repos, _config(), now=datetime.now(UTC).replace(tzinfo=None))
    assert m["open_total"] == 3
    assert "monthly_wheel=2" in m["open_by_strategy"]
    assert "put_spread=1" in m["open_by_strategy"]


# -- regime row -------------------------------------------------------------


@pytest.mark.asyncio
async def test_regime_pulled_from_latest_snapshot(db_repos):
    from datetime import date as _date
    c = await db_repos.db.connect()
    await c.execute(
        "INSERT INTO regime_snapshots (snapshot_date, regime, csps_allowed, bear_calls_allowed) "
        "VALUES (?, ?, ?, ?)",
        (_date.today().isoformat(), Regime.BEAR_TREND.value, 0, 1),
    )
    await c.commit()

    m = await compute_metrics(db_repos, _config(), now=datetime.now(UTC).replace(tzinfo=None))
    assert m["regime"] == "BEAR_TREND"
    assert m["csps_allowed"] == "✗"
    assert m["bear_calls_allowed"] == "✓"


# -- auto-disabled inclusion ------------------------------------------------


@pytest.mark.asyncio
async def test_auto_disabled_strategies_listed(db_repos):
    now = datetime.now(UTC).replace(tzinfo=None)
    await db_repos.strategy_runtime.disable(
        "put_spread",
        until=now + timedelta(days=10),
        reason="rolling drawdown -$400",
        now=now,
    )
    m = await compute_metrics(db_repos, _config(), now=now)
    assert "put_spread" in m["auto_disabled"]
