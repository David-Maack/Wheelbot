"""risk/expectancy_floor — per-strategy expectancy gate (2026-07-06 sprint).

Companion suite to test_win_rate_floor.py; also covers the pause-column
sharing contract between the two floors (neither clobbers the other).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from core.models import CycleOutcome, WheelCycle
from core.strategies import StrategyDefinition
from risk import win_rate_floor
from risk.expectancy_floor import (
    ExpectancyState,
    check_and_apply,
    current_pause_state,
    evaluate,
    get_expectancy_overview,
)


async def _no_op_notify(*args, **kwargs):
    return None


def _strategy(
    *,
    id_: str = "put_spread",
    expectancy_override: float | None = None,
) -> StrategyDefinition:
    params = {}
    if expectancy_override is not None:
        params["expectancy_floor_usd"] = expectancy_override
    return StrategyDefinition(
        id=id_, display_name="Test", type="vertical_spread",
        enabled=True, max_concurrent=4, params=params,
    )


def _cfg(
    *,
    enabled: bool = True,
    min_cycles: int = 10,
    min_expectancy: float = 0.0,
    win_rate_enabled: bool = False,
) -> dict:
    return {
        "account": {"id": "test"},
        "risk": {
            "expectancy_floor": {
                "enabled": enabled,
                "min_closed_cycles": min_cycles,
                "min_expectancy_usd": min_expectancy,
                "pause_duration_days": 14,
            },
            "win_rate_floor": {
                "enabled": win_rate_enabled,
                "min_closed_cycles": min_cycles,
                "min_win_rate_pct": 60,
            },
        },
    }


async def _insert_cycles(db_repos, *, strategy_id: str, pnls: list[float]) -> None:
    base = datetime.now(UTC).replace(tzinfo=None)
    for i, pnl in enumerate(pnls):
        ended = base - timedelta(days=i)
        await db_repos.cycles.insert(
            WheelCycle(
                account_id="test", symbol="F", strategy_id=strategy_id,
                started_at=ended - timedelta(days=10), ended_at=ended,
                final_pnl=pnl,
                cycle_outcome=CycleOutcome.SPREAD_CLOSED_PROFIT
                if pnl > 0 else CycleOutcome.SPREAD_MAX_LOSS,
            )
        )


# The live motivating case: 6 wins of $54 vs 4 stops of -$150 = 60% win rate
# (passes the win-rate floor) but -$27.60/cycle expectancy.
WINS_OFTEN_LOSES_BIG = [54.0, 54.0, 54.0, 54.0, 54.0, 54.0, -150.0, -150.0, -150.0, -150.0]
# Profitable but lumpy: 5 wins of $120 vs 5 losses of -$40 = 50% win rate
# (fails the win-rate floor) yet +$40/cycle expectancy.
LOSES_OFTEN_WINS_BIG = [120.0, -40.0, 120.0, -40.0, 120.0, -40.0, 120.0, -40.0, 120.0, -40.0]


# -- evaluate --------------------------------------------------------------


@pytest.mark.asyncio
async def test_insufficient_data(db_repos):
    await _insert_cycles(db_repos, strategy_id="put_spread", pnls=[100, -50, 80])
    result = await evaluate(db_repos, "put_spread", _cfg(), account_id="test")
    assert result.action == "insufficient_data"
    assert result.expectancy is None
    assert result.sample_size == 3


@pytest.mark.asyncio
async def test_evaluate_mean_math(db_repos):
    await _insert_cycles(db_repos, strategy_id="put_spread", pnls=WINS_OFTEN_LOSES_BIG)
    result = await evaluate(db_repos, "put_spread", _cfg(), account_id="test")
    assert result.expectancy == pytest.approx((54 * 6 - 150 * 4) / 10)  # -27.60


@pytest.mark.asyncio
async def test_none_final_pnl_cycle_excluded_by_repo(db_repos):
    """A closed cycle with NULL final_pnl doesn't count toward the sample —
    list_closed_for_strategy filters it (same contract as the win-rate floor
    and drawdown breaker), so a partially-recorded close can't bias the mean."""
    pnls = [100.0] * 9
    await _insert_cycles(db_repos, strategy_id="put_spread", pnls=pnls)
    base = datetime.now(UTC).replace(tzinfo=None)
    await db_repos.cycles.insert(
        WheelCycle(account_id="test", symbol="F", strategy_id="put_spread",
                   started_at=base - timedelta(days=30), ended_at=base - timedelta(days=20),
                   final_pnl=None, cycle_outcome=CycleOutcome.SPREAD_CLOSED_PROFIT)
    )
    result = await evaluate(db_repos, "put_spread", _cfg(), account_id="test")
    assert result.sample_size == 9
    assert result.action == "insufficient_data"


# -- check_and_apply: the blind-spot cases -----------------------------------


@pytest.mark.asyncio
async def test_wins_often_loses_big_pauses(db_repos, monkeypatch):
    """60% win rate passes the win-rate floor; -$27.60/cycle trips this one."""
    monkeypatch.setattr("risk.expectancy_floor.notify", _no_op_notify)
    await _insert_cycles(db_repos, strategy_id="put_spread", pnls=WINS_OFTEN_LOSES_BIG)
    result = await check_and_apply(db_repos, _strategy(), _cfg())
    assert result.action == "pause"
    state = await current_pause_state(db_repos, "put_spread")
    assert state is ExpectancyState.PAUSED_NEGATIVE_EXPECTANCY
    row = await db_repos.strategy_runtime.get("put_spread")
    assert row["pause_state"] == "NEGATIVE_EXPECTANCY"
    assert "expectancy_floor" in row["paused_reason"]


@pytest.mark.asyncio
async def test_loses_often_wins_big_stays_normal(db_repos, monkeypatch):
    """50% win rate would fail the win-rate floor; +$40/cycle passes this one."""
    monkeypatch.setattr("risk.expectancy_floor.notify", _no_op_notify)
    await _insert_cycles(db_repos, strategy_id="put_spread", pnls=LOSES_OFTEN_WINS_BIG)
    result = await check_and_apply(db_repos, _strategy(), _cfg())
    assert result.action == "normal"
    assert await db_repos.strategy_runtime.get("put_spread") is None


@pytest.mark.asyncio
async def test_disabled_feature_is_noop(db_repos):
    await _insert_cycles(db_repos, strategy_id="put_spread", pnls=WINS_OFTEN_LOSES_BIG)
    result = await check_and_apply(db_repos, _strategy(), _cfg(enabled=False))
    assert result.action == "normal"
    assert await db_repos.strategy_runtime.get("put_spread") is None


@pytest.mark.asyncio
async def test_per_strategy_override(db_repos, monkeypatch):
    """Global floor $0 passes +$40/cycle, but a +$50 per-strategy override trips."""
    monkeypatch.setattr("risk.expectancy_floor.notify", _no_op_notify)
    await _insert_cycles(db_repos, strategy_id="put_spread", pnls=LOSES_OFTEN_WINS_BIG)
    result = await check_and_apply(
        db_repos, _strategy(expectancy_override=50.0), _cfg(),
    )
    assert result.action == "pause"


@pytest.mark.asyncio
async def test_recovery_holds_pause(db_repos, monkeypatch):
    """Expectancy recovering above the floor does NOT auto-clear the pause."""
    monkeypatch.setattr("risk.expectancy_floor.notify", _no_op_notify)
    await _insert_cycles(db_repos, strategy_id="put_spread", pnls=WINS_OFTEN_LOSES_BIG)
    await check_and_apply(db_repos, _strategy(), _cfg())
    # Ten fresh profitable cycles push the rolling mean positive.
    await _insert_cycles(db_repos, strategy_id="put_spread", pnls=[100.0] * 10)
    result = await check_and_apply(db_repos, _strategy(), _cfg())
    assert result.action == "pause"  # held
    assert (await current_pause_state(db_repos, "put_spread")) is (
        ExpectancyState.PAUSED_NEGATIVE_EXPECTANCY
    )


# -- pause-column sharing between the two floors -------------------------------


@pytest.mark.asyncio
async def test_expectancy_defers_to_existing_win_rate_pause(db_repos, monkeypatch):
    """Win-rate floor already owns the pause → expectancy floor holds, never
    overwrites the reason columns."""
    monkeypatch.setattr("risk.expectancy_floor.notify", _no_op_notify)
    await db_repos.strategy_runtime.mark_paused(
        "put_spread", reason="win_rate_floor: 50% < 60%", state="LOW_WIN_RATE",
    )
    await _insert_cycles(db_repos, strategy_id="put_spread", pnls=WINS_OFTEN_LOSES_BIG)
    result = await check_and_apply(db_repos, _strategy(), _cfg())
    assert result.action == "held_other_pause"
    row = await db_repos.strategy_runtime.get("put_spread")
    assert row["pause_state"] == "LOW_WIN_RATE"
    assert "win_rate_floor" in row["paused_reason"]


@pytest.mark.asyncio
async def test_win_rate_defers_to_existing_expectancy_pause(db_repos, monkeypatch):
    """The mirror case: expectancy floor owns the pause → win-rate floor holds."""
    monkeypatch.setattr("risk.expectancy_floor.notify", _no_op_notify)
    monkeypatch.setattr("risk.win_rate_floor.notify", _no_op_notify)
    # WINS_OFTEN_LOSES_BIG has a 60% win rate — set the win-rate floor at 70
    # so BOTH floors want to pause; expectancy runs first here.
    await _insert_cycles(db_repos, strategy_id="put_spread", pnls=WINS_OFTEN_LOSES_BIG)
    await check_and_apply(db_repos, _strategy(), _cfg())
    cfg = _cfg(win_rate_enabled=True)
    cfg["risk"]["win_rate_floor"]["min_win_rate_pct"] = 70
    await win_rate_floor.check_and_apply(db_repos, _strategy(), cfg)
    row = await db_repos.strategy_runtime.get("put_spread")
    assert row["pause_state"] == "NEGATIVE_EXPECTANCY"
    assert "expectancy_floor" in row["paused_reason"]


@pytest.mark.asyncio
async def test_list_paused_surfaces_both_literals(db_repos):
    await db_repos.strategy_runtime.mark_paused(
        "monthly_wheel", reason="win rate", state="LOW_WIN_RATE",
    )
    await db_repos.strategy_runtime.mark_paused(
        "put_spread", reason="expectancy", state="NEGATIVE_EXPECTANCY",
    )
    paused = await db_repos.strategy_runtime.list_paused()
    assert {r["strategy_id"] for r in paused} == {"monthly_wheel", "put_spread"}


# -- dashboard overview ----------------------------------------------------------


@pytest.mark.asyncio
async def test_overview_shows_both_truths(db_repos, monkeypatch):
    """A paused strategy with a recovered mean shows PAUSED state + the fresh
    (positive) expectancy."""
    monkeypatch.setattr("risk.expectancy_floor.notify", _no_op_notify)
    await _insert_cycles(db_repos, strategy_id="put_spread", pnls=WINS_OFTEN_LOSES_BIG)
    await check_and_apply(db_repos, _strategy(), _cfg())
    await _insert_cycles(db_repos, strategy_id="put_spread", pnls=[100.0] * 10)
    overview = await get_expectancy_overview(db_repos, [_strategy()], _cfg())
    info = overview["put_spread"]
    assert info["state"] == "NEGATIVE_EXPECTANCY"
    assert info["expectancy"] > 0
    assert info["threshold"] == 0.0
