"""2026-08-26 audit — cycle/position parity phase.

The June-23 incident: nine wheel_cycles sat "open" for months after their
positions went IDLE (close handler / earnings flag_manual) while the broker
exposure lived on and was liquidated with no DB record. ~$3.8k of realized
P&L was invisible until a manual backfill. `_audit_cycle_parity` asserts the
invariants that would have caught it on the next tick:

  (a) open cycle older than min-age -> position must be in an
      exposure-bearing state;
  (b) rows carrying shares are in share-holding states and per-symbol share
      totals agree with the broker.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from core.models import Position, PositionState, WheelCycle
from execution.reconciler import Reconciler
from platforms.paper_broker import PaperBroker


def _utc(dt: datetime | None = None) -> datetime:
    return (dt or datetime.now(UTC)).replace(tzinfo=None)


def _config(**reconciler_overrides) -> dict:
    cfg: dict = {"account": {"id": "test", "broker": "paper"}}
    if reconciler_overrides:
        cfg["reconciler"] = reconciler_overrides
    return cfg


async def _insert_position(db_repos, symbol, strategy, state, shares=0):
    return await db_repos.positions.insert(
        Position(
            account_id="test",
            symbol=symbol,
            strategy_id=strategy,
            state=state,
            shares=shares,
            state_changed_at=_utc(),
            state_change_reason="test setup",
        )
    )


async def _insert_open_cycle(db_repos, symbol, strategy, *, days_old=30):
    return await db_repos.cycles.insert(
        WheelCycle(
            account_id="test",
            symbol=symbol,
            strategy_id=strategy,
            started_at=_utc() - timedelta(days=days_old),
            initial_capital_at_risk=10_000.0,
        )
    )


@pytest.mark.asyncio
async def test_orphaned_open_cycle_flags_manual_intervention(db_repos):
    """The COIN/PLTR signature: open cycle, position IDLE, broker empty."""
    await _insert_position(db_repos, "COIN", "weekly_wheel", PositionState.IDLE)
    cycle_id = await _insert_open_cycle(db_repos, "COIN", "weekly_wheel")

    rec = Reconciler(PaperBroker(cash=100_000), db_repos, _config())
    summary = await rec.reconcile_once()

    assert summary.parity_flags >= 1
    pos = await db_repos.positions.get_by_symbol(
        "test", "COIN", strategy_id="weekly_wheel"
    )
    assert pos.state == PositionState.MANUAL_INTERVENTION
    assert f"cycle {cycle_id}" in (pos.state_change_reason or "")
    assert "orphaned" in (pos.state_change_reason or "")


@pytest.mark.asyncio
async def test_fresh_open_cycle_is_left_alone(db_repos):
    """Young cycles churn states legitimately between legs — no flag."""
    await _insert_position(db_repos, "F", "monthly_wheel", PositionState.IDLE)
    await _insert_open_cycle(db_repos, "F", "monthly_wheel", days_old=1)

    rec = Reconciler(PaperBroker(cash=100_000), db_repos, _config())
    summary = await rec.reconcile_once()

    assert summary.parity_flags == 0
    pos = await db_repos.positions.get_by_symbol(
        "test", "F", strategy_id="monthly_wheel"
    )
    assert pos.state == PositionState.IDLE


@pytest.mark.asyncio
async def test_open_cycle_with_live_exposure_is_left_alone(db_repos):
    """SHARES_HELD position whose broker stock matches — healthy wheel."""
    broker = PaperBroker(cash=100_000)
    broker._stock["INTC"] = (100, 95.0)
    await _insert_position(
        db_repos, "INTC", "monthly_wheel", PositionState.SHARES_HELD, shares=100
    )
    await _insert_open_cycle(db_repos, "INTC", "monthly_wheel")

    rec = Reconciler(broker, db_repos, _config())
    summary = await rec.reconcile_once()

    assert summary.parity_flags == 0
    pos = await db_repos.positions.get_by_symbol(
        "test", "INTC", strategy_id="monthly_wheel"
    )
    assert pos.state == PositionState.SHARES_HELD


@pytest.mark.asyncio
async def test_idle_row_with_shares_flags(db_repos):
    """The COIN ghost-shares signature: IDLE row still holding 100 shares."""
    await _insert_position(
        db_repos, "COIN", "weekly_wheel", PositionState.IDLE, shares=100
    )

    rec = Reconciler(PaperBroker(cash=100_000), db_repos, _config())
    summary = await rec.reconcile_once()

    assert summary.parity_flags >= 1
    pos = await db_repos.positions.get_by_symbol(
        "test", "COIN", strategy_id="weekly_wheel"
    )
    assert pos.state == PositionState.MANUAL_INTERVENTION
    assert "untracked stock" in (pos.state_change_reason or "")


@pytest.mark.asyncio
async def test_share_count_mismatch_flags(db_repos):
    """DB says SHARES_HELD 100, broker holds none — divergence, not a guess."""
    await _insert_position(
        db_repos, "PLTR", "weekly_wheel", PositionState.SHARES_HELD, shares=100
    )

    rec = Reconciler(PaperBroker(cash=100_000), db_repos, _config())
    summary = await rec.reconcile_once()

    assert summary.parity_flags >= 1
    pos = await db_repos.positions.get_by_symbol(
        "test", "PLTR", strategy_id="weekly_wheel"
    )
    assert pos.state == PositionState.MANUAL_INTERVENTION
    assert "share count mismatch" in (pos.state_change_reason or "")


@pytest.mark.asyncio
async def test_parity_audit_can_be_disabled(db_repos):
    """Config kill-switch for the phase — orphaned cycle stays unflagged."""
    await _insert_position(db_repos, "COIN", "weekly_wheel", PositionState.IDLE)
    await _insert_open_cycle(db_repos, "COIN", "weekly_wheel")

    rec = Reconciler(
        PaperBroker(cash=100_000), db_repos, _config(cycle_parity_enabled=False)
    )
    summary = await rec.reconcile_once()

    assert summary.parity_flags == 0
    pos = await db_repos.positions.get_by_symbol(
        "test", "COIN", strategy_id="weekly_wheel"
    )
    assert pos.state == PositionState.IDLE


@pytest.mark.asyncio
async def test_operator_owned_states_not_reflagged(db_repos):
    """A cycle whose position is already MANUAL_INTERVENTION is in front of a
    human — the audit must not spam re-flags (Discord dedup contract)."""
    await _insert_position(
        db_repos, "META", "put_spread", PositionState.MANUAL_INTERVENTION
    )
    await _insert_open_cycle(db_repos, "META", "put_spread")

    rec = Reconciler(PaperBroker(cash=100_000), db_repos, _config())
    summary = await rec.reconcile_once()

    assert summary.parity_flags == 0
