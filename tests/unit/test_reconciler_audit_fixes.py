"""2026-07-01 audit HIGH fixes: PMCC_CLOSING cancel-restore and per-(symbol,
strategy) position diffing (the last-wins-per-symbol collapse)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from core.models import (
    Order,
    OrderStatus,
    OrderType,
    OptionType,
    Position,
    PositionState,
    WheelCycle,
)
from execution.reconciler import Reconciler
from platforms.paper_broker import PaperBroker


def _config() -> dict:
    return {"account": {"id": "test", "broker": "paper"}}


def _utc() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


# -- fix 1: PMCC_CLOSING strand ----------------------------------------------


@pytest.mark.asyncio
async def test_pmcc_closing_cancel_restores_long_open(db_repos):
    """A PMCC long-close (SELL_TO_CLOSE) cancelled at the broker (e.g. EOD DAY
    -order cancel) must restore PMCC_CLOSING -> PMCC_LONG_OPEN with the cycle
    intact, so the close orchestrator re-proposes — instead of stranding the
    LEAP unmanaged forever."""
    broker = PaperBroker(cash=20_000)
    cycle_id = await db_repos.cycles.insert(WheelCycle(
        account_id="test", symbol="BAC", strategy_id="pmcc", started_at=_utc(),
    ))
    close = Order(
        account_id="test", symbol="BAC", strategy_id="pmcc",
        order_type=OrderType.SELL_TO_CLOSE, contract_symbol="BAC261218C00050000",
        strike=50.0, expiration=date(2026, 12, 18), option_type=OptionType.CALL,
        quantity=1, limit_price=9.10, status=OrderStatus.PENDING,
        placed_at=_utc(), client_order_id="wb-pmcc-close-1", cycle_id=cycle_id,
    )
    bo = await broker.place_order(close)
    await db_repos.orders.insert(close.model_copy(update={"broker_order_id": bo.broker_order_id}))
    pos_id = await db_repos.positions.insert(Position(
        account_id="test", symbol="BAC", strategy_id="pmcc",
        state=PositionState.PMCC_CLOSING, shares=0, current_cycle_id=cycle_id,
        state_changed_at=_utc(), state_change_reason="router_pending:wb-pmcc-close-1",
    ))

    await broker.cancel_order(bo.broker_order_id)
    rec = Reconciler(broker, db_repos, _config())
    summary = await rec.reconcile_once()

    assert summary.cancellations_processed == 1
    p = await db_repos.positions.get(pos_id)
    assert p.state == PositionState.PMCC_LONG_OPEN
    assert p.current_cycle_id == cycle_id  # cycle survives — long still held


# -- fix 2: per-(symbol, strategy) diffing ------------------------------------


async def _seed_filled_csp(db_repos, broker, *, strategy_id: str, occ: str,
                           strike: float, coid: str):
    """Router-pending seed + broker fill for one strategy's CSP on F."""
    o = Order(
        account_id="test", symbol="F", strategy_id=strategy_id,
        order_type=OrderType.SELL_TO_OPEN, contract_symbol=occ, strike=strike,
        expiration=date(2026, 8, 21), option_type=OptionType.PUT, quantity=1,
        limit_price=0.50, status=OrderStatus.PENDING, placed_at=_utc(),
        client_order_id=coid,
    )
    bo = await broker.place_order(o)
    await db_repos.orders.insert(o.model_copy(update={"broker_order_id": bo.broker_order_id}))
    await db_repos.positions.insert(Position(
        account_id="test", symbol="F", strategy_id=strategy_id,
        state=PositionState.CSP_PENDING, shares=0, state_changed_at=_utc(),
        state_change_reason=f"router_pending:{coid}",
    ))
    await broker.fill_order(bo.broker_order_id, fill_price=0.50)


@pytest.mark.asyncio
async def test_diff_covers_every_strategy_position_on_a_symbol(db_repos):
    """Two strategies hold CSPs on the SAME symbol; both shorts expire. The old
    last-wins-per-symbol dict diffed only ONE of them — the other's expiration
    was never inferred. Both must return to IDLE with both cycles closed."""
    broker = PaperBroker(cash=50_000)
    await _seed_filled_csp(db_repos, broker, strategy_id="monthly_wheel",
                           occ="F260821P00010000", strike=10.0, coid="wb-a-1")
    await _seed_filled_csp(db_repos, broker, strategy_id="weekly_wheel",
                           occ="F260821P00009000", strike=9.0, coid="wb-b-1")
    rec = Reconciler(broker, db_repos, _config())
    await rec.reconcile_once()  # both fills -> both CSP_OPEN, 2 cycles

    a = await db_repos.positions.get_by_symbol("test", "F", strategy_id="monthly_wheel")
    b = await db_repos.positions.get_by_symbol("test", "F", strategy_id="weekly_wheel")
    assert a.state == PositionState.CSP_OPEN and b.state == PositionState.CSP_OPEN

    await broker.expire("F260821P00010000")
    await broker.expire("F260821P00009000")
    summary = await rec.reconcile_once()

    a = await db_repos.positions.get_by_symbol("test", "F", strategy_id="monthly_wheel")
    b = await db_repos.positions.get_by_symbol("test", "F", strategy_id="weekly_wheel")
    assert a.state == PositionState.IDLE, "first strategy's expiration missed"
    assert b.state == PositionState.IDLE, "second strategy's expiration missed"
    assert summary.expirations_processed == 2
    assert summary.cycles_closed == 2
