"""Swing lifecycle in the reconciler (sub-sprint 2.2b): the single deep-ITM long
goes IDLE → SWING_PENDING → SWING_OPEN (BUY_TO_OPEN fill) → IDLE (SELL_TO_CLOSE
fill), opening and closing exactly one cycle."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from core.models import (
    Order,
    OrderStatus,
    OrderType,
    OptionType,
    Position,
    PositionState,
)
from execution.reconciler import Reconciler
from platforms.paper_broker import PaperBroker

_OCC = "SPY260821C00590000"


def _swing_config() -> dict:
    return {
        "account": {"id": "test", "broker": "paper"},
        "strategies": [
            {"id": "spy_swing_opt", "type": "swing", "enabled": True,
             "max_concurrent": 1, "params": {"symbol": "SPY"}},
        ],
    }


def _utc() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _swing_order(order_type: OrderType, coid: str, price: float, **extra) -> Order:
    return Order(
        account_id="test", symbol="SPY", order_type=order_type,
        contract_symbol=_OCC, strike=590.0, expiration=date(2026, 8, 21),
        option_type=OptionType.CALL, quantity=1, limit_price=price,
        status=OrderStatus.PENDING, placed_at=_utc(), client_order_id=coid,
        strategy_id="spy_swing_opt", **extra,
    )


async def _seed_pending(db_repos, broker: PaperBroker):
    placed = _swing_order(OrderType.BUY_TO_OPEN, "wb-swing-1", 20.0)
    bo = await broker.place_order(placed)
    await db_repos.orders.insert(placed.model_copy(update={"broker_order_id": bo.broker_order_id}))
    pos_id = await db_repos.positions.insert(Position(
        account_id="test", symbol="SPY", strategy_id="spy_swing_opt",
        state=PositionState.SWING_PENDING, shares=0, state_changed_at=_utc(),
        state_change_reason="router_pending:wb-swing-1",
    ))
    return await db_repos.positions.get(pos_id), bo


@pytest.mark.asyncio
async def test_swing_buy_fill_opens_position_and_cycle(db_repos):
    broker = PaperBroker(cash=20_000)
    _pos, bo = await _seed_pending(db_repos, broker)
    await broker.fill_order(bo.broker_order_id, fill_price=20.0)

    rec = Reconciler(broker, db_repos, _swing_config())
    summary = await rec.reconcile_once()

    assert summary.fills_processed == 1
    assert summary.cycles_opened == 1
    p = await db_repos.positions.get_by_symbol("test", "SPY", strategy_id="spy_swing_opt")
    assert p is not None and p.state == PositionState.SWING_OPEN
    assert p.current_cycle_id is not None


@pytest.mark.asyncio
async def test_swing_sell_to_close_returns_to_idle_and_closes_cycle(db_repos):
    broker = PaperBroker(cash=20_000)
    _pos, bo = await _seed_pending(db_repos, broker)
    await broker.fill_order(bo.broker_order_id, fill_price=20.0)
    rec = Reconciler(broker, db_repos, _swing_config())
    await rec.reconcile_once()  # → SWING_OPEN

    close = _swing_order(OrderType.SELL_TO_CLOSE, "wb-swing-close-1", 24.0,
                         trigger_reason="swing_target")
    cbo = await broker.place_order(close)
    await db_repos.orders.insert(close.model_copy(update={"broker_order_id": cbo.broker_order_id}))
    await broker.fill_order(cbo.broker_order_id, fill_price=24.0)

    summary = await rec.reconcile_once()
    p = await db_repos.positions.get_by_symbol("test", "SPY", strategy_id="spy_swing_opt")
    assert p is not None and p.state == PositionState.IDLE
    assert p.current_cycle_id is None
    assert summary.cycles_closed == 1
    assert len(await db_repos.cycles.list_open("test")) == 0
