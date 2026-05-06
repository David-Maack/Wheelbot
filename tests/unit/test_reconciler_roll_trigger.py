"""Reconciler roll-trigger scan integration."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from core.models import (
    OptionContract,
    OptionType,
    Order,
    OrderStatus,
    OrderType,
    Position,
    PositionState,
    Quote,
    WheelCycle,
)
from execution.reconciler import Reconciler
from platforms.paper_broker import PaperBroker


def _utc():
    return datetime.now(UTC).replace(tzinfo=None)


async def _seed_csp_open(db_repos, broker: PaperBroker) -> int:
    """Seed an open CSP_OPEN locally + an actual open short leg at the broker
    so the reconciler's position diff doesn't interpret it as expired."""
    cid = await db_repos.cycles.insert(
        WheelCycle(account_id="test", symbol="F", started_at=_utc())
    )
    occ = "F250706P00009500"
    placed = await broker.place_order(
        Order(
            account_id="test", symbol="F",
            order_type=OrderType.SELL_TO_OPEN,
            contract_symbol=occ,
            strike=9.5, expiration=date.today() + timedelta(days=14),
            option_type=OptionType.PUT,
            quantity=1, limit_price=0.50,
            status=OrderStatus.PENDING,
            placed_at=_utc(),
            client_order_id="wb-it-1",
        )
    )
    await broker.fill_order(placed.broker_order_id, fill_price=0.50)
    # Persist a matching local order with cycle_id + FILLED.
    await db_repos.orders.insert(
        Order(
            account_id="test", symbol="F",
            cycle_id=cid,
            order_type=OrderType.SELL_TO_OPEN,
            contract_symbol=occ,
            strike=9.5, expiration=date.today() + timedelta(days=14),
            option_type=OptionType.PUT,
            quantity=1, limit_price=0.50, fill_price=0.50,
            status=OrderStatus.FILLED,
            placed_at=_utc(),
            client_order_id="wb-it-1",
            broker_order_id=placed.broker_order_id,
        )
    )
    pos_id = await db_repos.positions.insert(
        Position(
            account_id="test", symbol="F",
            state=PositionState.CSP_OPEN,
            shares=0,
            current_cycle_id=cid,
            state_changed_at=_utc(),
        )
    )
    broker.seed_quote(Quote(symbol=occ, bid=1.40, ask=1.60))
    return pos_id


@pytest.mark.asyncio
async def test_roll_evaluator_called_for_open_short(db_repos):
    broker = PaperBroker(cash=20_000)
    await _seed_csp_open(db_repos, broker)

    calls: list[tuple[str, str]] = []

    async def stub(position, short, mid):
        calls.append((position.symbol, short.contract_symbol or ""))
        from types import SimpleNamespace
        return SimpleNamespace(action="ROLL")

    rec = Reconciler(broker, db_repos, {"account": {"id": "test"}}, roll_evaluator=stub)
    summary = await rec.reconcile_once()
    assert summary.rolls_evaluated == 1
    assert calls == [("F", "F250706P00009500")]


@pytest.mark.asyncio
async def test_roll_evaluator_skips_below_trigger(db_repos):
    """Stub returns action=None — reconciler shouldn't increment rolls_evaluated."""
    broker = PaperBroker(cash=20_000)
    await _seed_csp_open(db_repos, broker)

    async def stub(position, short, mid):
        from types import SimpleNamespace
        return SimpleNamespace(action=None)

    rec = Reconciler(broker, db_repos, {"account": {"id": "test"}}, roll_evaluator=stub)
    summary = await rec.reconcile_once()
    assert summary.rolls_evaluated == 0


@pytest.mark.asyncio
async def test_no_evaluator_means_no_roll_scan(db_repos):
    broker = PaperBroker(cash=20_000)
    await _seed_csp_open(db_repos, broker)

    rec = Reconciler(broker, db_repos, {"account": {"id": "test"}}, roll_evaluator=None)
    summary = await rec.reconcile_once()
    assert summary.rolls_evaluated == 0
