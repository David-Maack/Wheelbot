"""Spec §13 #29 — broker-vs-local mismatches → MANUAL_INTERVENTION.

The spec is explicit: the reconciler must NEVER auto-correct a divergence
between broker truth and our DB. It flags MANUAL_INTERVENTION and waits for
a human. Some of these scenarios overlap with `test_reconciler.py`; that's
deliberate — this file is the dedicated gate to live trading.
"""

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
)
from execution.reconciler import Reconciler
from platforms.paper_broker import PaperBroker


def _utc(dt: datetime | None = None) -> datetime:
    return (dt or datetime.now(UTC)).replace(tzinfo=None)


def _config() -> dict:
    return {"account": {"id": "test", "broker": "paper"}}


@pytest.mark.asyncio
async def test_unknown_symbol_at_broker_flags_manual_intervention(db_repos):
    """Broker shows shares for a symbol we don't track at all."""
    broker = PaperBroker(cash=20_000)
    broker._stock["XYZ"] = (250, 12.34)

    rec = Reconciler(broker, db_repos, _config())
    summary = await rec.reconcile_once()

    assert summary.manual_interventions >= 1
    pos = await db_repos.positions.get_by_symbol("test", "XYZ")
    assert pos is not None
    assert pos.state == PositionState.MANUAL_INTERVENTION
    assert "broker shows" in (pos.state_change_reason or "")


@pytest.mark.asyncio
async def test_unknown_order_id_flags_position(db_repos):
    """Broker filled an order with a client_order_id we don't know about."""
    broker = PaperBroker(cash=20_000)
    rogue = Order(
        account_id="test",
        symbol="F",
        order_type=OrderType.SELL_TO_OPEN,
        contract_symbol="F250706P00009500",
        strike=9.5,
        expiration=date(2025, 7, 6),
        option_type=OptionType.PUT,
        quantity=1,
        limit_price=0.50,
        status=OrderStatus.PENDING,
        placed_at=_utc(),
        client_order_id="not-from-our-router",
    )
    placed = await broker.place_order(rogue)
    await broker.fill_order(placed.broker_order_id, fill_price=0.50)

    rec = Reconciler(broker, db_repos, _config())
    summary = await rec.reconcile_once()

    assert summary.manual_interventions >= 1
    pos = await db_repos.positions.get_by_symbol("test", "F")
    assert pos is not None
    assert pos.state == PositionState.MANUAL_INTERVENTION


@pytest.mark.asyncio
async def test_local_shares_held_broker_zero_does_not_silently_revert(db_repos):
    """Local says SHARES_HELD; broker shows zero. Reconciler must NOT silently
    decide we sold them — without a corresponding order trail it's a mismatch."""
    broker = PaperBroker(cash=20_000)
    now = _utc()
    await db_repos.positions.insert(
        Position(
            account_id="test",
            symbol="F",
            state=PositionState.SHARES_HELD,
            shares=100,
            cost_basis=9.0,
            state_changed_at=now,
        )
    )
    rec = Reconciler(broker, db_repos, _config())
    summary = await rec.reconcile_once()

    pos = await db_repos.positions.get_by_symbol("test", "F")
    assert pos is not None
    # The position must NOT have been silently set to IDLE.
    assert pos.state != PositionState.IDLE


@pytest.mark.asyncio
async def test_manual_intervention_is_sticky_across_reconciles(db_repos):
    """Once flagged, repeat reconcile passes don't silently un-flag."""
    broker = PaperBroker(cash=20_000)
    broker._stock["GHOST"] = (100, 5.0)

    rec = Reconciler(broker, db_repos, _config())
    await rec.reconcile_once()
    pos1 = await db_repos.positions.get_by_symbol("test", "GHOST")
    assert pos1.state == PositionState.MANUAL_INTERVENTION

    # Even after the broker's view stays the same, position stays MANUAL_INTERVENTION.
    await rec.reconcile_once()
    pos2 = await db_repos.positions.get_by_symbol("test", "GHOST")
    assert pos2.state == PositionState.MANUAL_INTERVENTION


@pytest.mark.asyncio
async def test_only_affected_position_is_flagged(db_repos):
    """A mismatch on one symbol must not corrupt other healthy positions."""
    broker = PaperBroker(cash=20_000)
    broker._stock["GHOST"] = (100, 5.0)

    now = _utc()
    await db_repos.positions.insert(
        Position(
            account_id="test",
            symbol="HEALTHY",
            state=PositionState.IDLE,
            shares=0,
            state_changed_at=now,
        )
    )

    rec = Reconciler(broker, db_repos, _config())
    await rec.reconcile_once()

    healthy = await db_repos.positions.get_by_symbol("test", "HEALTHY")
    assert healthy.state == PositionState.IDLE
    ghost = await db_repos.positions.get_by_symbol("test", "GHOST")
    assert ghost.state == PositionState.MANUAL_INTERVENTION


@pytest.mark.asyncio
async def test_unknown_broker_position_is_recorded_with_reason(db_repos):
    """The state_change_reason must explain WHY the position was flagged so
    the operator has something to act on."""
    broker = PaperBroker(cash=20_000)
    broker._stock["MYSTERY"] = (50, 7.0)

    rec = Reconciler(broker, db_repos, _config())
    await rec.reconcile_once()

    pos = await db_repos.positions.get_by_symbol("test", "MYSTERY")
    assert pos is not None
    assert pos.state == PositionState.MANUAL_INTERVENTION
    reason = pos.state_change_reason or ""
    assert "broker" in reason.lower() or "MYSTERY" in reason
