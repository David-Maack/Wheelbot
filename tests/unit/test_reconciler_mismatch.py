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
async def test_manual_intervention_does_not_renotify_when_already_flagged(db_repos, monkeypatch):
    """Repeat MANUAL_INTERVENTION on an already-flagged position must NOT
    send a duplicate Discord notification. The dedup check fires before
    the notify call, so the operator only gets pinged once per event."""
    notify_calls: list[tuple] = []
    async def _stub_notify(event_type, title, **payload):
        notify_calls.append((event_type, title, payload))
    monkeypatch.setattr("execution.reconciler.notify", _stub_notify)

    broker = PaperBroker(cash=20_000)
    broker._stock["GHOST"] = (100, 5.0)

    rec = Reconciler(broker, db_repos, _config())
    # First reconcile: flag + notify.
    await rec.reconcile_once()
    assert len(notify_calls) == 1

    # Second reconcile: broker still shows the same mismatch; position is
    # already MANUAL_INTERVENTION. The dedup must fire — no second ping.
    await rec.reconcile_once()
    assert len(notify_calls) == 1, "duplicate Discord notify on repeat MANUAL_INTERVENTION"


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


# -- Single-leg wheel cancellation handling (Sprint 14) ---------------------


@pytest.mark.asyncio
async def test_csp_pending_returns_to_idle_on_sell_to_open_cancel(db_repos):
    """Wheel CSP entry cancellation: CSP_PENDING → IDLE on the next reconcile.

    Regression for the 2026-05-21 MARA situation: bot tried to roll into a
    new weekly CSP after a successful profit-close; the SELL_TO_OPEN got
    cancelled at the broker, and without _on_cancel the position stuck at
    CSP_PENDING forever (couldn't propose new entries since the symbol was
    already 'pending')."""
    broker = PaperBroker(cash=20_000)
    rogue = Order(
        account_id="test",
        symbol="MARA",
        order_type=OrderType.SELL_TO_OPEN,
        contract_symbol="MARA260529P00011000",
        strike=11.0,
        expiration=date(2026, 5, 29),
        option_type=OptionType.PUT,
        quantity=1,
        limit_price=0.20,
        status=OrderStatus.PENDING,
        placed_at=_utc(),
        client_order_id="wb-csp-test",
    )
    placed = await broker.place_order(rogue)
    # Persist a local CSP_PENDING row that matches the in-flight order.
    await db_repos.orders.insert(
        rogue.model_copy(
            update={"broker_order_id": placed.broker_order_id, "status": OrderStatus.PENDING},
        )
    )
    await db_repos.positions.insert(
        Position(
            account_id="test",
            symbol="MARA",
            strategy_id="weekly_wheel",
            state=PositionState.CSP_PENDING,
            shares=0,
            state_changed_at=_utc(),
        )
    )

    # Broker cancels the order before fill.
    await broker.cancel_order(placed.broker_order_id)

    rec = Reconciler(broker, db_repos, _config())
    summary = await rec.reconcile_once()
    assert summary.cancellations_processed == 1
    pos = await db_repos.positions.get_by_symbol("test", "MARA")
    assert pos.state == PositionState.IDLE
    assert pos.current_cycle_id is None


@pytest.mark.asyncio
async def test_cc_pending_returns_to_shares_held_on_sell_to_open_cancel(db_repos):
    """Wheel CC entry cancellation: CC_PENDING → SHARES_HELD (still own underlying)."""
    broker = PaperBroker(cash=20_000)
    rogue = Order(
        account_id="test",
        symbol="F",
        order_type=OrderType.SELL_TO_OPEN,
        contract_symbol="F260620C00012000",
        strike=12.0,
        expiration=date(2026, 6, 20),
        option_type=OptionType.CALL,
        quantity=1,
        limit_price=0.30,
        status=OrderStatus.PENDING,
        placed_at=_utc(),
        client_order_id="wb-cc-test",
    )
    placed = await broker.place_order(rogue)
    await db_repos.orders.insert(
        rogue.model_copy(
            update={"broker_order_id": placed.broker_order_id, "status": OrderStatus.PENDING},
        )
    )
    await db_repos.positions.insert(
        Position(
            account_id="test",
            symbol="F",
            strategy_id="monthly_wheel",
            state=PositionState.CC_PENDING,
            shares=100,
            cost_basis=10.0,
            state_changed_at=_utc(),
        )
    )

    await broker.cancel_order(placed.broker_order_id)

    rec = Reconciler(broker, db_repos, _config())
    summary = await rec.reconcile_once()
    assert summary.cancellations_processed == 1
    pos = await db_repos.positions.get_by_symbol("test", "F")
    # We still own the shares; just no covered call sold against them yet.
    assert pos.state == PositionState.SHARES_HELD
    assert pos.shares == 100


@pytest.mark.asyncio
async def test_single_leg_cancel_does_not_clobber_manual_intervention(db_repos):
    """Defensive guard: position already at MANUAL_INTERVENTION must not be
    overwritten by a stale single-leg cancellation."""
    broker = PaperBroker(cash=20_000)
    rogue = Order(
        account_id="test",
        symbol="MARA",
        order_type=OrderType.SELL_TO_OPEN,
        contract_symbol="MARA260529P00011000",
        strike=11.0,
        expiration=date(2026, 5, 29),
        option_type=OptionType.PUT,
        quantity=1,
        limit_price=0.20,
        status=OrderStatus.PENDING,
        placed_at=_utc(),
        client_order_id="wb-csp-defensive",
    )
    placed = await broker.place_order(rogue)
    await db_repos.orders.insert(
        rogue.model_copy(
            update={"broker_order_id": placed.broker_order_id, "status": OrderStatus.PENDING},
        )
    )
    # Position has been flagged by another rule — MUST NOT be auto-reset.
    await db_repos.positions.insert(
        Position(
            account_id="test",
            symbol="MARA",
            strategy_id="weekly_wheel",
            state=PositionState.MANUAL_INTERVENTION,
            shares=0,
            state_changed_at=_utc(),
            state_change_reason="set by another rule",
        )
    )
    await broker.cancel_order(placed.broker_order_id)

    rec = Reconciler(broker, db_repos, _config())
    summary = await rec.reconcile_once()
    assert summary.cancellations_processed == 0
    pos = await db_repos.positions.get_by_symbol("test", "MARA")
    assert pos.state == PositionState.MANUAL_INTERVENTION  # untouched


@pytest.mark.asyncio
async def test_partial_fill_then_cancel_flags_manual_not_reset(db_repos):
    """Finding #7: a multi-contract order that partially fills and then cancels
    leaves REAL contracts live at the broker. _on_cancel would reset the
    position to IDLE, orphaning them. Instead the reconciler must flag
    MANUAL_INTERVENTION and leave the cycle/position for a human."""
    broker = PaperBroker(cash=20_000)
    rogue = Order(
        account_id="test",
        symbol="MARA",
        order_type=OrderType.SELL_TO_OPEN,
        contract_symbol="MARA260529P00011000",
        strike=11.0,
        expiration=date(2026, 5, 29),
        option_type=OptionType.PUT,
        quantity=3,                       # multi-contract → partial fill possible
        limit_price=0.20,
        status=OrderStatus.PENDING,
        placed_at=_utc(),
        client_order_id="wb-csp-partial",
    )
    placed = await broker.place_order(rogue)
    # We'd recorded a PARTIAL on a prior tick (1 of 3 contracts filled).
    await db_repos.orders.insert(
        rogue.model_copy(
            update={
                "broker_order_id": placed.broker_order_id,
                "status": OrderStatus.PARTIAL,
            },
        )
    )
    await db_repos.positions.insert(
        Position(
            account_id="test",
            symbol="MARA",
            strategy_id="weekly_wheel",
            state=PositionState.CSP_PENDING,
            shares=0,
            state_changed_at=_utc(),
        )
    )

    # Remaining quantity cancelled (e.g. EOD on a DAY order).
    await broker.cancel_order(placed.broker_order_id)

    rec = Reconciler(broker, db_repos, _config())
    summary = await rec.reconcile_once()

    # NOT treated as a clean cancel — no reset to IDLE.
    assert summary.cancellations_processed == 0
    assert summary.manual_interventions == 1
    pos = await db_repos.positions.get_by_symbol("test", "MARA")
    assert pos.state == PositionState.MANUAL_INTERVENTION


def test_observed_partial_fill_signals():
    """Unit-cover both detection signals of _observed_partial_fill."""
    from execution.reconciler import _observed_partial_fill

    def _order(**kw):
        base = dict(
            account_id="t", symbol="F", order_type=OrderType.SELL_TO_OPEN,
            quantity=3, status=OrderStatus.PENDING, placed_at=_utc(),
        )
        base.update(kw)
        return Order(**base)

    # 1. Persisted PARTIAL on the local row.
    assert _observed_partial_fill(
        _order(status=OrderStatus.PARTIAL), _order(status=OrderStatus.CANCELLED)
    ) is True

    # 2. Best-effort: raw payload shows a partial filled_qty below ordered qty.
    bv = _order(status=OrderStatus.CANCELLED, raw_response={"filled_qty": "1"})
    assert _observed_partial_fill(_order(status=OrderStatus.PENDING), bv) is True

    # Fully filled (filled_qty == quantity) is NOT a partial.
    bv_full = _order(status=OrderStatus.CANCELLED, raw_response={"filled_qty": "3"})
    assert _observed_partial_fill(_order(status=OrderStatus.PENDING), bv_full) is False

    # No fill at all → clean cancel.
    bv_zero = _order(status=OrderStatus.CANCELLED, raw_response={"filled_qty": "0"})
    assert _observed_partial_fill(_order(status=OrderStatus.PENDING), bv_zero) is False

    # Missing / malformed payload → no false positive.
    assert _observed_partial_fill(
        _order(status=OrderStatus.PENDING), _order(status=OrderStatus.CANCELLED)
    ) is False
    bv_bad = _order(status=OrderStatus.CANCELLED, raw_response={"filled_qty": "oops"})
    assert _observed_partial_fill(_order(status=OrderStatus.PENDING), bv_bad) is False
