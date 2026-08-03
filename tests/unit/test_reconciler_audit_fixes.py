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


# -- 2026-08-03 fix: PMCC *_PENDING stranded-state self-heal ------------------


async def _seed_pmcc_short_pending(
    db_repos, *, sto_status: OrderStatus
) -> tuple[int, int]:
    """PMCC holding its LEAP, short-call STO in the given terminal status, and
    the position stranded in PMCC_SHORT_PENDING — the lost-event shape that
    held CCL/F frozen for weeks live."""
    cycle_id = await db_repos.cycles.insert(WheelCycle(
        account_id="test", symbol="CCL", strategy_id="pmcc", started_at=_utc(),
    ))
    sto = Order(
        account_id="test", symbol="CCL", strategy_id="pmcc",
        order_type=OrderType.SELL_TO_OPEN, contract_symbol="CCL260807C00026000",
        strike=26.0, expiration=date(2026, 8, 7), option_type=OptionType.CALL,
        quantity=1, limit_price=0.55, status=sto_status,
        placed_at=_utc(), client_order_id="wb-pmcc-sto-1", cycle_id=cycle_id,
    )
    await db_repos.orders.insert(sto)
    pos_id = await db_repos.positions.insert(Position(
        account_id="test", symbol="CCL", strategy_id="pmcc",
        state=PositionState.PMCC_SHORT_PENDING, shares=0,
        current_cycle_id=cycle_id, state_changed_at=_utc(),
        state_change_reason="router_pending:wb-pmcc-sto-1",
    ))
    return pos_id, cycle_id


@pytest.mark.asyncio
async def test_pmcc_short_pending_selfheals_to_long_open_when_sto_died(db_repos):
    """2026-08-03 live incident: the short-call STO died (EOD cancel) but the
    cancel event was lost across a restart, so the position sat in
    PMCC_SHORT_PENDING for weeks — holding a global cap slot AND freezing
    short-call income on the LEAP. With no order in flight the reconciler
    restores PMCC_LONG_OPEN with the cycle intact."""
    pos_id, cycle_id = await _seed_pmcc_short_pending(
        db_repos, sto_status=OrderStatus.CANCELLED
    )
    rec = Reconciler(PaperBroker(cash=20_000), db_repos, _config())
    await rec.reconcile_once()
    p = await db_repos.positions.get(pos_id)
    assert p.state == PositionState.PMCC_LONG_OPEN
    assert p.current_cycle_id == cycle_id


@pytest.mark.asyncio
async def test_pmcc_short_pending_selfheals_to_both_open_when_sto_filled(db_repos):
    """Mirror case: the STO actually FILLED but the fill event was lost. The
    short is live at the broker, so the heal must land on PMCC_BOTH_OPEN."""
    pos_id, cycle_id = await _seed_pmcc_short_pending(
        db_repos, sto_status=OrderStatus.FILLED
    )
    rec = Reconciler(PaperBroker(cash=20_000), db_repos, _config())
    await rec.reconcile_once()
    p = await db_repos.positions.get(pos_id)
    assert p.state == PositionState.PMCC_BOTH_OPEN
    assert p.current_cycle_id == cycle_id


@pytest.mark.asyncio
async def test_pmcc_short_pending_heal_ignores_other_strategy_orders(db_repos):
    """A calendar's FILLED front short call on the SAME underlying must not
    contaminate the pmcc heal (live shape: CCL calendar + CCL pmcc). The heal
    reads only the pmcc strategy's own order stream."""
    pos_id, _cycle_id = await _seed_pmcc_short_pending(
        db_repos, sto_status=OrderStatus.CANCELLED
    )
    calendar_sto = Order(
        account_id="test", symbol="CCL", strategy_id="calendar",
        order_type=OrderType.SELL_TO_OPEN, contract_symbol="CCL260807C00027000",
        strike=27.0, expiration=date(2026, 8, 7), option_type=OptionType.CALL,
        quantity=1, limit_price=0.40, status=OrderStatus.FILLED,
        placed_at=_utc() + timedelta(minutes=1), client_order_id="wb-cal-sto-1",
    )
    await db_repos.orders.insert(calendar_sto)
    rec = Reconciler(PaperBroker(cash=20_000), db_repos, _config())
    await rec.reconcile_once()
    p = await db_repos.positions.get(pos_id)
    # The pmcc STO was CANCELLED -> LONG_OPEN, despite the calendar's FILLED
    # short call being the newest CALL order on the symbol.
    assert p.state == PositionState.PMCC_LONG_OPEN


@pytest.mark.asyncio
async def test_pmcc_long_pending_selfheals_to_idle_when_bto_died(db_repos):
    """LONG_PENDING with a dead BUY_TO_OPEN and nothing in flight: no LEAP was
    ever acquired, so the position returns to IDLE (no cycle to keep)."""
    bto = Order(
        account_id="test", symbol="F", strategy_id="pmcc",
        order_type=OrderType.BUY_TO_OPEN, contract_symbol="F261218C00011000",
        strike=11.0, expiration=date(2026, 12, 18), option_type=OptionType.CALL,
        quantity=1, limit_price=4.20, status=OrderStatus.CANCELLED,
        placed_at=_utc(), client_order_id="wb-pmcc-bto-1",
    )
    await db_repos.orders.insert(bto)
    pos_id = await db_repos.positions.insert(Position(
        account_id="test", symbol="F", strategy_id="pmcc",
        state=PositionState.PMCC_LONG_PENDING, shares=0, state_changed_at=_utc(),
        state_change_reason="router_pending:wb-pmcc-bto-1",
    ))
    rec = Reconciler(PaperBroker(cash=20_000), db_repos, _config())
    await rec.reconcile_once()
    p = await db_repos.positions.get(pos_id)
    assert p.state == PositionState.IDLE


@pytest.mark.asyncio
async def test_pmcc_long_pending_selfheals_to_long_open_on_lost_fill(db_repos):
    """LONG_PENDING whose BUY_TO_OPEN FILLED but the fill event was lost: the
    heal must both restore PMCC_LONG_OPEN and recreate the cycle the lost fill
    would have opened (debit-based capital at risk)."""
    bto = Order(
        account_id="test", symbol="F", strategy_id="pmcc",
        order_type=OrderType.BUY_TO_OPEN, contract_symbol="F261218C00011000",
        strike=11.0, expiration=date(2026, 12, 18), option_type=OptionType.CALL,
        quantity=1, limit_price=4.20, fill_price=4.15,
        status=OrderStatus.FILLED, placed_at=_utc(),
        client_order_id="wb-pmcc-bto-2",
    )
    await db_repos.orders.insert(bto)
    pos_id = await db_repos.positions.insert(Position(
        account_id="test", symbol="F", strategy_id="pmcc",
        state=PositionState.PMCC_LONG_PENDING, shares=0, state_changed_at=_utc(),
        state_change_reason="router_pending:wb-pmcc-bto-2",
    ))
    rec = Reconciler(PaperBroker(cash=20_000), db_repos, _config())
    await rec.reconcile_once()
    p = await db_repos.positions.get(pos_id)
    assert p.state == PositionState.PMCC_LONG_OPEN
    assert p.current_cycle_id is not None
    cycle = await db_repos.cycles.get(p.current_cycle_id)
    assert cycle.initial_capital_at_risk == pytest.approx(415.0)
