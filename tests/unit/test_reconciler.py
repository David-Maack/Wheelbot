"""execution/reconciler — the most critical module in the bot.

Heavy coverage of the §10 transition table:
  fill        → CSP_PENDING/CC_PENDING → *_OPEN
  assignment  → CSP_OPEN  → SHARES_HELD with cost_basis = strike - premium
  expiration  → CSP_OPEN  → IDLE + cycle close
                CC_OPEN   → SHARES_HELD
  called away → CC_OPEN   → IDLE + cycle close + realized P&L
  mismatches  → MANUAL_INTERVENTION (never auto-correct)
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from core.broker import BrokerUnavailable
from core.models import (
    CycleOutcome,
    Order,
    OrderStatus,
    OrderType,
    OptionType,
    Position,
    PositionState,
    Quote,
    WheelCycle,
)
from execution.reconciler import ReconcileSummary, Reconciler
from platforms.paper_broker import PaperBroker


def _config() -> dict:
    return {"account": {"id": "test", "broker": "paper"}}


def _utc(dt: datetime | None = None) -> datetime:
    return (dt or datetime.now(UTC)).replace(tzinfo=None)


async def _seed_csp_pending(db_repos, broker: PaperBroker) -> tuple[Position, Order]:
    """Helper: simulate the router writing a CSP_PENDING row + a PENDING order, then have
    the broker accept that order (so it shows up in get_orders_since)."""
    placed_order = Order(
        account_id="test",
        symbol="F",
        order_type=OrderType.SELL_TO_OPEN,
        contract_symbol="F250706P00009500",
        strike=9.5,
        expiration=date(2025, 6, 1) + timedelta(days=35),
        option_type=OptionType.PUT,
        quantity=1,
        limit_price=0.50,
        status=OrderStatus.PENDING,
        placed_at=_utc(),
        client_order_id="wb-test-csp-1",
    )
    broker_order = await broker.place_order(placed_order)
    # Persist locally with the same client_order_id and broker_order_id.
    persisted = placed_order.model_copy(update={"broker_order_id": broker_order.broker_order_id})
    await db_repos.orders.insert(persisted)
    pos_id = await db_repos.positions.insert(
        Position(
            account_id="test",
            symbol="F",
            state=PositionState.CSP_PENDING,
            shares=0,
            state_changed_at=_utc(),
            state_change_reason=f"router_pending:{persisted.client_order_id}",
        )
    )
    pos = await db_repos.positions.get(pos_id)
    return pos, broker_order


@pytest.mark.asyncio
async def test_fill_transitions_csp_pending_to_csp_open(db_repos):
    broker = PaperBroker(cash=20_000)
    pos, broker_order = await _seed_csp_pending(db_repos, broker)

    await broker.fill_order(broker_order.broker_order_id, fill_price=0.50)

    rec = Reconciler(broker, db_repos, _config())
    summary = await rec.reconcile_once()
    assert summary.fills_processed == 1

    updated = await db_repos.positions.get_by_symbol("test", "F")
    assert updated is not None
    assert updated.state == PositionState.CSP_OPEN
    # Cycle was opened.
    assert summary.cycles_opened == 1
    cycles = await db_repos.cycles.list_open("test")
    assert len(cycles) == 1
    assert cycles[0].symbol == "F"
    assert cycles[0].initial_csp_strike == 9.5


@pytest.mark.asyncio
async def test_assignment_transitions_to_shares_held_with_cost_basis(db_repos):
    broker = PaperBroker(cash=20_000)
    pos, broker_order = await _seed_csp_pending(db_repos, broker)
    await broker.fill_order(broker_order.broker_order_id, fill_price=0.50)
    rec = Reconciler(broker, db_repos, _config())
    await rec.reconcile_once()  # → CSP_OPEN, cycle opened

    # Now simulate assignment at the broker.
    await broker.assign("F250706P00009500")
    await rec.reconcile_once()

    updated = await db_repos.positions.get_by_symbol("test", "F")
    assert updated is not None
    assert updated.state == PositionState.SHARES_HELD
    assert updated.shares == 100
    # cost_basis = strike (9.5) - premium_per_share (0.50) = 9.00
    assert updated.cost_basis == pytest.approx(9.0)


@pytest.mark.asyncio
async def test_csp_worthless_expiration_closes_cycle_and_returns_to_idle(db_repos):
    broker = PaperBroker(cash=20_000)
    _pos, broker_order = await _seed_csp_pending(db_repos, broker)
    await broker.fill_order(broker_order.broker_order_id, fill_price=0.50)
    rec = Reconciler(broker, db_repos, _config())
    await rec.reconcile_once()  # CSP_OPEN

    # "Expire worthless" — broker no longer shows the open short leg.
    await broker.expire("F250706P00009500")
    summary = await rec.reconcile_once()

    updated = await db_repos.positions.get_by_symbol("test", "F")
    assert updated.state == PositionState.IDLE
    assert summary.expirations_processed == 1
    assert summary.cycles_closed == 1


@pytest.mark.asyncio
async def test_unknown_broker_position_flags_manual_intervention(db_repos):
    """Broker reports a position we have no local row for → MANUAL_INTERVENTION."""
    broker = PaperBroker(cash=20_000)
    # Inject a fake position into the broker without going through any order.
    broker._stock["F"] = (200, 9.50)

    rec = Reconciler(broker, db_repos, _config())
    summary = await rec.reconcile_once()

    assert summary.manual_interventions >= 1
    flagged = await db_repos.positions.get_by_symbol("test", "F")
    assert flagged is not None
    assert flagged.state == PositionState.MANUAL_INTERVENTION


@pytest.mark.asyncio
async def test_unknown_order_at_broker_flags_position(db_repos):
    """Broker has an order with a client_order_id we don't know → MANUAL_INTERVENTION."""
    broker = PaperBroker(cash=20_000)
    rogue = Order(
        account_id="test",
        symbol="F",
        order_type=OrderType.SELL_TO_OPEN,
        contract_symbol="F250706P00009500",
        strike=9.5,
        expiration=date(2025, 6, 1) + timedelta(days=35),
        option_type=OptionType.PUT,
        quantity=1,
        limit_price=0.50,
        status=OrderStatus.PENDING,
        placed_at=_utc(),
        client_order_id="wb-rogue-id",  # never persisted locally
    )
    placed = await broker.place_order(rogue)
    await broker.fill_order(placed.broker_order_id, fill_price=0.50)

    rec = Reconciler(broker, db_repos, _config())
    summary = await rec.reconcile_once()
    assert summary.manual_interventions >= 1
    flagged = await db_repos.positions.get_by_symbol("test", "F")
    assert flagged is not None
    assert flagged.state == PositionState.MANUAL_INTERVENTION


@pytest.mark.asyncio
async def test_state_log_records_transitions(db_repos):
    broker = PaperBroker(cash=20_000)
    _pos, broker_order = await _seed_csp_pending(db_repos, broker)
    await broker.fill_order(broker_order.broker_order_id, fill_price=0.50)
    rec = Reconciler(broker, db_repos, _config())
    await rec.reconcile_once()

    pos = await db_repos.positions.get_by_symbol("test", "F")
    log_rows = await db_repos.state_log.list_for_position(pos.id)
    # At least one row representing the fill transition.
    assert any(r.to_state == PositionState.CSP_OPEN for r in log_rows)


@pytest.mark.asyncio
async def test_pending_order_not_yet_filled_is_a_noop(db_repos):
    broker = PaperBroker(cash=20_000)
    _pos, _broker_order = await _seed_csp_pending(db_repos, broker)
    rec = Reconciler(broker, db_repos, _config())
    summary = await rec.reconcile_once()
    assert summary.fills_processed == 0
    assert summary.cycles_opened == 0
    pos = await db_repos.positions.get_by_symbol("test", "F")
    assert pos.state == PositionState.CSP_PENDING


@pytest.mark.asyncio
async def test_assignment_persists_synthetic_stock_buy_for_cycle_pnl(db_repos):
    """Assignment must write a BUY_TO_OPEN row so cycle P&L includes the cost
    basis. Without it, manual_close after a stock drop reports a fake gain."""
    broker = PaperBroker(cash=20_000)
    _pos, broker_order = await _seed_csp_pending(db_repos, broker)
    await broker.fill_order(broker_order.broker_order_id, fill_price=0.50)
    rec = Reconciler(broker, db_repos, _config())
    await rec.reconcile_once()  # CSP_OPEN + cycle opened

    await broker.assign("F250706P00009500")
    await rec.reconcile_once()

    # Cycle order log should now contain a BUY_TO_OPEN at strike 9.5 for 100 shares.
    pos = await db_repos.positions.get_by_symbol("test", "F")
    cycle = await db_repos.cycles.get(pos.current_cycle_id) if pos.current_cycle_id else None
    if cycle is None:
        # cycle was closed in this assignment? assignment alone shouldn't close it.
        c = await db_repos.db.connect()
        async with c.execute("SELECT * FROM orders WHERE order_type = ?", (OrderType.BUY_TO_OPEN.value,)) as cur:
            rows = await cur.fetchall()
    else:
        c = await db_repos.db.connect()
        async with c.execute(
            "SELECT * FROM orders WHERE cycle_id = ? AND order_type = ?",
            (pos.current_cycle_id, OrderType.BUY_TO_OPEN.value),
        ) as cur:
            rows = await cur.fetchall()
    assert any(
        int(r["quantity"]) == 100 and float(r["fill_price"]) == pytest.approx(9.5)
        for r in rows
    ), "expected synthetic BUY_TO_OPEN @ strike 9.5 × 100 shares for the cycle"


@pytest.mark.asyncio
async def test_called_away_persists_synthetic_stock_sell_for_cycle_pnl(db_repos):
    """Called-away must persist a SELL_TO_CLOSE @ CC strike so cycle P&L
    captures the share gain. Pre-fix this returned +80 (premium only)."""
    # Seed a closed-cycle's prior orders directly (the paper broker collapses
    # a stock-+-short-call pair into one position row, which would prevent the
    # reconciler from seeing the CC_OPEN state — fine for real brokers, but a
    # surprise here. So we test _on_called_away as a unit.)
    cycle_id = await db_repos.cycles.insert(
        WheelCycle(
            account_id="test", symbol="F", started_at=_utc(),
            initial_csp_strike=9.5, initial_csp_premium=50.0, n_orders=2,
        )
    )
    # CSP fill: collected $50 premium.
    await db_repos.orders.insert(
        Order(
            account_id="test", symbol="F", cycle_id=cycle_id,
            order_type=OrderType.SELL_TO_OPEN,
            contract_symbol="F250706P00009500",
            strike=9.5, expiration=date(2025, 7, 6),
            option_type=OptionType.PUT,
            quantity=1, fill_price=0.50,
            status=OrderStatus.FILLED,
            placed_at=_utc(),
            client_order_id="wb-csp",
        )
    )
    # Synthetic BUY_TO_OPEN from assignment (this is what _on_assignment writes).
    await db_repos.orders.insert(
        Order(
            account_id="test", symbol="F", cycle_id=cycle_id,
            order_type=OrderType.BUY_TO_OPEN,
            contract_symbol=None, strike=None, expiration=None, option_type=None,
            quantity=100, fill_price=9.0,
            status=OrderStatus.FILLED,
            placed_at=_utc(),
            client_order_id="wb-csp-assign",
        )
    )
    # CC fill at strike 10.5 / premium 0.30.
    await db_repos.orders.insert(
        Order(
            account_id="test", symbol="F", cycle_id=cycle_id,
            order_type=OrderType.SELL_TO_OPEN,
            contract_symbol="F250706C00010500",
            strike=10.5, expiration=date(2025, 7, 6),
            option_type=OptionType.CALL,
            quantity=1, fill_price=0.30,
            status=OrderStatus.FILLED,
            placed_at=_utc(),
            client_order_id="wb-cc",
        )
    )
    pos_id = await db_repos.positions.insert(
        Position(
            account_id="test", symbol="F",
            state=PositionState.CC_OPEN,
            shares=100, cost_basis=9.0,
            current_cycle_id=cycle_id,
            state_changed_at=_utc(),
        )
    )

    broker = PaperBroker(cash=30_000)
    rec = Reconciler(broker, db_repos, _config())
    summary = ReconcileSummary()
    pos = await db_repos.positions.get(pos_id)
    await rec._on_called_away(pos, summary)

    closed = await db_repos.cycles.get(cycle_id)
    # Expected P&L:  +50 (CSP) - 900 (BTO @ 9.0) + 30 (CC) + 1050 (STC @ 10.5) = +230
    assert closed.cycle_outcome == "CC_CALLED_AWAY"
    assert closed.final_pnl == pytest.approx(230.0)


@pytest.mark.asyncio
async def test_called_away_missing_cc_strike_flags_manual_intervention(db_repos):
    """Finding #5: if shares were held but the CC strike can't be recovered, we
    can't record the offsetting share SALE. Closing the cycle now would book the
    full cost basis as a phantom loss. Instead, flag MANUAL_INTERVENTION and
    leave the cycle open for a human."""
    cycle_id = await db_repos.cycles.insert(
        WheelCycle(
            account_id="test", symbol="F", started_at=_utc(),
            initial_csp_strike=9.5, initial_csp_premium=50.0, n_orders=2,
        )
    )
    # CSP fill (+50 premium).
    await db_repos.orders.insert(
        Order(
            account_id="test", symbol="F", cycle_id=cycle_id,
            order_type=OrderType.SELL_TO_OPEN,
            contract_symbol="F250706P00009500",
            strike=9.5, expiration=date(2025, 7, 6),
            option_type=OptionType.PUT,
            quantity=1, fill_price=0.50,
            status=OrderStatus.FILLED,
            placed_at=_utc(),
            client_order_id="wb-csp",
        )
    )
    # Synthetic BUY_TO_OPEN from assignment (the share purchase, -900).
    await db_repos.orders.insert(
        Order(
            account_id="test", symbol="F", cycle_id=cycle_id,
            order_type=OrderType.BUY_TO_OPEN,
            contract_symbol=None, strike=None, expiration=None, option_type=None,
            quantity=100, fill_price=9.0,
            status=OrderStatus.FILLED,
            placed_at=_utc(),
            client_order_id="wb-csp-assign",
        )
    )
    # NOTE: deliberately NO CC SELL_TO_OPEN CALL order → _cycle_cc_strike None.
    pos_id = await db_repos.positions.insert(
        Position(
            account_id="test", symbol="F",
            state=PositionState.CC_OPEN,
            shares=100, cost_basis=9.0,
            current_cycle_id=cycle_id,
            state_changed_at=_utc(),
        )
    )

    broker = PaperBroker(cash=30_000)
    rec = Reconciler(broker, db_repos, _config())
    summary = ReconcileSummary()
    pos = await db_repos.positions.get(pos_id)
    await rec._on_called_away(pos, summary)

    # Cycle stays OPEN (not closed at a phantom loss).
    cyc = await db_repos.cycles.get(cycle_id)
    assert cyc.cycle_outcome is None
    assert cyc.final_pnl is None
    # Position is flagged for human review.
    reloaded = await db_repos.positions.get(pos_id)
    assert reloaded.state == PositionState.MANUAL_INTERVENTION
    assert summary.called_aways_processed == 0
    assert summary.manual_interventions == 1


@pytest.mark.asyncio
async def test_pending_orders_keep_cursor_from_advancing_past_them(db_repos):
    """Regression: Alpaca returns 'accepted' + filled_avg_price set briefly
    before flipping to 'filled'. If our cursor advances past placed_at, we
    miss the eventual FILLED transition. Reconciler must hold lookback to
    include the oldest in-flight order."""

    class _RecordingBroker(PaperBroker):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.since_calls: list[datetime] = []

        async def get_orders_since(self, since: datetime):
            self.since_calls.append(since)
            return await super().get_orders_since(since)

    broker = _RecordingBroker(cash=20_000)
    pos, broker_order = await _seed_csp_pending(db_repos, broker)
    rec = Reconciler(broker, db_repos, _config())

    # First tick — sees the pending order, advances cursor.
    await rec.reconcile_once()
    cursor_after_tick1 = rec._orders_cursor
    assert cursor_after_tick1 is not None

    # Sleep-equivalent: bump the cursor as if 5 minutes passed.
    rec._orders_cursor = cursor_after_tick1 + timedelta(seconds=300)
    placed_at = (await db_repos.orders.get_by_client_id("wb-test-csp-1")).placed_at

    # Second tick — there's still a pending order. The lookback must extend
    # back to (at least) the pending order's placed_at, not the bumped cursor.
    await rec.reconcile_once()
    last_since = broker.since_calls[-1]
    assert last_since <= placed_at, (
        f"reconciler cursor advanced past pending order ({last_since} > {placed_at})"
    )


@pytest.mark.asyncio
async def test_already_idle_position_stays_idle(db_repos):
    broker = PaperBroker(cash=20_000)
    await db_repos.positions.insert(
        Position(
            account_id="test",
            symbol="F",
            state=PositionState.IDLE,
            shares=0,
            state_changed_at=_utc(),
        )
    )
    rec = Reconciler(broker, db_repos, _config())
    summary = await rec.reconcile_once()
    pos = await db_repos.positions.get_by_symbol("test", "F")
    assert pos.state == PositionState.IDLE
    assert summary.manual_interventions == 0


@pytest.mark.asyncio
async def test_csp_buyback_nets_into_cycle_pnl_and_labels_loss(db_repos):
    """Regression for the 2026-05-28 P&L-inflation bug: a CSP bought back at a
    loss must net the buyback debit into final_pnl (not book full premium) and
    label the outcome CSP_CLOSED_LOSS, not _PROFIT."""
    broker = PaperBroker(cash=20_000)
    now = _utc()
    # Open cycle from a CSP sold for $0.46.
    cycle_id = await db_repos.cycles.insert(
        WheelCycle(
            account_id="test",
            symbol="KMI",
            strategy_id="monthly_wheel",
            started_at=now - timedelta(days=10),
            initial_csp_strike=33.5,
            initial_csp_premium=0.46,
        )
    )
    # The original SELL_TO_OPEN, already cycle-linked + filled.
    await db_repos.orders.insert(
        Order(
            account_id="test", symbol="KMI", strategy_id="monthly_wheel",
            cycle_id=cycle_id,
            order_type=OrderType.SELL_TO_OPEN,
            contract_symbol="KMI260618P00033500",
            strike=33.5, expiration=date(2026, 6, 18), option_type=OptionType.PUT,
            quantity=1, limit_price=0.46, fill_price=0.46,
            status=OrderStatus.FILLED, placed_at=now - timedelta(days=10),
            filled_at=now - timedelta(days=10),
        )
    )
    # CSP_OPEN position on that cycle.
    await db_repos.positions.insert(
        Position(
            account_id="test", symbol="KMI", strategy_id="monthly_wheel",
            state=PositionState.CSP_OPEN, shares=0,
            current_cycle_id=cycle_id, state_changed_at=now,
        )
    )
    # Stop-loss buyback at $1.54 — placed + filled, NOT yet cycle-linked.
    buyback = Order(
        account_id="test", symbol="KMI", strategy_id="monthly_wheel",
        order_type=OrderType.BUY_TO_CLOSE,
        contract_symbol="KMI260618P00033500",
        strike=33.5, expiration=date(2026, 6, 18), option_type=OptionType.PUT,
        quantity=1, limit_price=1.54,
        status=OrderStatus.PENDING, placed_at=now,
        client_order_id="wb-kmi-stop",
    )
    bro = await broker.place_order(buyback)
    await db_repos.orders.insert(buyback.model_copy(update={"broker_order_id": bro.broker_order_id}))
    await broker.fill_order(bro.broker_order_id, fill_price=1.54)

    rec = Reconciler(broker, db_repos, _config())
    await rec.reconcile_once()

    cycle = await db_repos.cycles.get(cycle_id)
    # Premium 0.46 − buyback 1.54 = −1.08 × 100 = −108, not the +46 the bug showed.
    assert cycle.final_pnl == pytest.approx(-108.0)
    assert cycle.cycle_outcome == CycleOutcome.CSP_CLOSED_LOSS.value


@pytest.mark.asyncio
async def test_csp_profit_close_nets_buyback_and_keeps_profit_label(db_repos):
    """Profit-close: sold $1.00, bought back $0.40 → +$60, labeled PROFIT."""
    broker = PaperBroker(cash=20_000)
    now = _utc()
    cycle_id = await db_repos.cycles.insert(
        WheelCycle(
            account_id="test", symbol="F", strategy_id="monthly_wheel",
            started_at=now - timedelta(days=10),
            initial_csp_strike=10.0, initial_csp_premium=1.00,
        )
    )
    await db_repos.orders.insert(
        Order(
            account_id="test", symbol="F", strategy_id="monthly_wheel",
            cycle_id=cycle_id, order_type=OrderType.SELL_TO_OPEN,
            contract_symbol="F250706P00010000",
            strike=10.0, expiration=date(2026, 6, 18), option_type=OptionType.PUT,
            quantity=1, limit_price=1.00, fill_price=1.00,
            status=OrderStatus.FILLED, placed_at=now - timedelta(days=10),
            filled_at=now - timedelta(days=10),
        )
    )
    await db_repos.positions.insert(
        Position(
            account_id="test", symbol="F", strategy_id="monthly_wheel",
            state=PositionState.CSP_OPEN, shares=0,
            current_cycle_id=cycle_id, state_changed_at=now,
        )
    )
    buyback = Order(
        account_id="test", symbol="F", strategy_id="monthly_wheel",
        order_type=OrderType.BUY_TO_CLOSE, contract_symbol="F250706P00010000",
        strike=10.0, expiration=date(2026, 6, 18), option_type=OptionType.PUT,
        quantity=1, limit_price=0.40,
        status=OrderStatus.PENDING, placed_at=now, client_order_id="wb-f-pc",
    )
    bro = await broker.place_order(buyback)
    await db_repos.orders.insert(buyback.model_copy(update={"broker_order_id": bro.broker_order_id}))
    await broker.fill_order(bro.broker_order_id, fill_price=0.40)

    rec = Reconciler(broker, db_repos, _config())
    await rec.reconcile_once()

    cycle = await db_repos.cycles.get(cycle_id)
    assert cycle.final_pnl == pytest.approx(60.0)
    assert cycle.cycle_outcome == CycleOutcome.CSP_CLOSED_PROFIT.value


@pytest.mark.asyncio
async def test_cc_fill_links_order_to_existing_cycle(db_repos):
    """Bug A regression: when a covered call (SELL_TO_OPEN call) fills on a
    SHARES_HELD position, the order must be tagged with the position's
    existing wheel cycle. Previously only puts got cycle-linked, so CC orders
    had cycle_id=NULL and the dashboard read the expired CSP for DTE/P&L."""
    broker = PaperBroker(cash=20_000)
    now = _utc()
    # Open wheel cycle (from the original CSP that was assigned).
    cycle_id = await db_repos.cycles.insert(
        WheelCycle(
            account_id="test",
            symbol="F",
            strategy_id="monthly_wheel",
            started_at=now - timedelta(days=20),
            initial_csp_strike=10.0,
            initial_csp_premium=0.40,
        )
    )
    # SHARES_HELD position pointing at that cycle.
    await db_repos.positions.insert(
        Position(
            account_id="test",
            symbol="F",
            strategy_id="monthly_wheel",
            state=PositionState.CC_PENDING,
            shares=100,
            cost_basis=9.60,
            current_cycle_id=cycle_id,
            state_changed_at=now,
        )
    )
    # A CC (SELL_TO_OPEN call) placed + filled, initially no cycle_id.
    cc_order = Order(
        account_id="test",
        symbol="F",
        strategy_id="monthly_wheel",
        order_type=OrderType.SELL_TO_OPEN,
        contract_symbol="F250706C00012000",
        strike=12.0,
        expiration=date(2025, 7, 6),
        option_type=OptionType.CALL,
        quantity=1,
        limit_price=0.30,
        status=OrderStatus.PENDING,
        placed_at=now,
        client_order_id="wb-test-cc-1",
    )
    broker_order = await broker.place_order(cc_order)
    persisted = cc_order.model_copy(update={"broker_order_id": broker_order.broker_order_id})
    cc_id = await db_repos.orders.insert(persisted)
    await broker.fill_order(broker_order.broker_order_id, fill_price=0.30)

    rec = Reconciler(broker, db_repos, _config())
    await rec.reconcile_once()

    # Position is now CC_OPEN, still on the same cycle.
    pos = await db_repos.positions.get_by_symbol("test", "F", strategy_id="monthly_wheel")
    assert pos.state == PositionState.CC_OPEN
    assert pos.current_cycle_id == cycle_id
    # The CC order is now tagged with the cycle (was NULL before the fix).
    cc_after = await db_repos.orders.get(cc_id)
    assert cc_after.cycle_id == cycle_id
