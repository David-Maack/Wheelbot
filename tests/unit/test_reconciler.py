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
async def test_called_away_quantity_comes_from_cycle_not_drifted_local_shares(db_repos):
    """Finding #12: the synthetic share SELL_TO_CLOSE must be sized from the
    cycle's CSP contract count (matching the assignment BUY_TO_OPEN), not from
    local.shares. If local.shares has drifted, sizing off it leaves the buy and
    sell legs unequal and corrupts cycle P&L."""
    cycle_id = await db_repos.cycles.insert(
        WheelCycle(
            account_id="test", symbol="F", started_at=_utc(),
            initial_csp_strike=9.5, initial_csp_premium=100.0, n_orders=3,
        )
    )
    # 2-contract CSP (+0.50 × 2 × 100 = +100).
    await db_repos.orders.insert(Order(
        account_id="test", symbol="F", cycle_id=cycle_id,
        order_type=OrderType.SELL_TO_OPEN, contract_symbol="F250706P00009500",
        strike=9.5, expiration=date(2025, 7, 6), option_type=OptionType.PUT,
        quantity=2, fill_price=0.50, status=OrderStatus.FILLED,
        placed_at=_utc(), client_order_id="wb-csp",
    ))
    # Assignment buy of 200 shares @ 9.0 (-1800).
    await db_repos.orders.insert(Order(
        account_id="test", symbol="F", cycle_id=cycle_id,
        order_type=OrderType.BUY_TO_OPEN, contract_symbol=None,
        quantity=200, fill_price=9.0, status=OrderStatus.FILLED,
        placed_at=_utc(), client_order_id="wb-assign",
    ))
    # CC sell of 2 contracts @ strike 10.5 / 0.30 (+60).
    await db_repos.orders.insert(Order(
        account_id="test", symbol="F", cycle_id=cycle_id,
        order_type=OrderType.SELL_TO_OPEN, contract_symbol="F250706C00010500",
        strike=10.5, expiration=date(2025, 7, 6), option_type=OptionType.CALL,
        quantity=2, fill_price=0.30, status=OrderStatus.FILLED,
        placed_at=_utc(), client_order_id="wb-cc",
    ))
    # Position shares DRIFTED to 100 (should be 200). The fix must ignore this.
    pos_id = await db_repos.positions.insert(Position(
        account_id="test", symbol="F", state=PositionState.CC_OPEN,
        shares=100, cost_basis=9.0, current_cycle_id=cycle_id,
        state_changed_at=_utc(),
    ))

    broker = PaperBroker(cash=30_000)
    rec = Reconciler(broker, db_repos, _config())
    summary = ReconcileSummary()
    pos = await db_repos.positions.get(pos_id)
    await rec._on_called_away(pos, summary)

    # The synthetic SELL_TO_CLOSE must be 200 shares (cycle), not 100 (drift).
    c = await db_repos.db.connect()
    async with c.execute(
        "SELECT quantity, fill_price FROM orders WHERE cycle_id = ? AND order_type = ?",
        (cycle_id, OrderType.SELL_TO_CLOSE.value),
    ) as cur:
        sell_rows = await cur.fetchall()
    assert len(sell_rows) == 1
    assert int(sell_rows[0]["quantity"]) == 200

    closed = await db_repos.cycles.get(cycle_id)
    # +100 (CSP) - 1800 (BTO 200@9) + 60 (CC) + 2100 (STC 200@10.5) = +460.
    # The buggy local.shares path would give 100@10.5 = +1050 → -590.
    assert closed.cycle_outcome == "CC_CALLED_AWAY"
    assert closed.final_pnl == pytest.approx(460.0)


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


# -- Covered-call lifecycle through reconcile_once (finding #8) ---------------
#
# An OPEN covered call holds the 100 shares AND a short call simultaneously, so
# the broker returns TWO rows for the same underlying. The reconciler must read
# the live short-call row as "still open" rather than collapsing the rows and
# mis-firing expiration.


async def _seed_cc_open(db_repos, broker: PaperBroker) -> tuple[int, str]:
    """Put the broker + DB into a live covered-call state and return
    (cycle_id, cc_occ_symbol)."""
    from core.models import OrderType as _OT

    cc_occ = "F250706C00010500"
    # Broker truth: 100 shares held + a short call open against them.
    broker._stock["F"] = (100, 9.0)
    broker._open_options[cc_occ] = Order(
        account_id="test", symbol="F",
        order_type=_OT.SELL_TO_OPEN,
        contract_symbol=cc_occ, strike=10.5,
        expiration=date(2025, 7, 6), option_type=OptionType.CALL,
        quantity=1, fill_price=0.30, status=OrderStatus.FILLED,
        placed_at=_utc(), client_order_id="wb-cc-open",
    )

    cycle_id = await db_repos.cycles.insert(
        WheelCycle(
            account_id="test", symbol="F", started_at=_utc(),
            initial_csp_strike=9.5, initial_csp_premium=50.0, n_orders=3,
        )
    )
    # CSP + assignment buy + CC sell so cc_strike + P&L resolve on called-away.
    await db_repos.orders.insert(Order(
        account_id="test", symbol="F", cycle_id=cycle_id,
        order_type=OrderType.SELL_TO_OPEN, contract_symbol="F250706P00009500",
        strike=9.5, expiration=date(2025, 7, 6), option_type=OptionType.PUT,
        quantity=1, fill_price=0.50, status=OrderStatus.FILLED,
        placed_at=_utc(), client_order_id="wb-csp",
    ))
    await db_repos.orders.insert(Order(
        account_id="test", symbol="F", cycle_id=cycle_id,
        order_type=OrderType.BUY_TO_OPEN, contract_symbol=None,
        quantity=100, fill_price=9.0, status=OrderStatus.FILLED,
        placed_at=_utc(), client_order_id="wb-assign",
    ))
    await db_repos.orders.insert(Order(
        account_id="test", symbol="F", cycle_id=cycle_id,
        order_type=OrderType.SELL_TO_OPEN, contract_symbol=cc_occ,
        strike=10.5, expiration=date(2025, 7, 6), option_type=OptionType.CALL,
        quantity=1, fill_price=0.30, status=OrderStatus.FILLED,
        placed_at=_utc(), client_order_id="wb-cc",
    ))
    await db_repos.positions.insert(Position(
        account_id="test", symbol="F", state=PositionState.CC_OPEN,
        shares=100, cost_basis=9.0, current_cycle_id=cycle_id,
        state_changed_at=_utc(),
    ))
    return cycle_id, cc_occ


@pytest.mark.asyncio
async def test_open_cc_is_not_treated_as_expired(db_repos):
    """The bug: broker returns both a SHARES_HELD row and a CC_OPEN row for an
    open covered call. Collapsing them (last wins) could surface SHARES_HELD and
    fire _on_cc_expiration prematurely. The live short call must keep it open."""
    broker = PaperBroker(cash=20_000)
    await _seed_cc_open(db_repos, broker)

    rec = Reconciler(broker, db_repos, _config())
    summary = await rec.reconcile_once()

    pos = await db_repos.positions.get_by_symbol("test", "F")
    assert pos.state == PositionState.CC_OPEN  # unchanged — still open
    assert summary.expirations_processed == 0
    assert summary.called_aways_processed == 0


@pytest.mark.asyncio
async def test_cc_expires_worthless_transitions_to_shares_held(db_repos):
    """Short call gone but shares remain → call expired worthless, keep shares."""
    broker = PaperBroker(cash=20_000)
    _cycle_id, cc_occ = await _seed_cc_open(db_repos, broker)
    await broker.expire(cc_occ)  # short call expires worthless

    rec = Reconciler(broker, db_repos, _config())
    summary = await rec.reconcile_once()

    pos = await db_repos.positions.get_by_symbol("test", "F")
    assert pos.state == PositionState.SHARES_HELD
    assert pos.shares == 100
    assert summary.expirations_processed == 1


@pytest.mark.asyncio
async def test_cc_called_away_transitions_to_idle_and_closes_cycle(db_repos):
    """Short call assigned → shares delivered out → called away, cycle closed."""
    broker = PaperBroker(cash=20_000)
    cycle_id, cc_occ = await _seed_cc_open(db_repos, broker)
    await broker.assign(cc_occ)  # short call assigned: shares sold at strike

    rec = Reconciler(broker, db_repos, _config())
    summary = await rec.reconcile_once()

    pos = await db_repos.positions.get_by_symbol("test", "F")
    assert pos.state == PositionState.IDLE
    assert pos.current_cycle_id is None
    assert summary.called_aways_processed == 1
    closed = await db_repos.cycles.get(cycle_id)
    assert closed.cycle_outcome == "CC_CALLED_AWAY"


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


# -- TICKET-014.5: exhaustive-dispatch guards ------------------------------

from execution.reconciler import _DIFF_ONE_NO_TRANSITION_STATES  # noqa: E402


# Mirror of the if/elif chain in _diff_one — the states it has an explicit
# handling branch for. Keep in sync when adding a branch. The partition test
# fails if a new PositionState is added without putting it here OR in
# _DIFF_ONE_NO_TRANSITION_STATES.
_DIFF_ONE_HANDLED_STATES = frozenset({
    PositionState.CSP_OPEN,
    PositionState.CC_OPEN,
    PositionState.CSP_PENDING,
    PositionState.CC_PENDING,
    PositionState.SPREAD_OPEN,
    PositionState.SPREAD_PENDING,
    # TICKET-015 PMCC active states with explicit _diff_one branches.
    PositionState.PMCC_LONG_OPEN,
    PositionState.PMCC_BOTH_OPEN,
    # 2026-08-03: PMCC pendings moved from no-transition to a stranded-PENDING
    # self-heal branch (CCL/F sat stuck for weeks holding cap slots).
    PositionState.PMCC_LONG_PENDING,
    PositionState.PMCC_SHORT_PENDING,
})


def test_diff_one_partition_is_exhaustive():
    """Every PositionState is either handled by a _diff_one branch or
    explicitly declared no-transition. A new state (PMCC_* in TICKET-015)
    forces a categorization decision or this test fails — the tripwire that
    makes the silent-fall-through bug PF-2 found impossible to reintroduce."""
    handled = _DIFF_ONE_HANDLED_STATES
    noop = _DIFF_ONE_NO_TRANSITION_STATES
    assert handled.isdisjoint(noop), "a state is both handled and declared no-op"
    missing = set(PositionState) - (handled | noop)
    assert not missing, f"uncategorized PositionState(s): {missing}"


@pytest.mark.asyncio
async def test_diff_one_uncategorized_state_flags_manual(db_repos, monkeypatch):
    """A state that is neither handled nor in the no-op set flags
    MANUAL_INTERVENTION instead of silently skipping. Simulated by removing a
    normally-no-op state (ROLL_EVAL) from the set."""
    import execution.reconciler as rec_mod
    reduced = frozenset(
        s for s in rec_mod._DIFF_ONE_NO_TRANSITION_STATES
        if s != PositionState.ROLL_EVAL
    )
    monkeypatch.setattr(rec_mod, "_DIFF_ONE_NO_TRANSITION_STATES", reduced)

    await db_repos.positions.insert(
        Position(
            account_id="test", symbol="F",
            state=PositionState.ROLL_EVAL, shares=0, state_changed_at=_utc(),
        )
    )
    rec = Reconciler(PaperBroker(), db_repos, _config())
    summary = ReconcileSummary()
    local = await db_repos.positions.get_by_symbol("test", "F")
    await rec._diff_one("F", local, [], summary)
    assert summary.manual_interventions == 1
    pos = await db_repos.positions.get_by_symbol("test", "F")
    assert pos.state == PositionState.MANUAL_INTERVENTION


@pytest.mark.asyncio
async def test_on_fill_unhandled_order_type_flags_manual(db_repos):
    """A filled order whose type _on_fill has no branch for (PMCC's long-call
    BUY_TO_OPEN before TICKET-015 wires it) flags MANUAL_INTERVENTION instead
    of silently no-op'ing."""
    broker = PaperBroker(cash=20_000)
    await db_repos.positions.insert(
        Position(account_id="test", symbol="F", state=PositionState.IDLE,
                 shares=0, state_changed_at=_utc())
    )
    rec = Reconciler(broker, db_repos, _config())
    summary = ReconcileSummary()
    order = Order(
        account_id="test", symbol="F", order_type=OrderType.BUY_TO_OPEN,
        contract_symbol="F250706C00012000", strike=12.0,
        expiration=date(2025, 7, 6), option_type=OptionType.CALL,
        quantity=1, fill_price=2.50, status=OrderStatus.FILLED,
        placed_at=_utc(), client_order_id="wb-pmcc-long-1",
    )
    await rec._on_fill(order, order, summary)
    assert summary.manual_interventions == 1
    pos = await db_repos.positions.get_by_symbol("test", "F")
    assert pos.state == PositionState.MANUAL_INTERVENTION


@pytest.mark.asyncio
async def test_process_orders_isolates_failing_order(db_repos, monkeypatch):
    """Part A: an exception inside _on_fill flags MANUAL_INTERVENTION and the
    reconcile tick does NOT abort — one bad order can't halt reconciliation
    for healthy positions."""
    broker = PaperBroker(cash=20_000)
    _pos, broker_order = await _seed_csp_pending(db_repos, broker)
    await broker.fill_order(broker_order.broker_order_id, fill_price=0.50)
    rec = Reconciler(broker, db_repos, _config())

    async def _boom(*a, **k):
        raise RuntimeError("simulated handler bug")
    monkeypatch.setattr(rec, "_on_fill", _boom)

    # Must return normally — the per-order try/except catches the RuntimeError.
    summary = await rec.reconcile_once()
    assert summary.manual_interventions >= 1
    flagged = await db_repos.positions.get_by_symbol("test", "F")
    assert flagged.state == PositionState.MANUAL_INTERVENTION


@pytest.mark.asyncio
async def test_reconcile_positions_isolates_failing_symbol(db_repos, monkeypatch):
    """Part A: an exception in _diff_one for one symbol doesn't abort the
    per-symbol loop for the others."""
    broker = PaperBroker(cash=20_000)
    await db_repos.positions.insert(
        Position(account_id="test", symbol="F", state=PositionState.CSP_OPEN,
                 shares=0, state_changed_at=_utc())
    )
    await db_repos.positions.insert(
        Position(account_id="test", symbol="BAC", state=PositionState.SHARES_HELD,
                 shares=100, cost_basis=30.0, state_changed_at=_utc())
    )
    rec = Reconciler(broker, db_repos, _config())
    orig = rec._diff_one

    async def _maybe_boom(symbol, local, rows, summary):
        if symbol == "F":
            raise RuntimeError("simulated diff bug")
        return await orig(symbol, local, rows, summary)
    monkeypatch.setattr(rec, "_diff_one", _maybe_boom)

    # The mere fact this returns (rather than propagating the RuntimeError)
    # proves the loop isolated F's failure.
    summary = await rec.reconcile_once()
    f = await db_repos.positions.get_by_symbol("test", "F")
    assert f.state == PositionState.MANUAL_INTERVENTION
    # BAC (SHARES_HELD, a no-op state) was reached and left untouched.
    bac = await db_repos.positions.get_by_symbol("test", "BAC")
    assert bac.state == PositionState.SHARES_HELD


@pytest.mark.asyncio
async def test_compute_cycle_pnl_option_leg_via_buy_to_open_priced_100x(db_repos):
    """A BUY_TO_OPEN carrying an option_type (PMCC long call) is priced at
    100x — the is_option guard discriminates it from synthetic stock legs.
    This means PMCC's long-call P&L is booked correctly the moment it lands,
    without TICKET-015 touching _compute_cycle_pnl."""
    rec = Reconciler(PaperBroker(), db_repos, _config())
    cycle_id = await db_repos.cycles.insert(
        WheelCycle(account_id="test", symbol="AAPL", started_at=_utc(), n_orders=1)
    )
    await db_repos.orders.insert(
        Order(account_id="test", symbol="AAPL", cycle_id=cycle_id,
              order_type=OrderType.BUY_TO_OPEN,
              contract_symbol="AAPL260116C00150000", strike=150.0,
              expiration=date(2026, 1, 16), option_type=OptionType.CALL,
              quantity=1, fill_price=5.00, status=OrderStatus.FILLED,
              placed_at=_utc(), client_order_id="wb-long")
    )
    pnl = await rec._compute_cycle_pnl(cycle_id)
    # BUY (debit), option 100x: -1 * 5.00 * 1 * 100 = -500
    assert pnl == pytest.approx(-500.0)


@pytest.mark.asyncio
async def test_compute_cycle_pnl_synthetic_stock_leg_priced_1x(db_repos):
    """A BUY_TO_OPEN with option_type=None (assignment synthetic stock leg)
    stays at 1x — existing assignment/called-away P&L behavior preserved."""
    rec = Reconciler(PaperBroker(), db_repos, _config())
    cycle_id = await db_repos.cycles.insert(
        WheelCycle(account_id="test", symbol="F", started_at=_utc(), n_orders=1)
    )
    await db_repos.orders.insert(
        Order(account_id="test", symbol="F", cycle_id=cycle_id,
              order_type=OrderType.BUY_TO_OPEN,
              contract_symbol=None, strike=None, expiration=None, option_type=None,
              quantity=100, fill_price=9.0, status=OrderStatus.FILLED,
              placed_at=_utc(), client_order_id="wb-synth")
    )
    pnl = await rec._compute_cycle_pnl(cycle_id)
    # BUY (debit), stock 1x: -1 * 9.0 * 100 * 1 = -900
    assert pnl == pytest.approx(-900.0)


# -- TICKET-015 PMCC: Phase A reconciler lifecycle -------------------------


def _pmcc_order(
    order_type: OrderType,
    option_type: OptionType,
    strike: float,
    *,
    fill_price: float,
    client_id: str,
    trigger_reason: str | None = None,
    quantity: int = 1,
    cycle_id: int | None = None,
) -> Order:
    return Order(
        account_id="test", symbol="AAPL", strategy_id="pmcc",
        cycle_id=cycle_id,
        order_type=order_type,
        contract_symbol=f"AAPL260116{'C' if option_type == OptionType.CALL else 'P'}{int(strike*1000):08d}",
        strike=strike, expiration=date(2026, 1, 16), option_type=option_type,
        quantity=quantity, fill_price=fill_price,
        status=OrderStatus.FILLED, placed_at=_utc(),
        client_order_id=client_id, trigger_reason=trigger_reason,
    )


async def _seed_pmcc_position(db_repos, state: PositionState, cycle_id: int | None = None):
    await db_repos.positions.insert(
        Position(
            account_id="test", symbol="AAPL", strategy_id="pmcc",
            state=state, shares=0, current_cycle_id=cycle_id,
            state_changed_at=_utc(),
        )
    )


@pytest.mark.asyncio
async def test_pmcc_long_fill_opens_cycle_and_long_open(db_repos):
    """BUY_TO_OPEN call (pmcc) → PMCC_LONG_OPEN + a new cycle, with the long
    debit recorded as capital-at-risk."""
    broker = PaperBroker()
    await _seed_pmcc_position(db_repos, PositionState.IDLE)
    rec = Reconciler(broker, db_repos, _config())
    summary = ReconcileSummary()
    order = _pmcc_order(OrderType.BUY_TO_OPEN, OptionType.CALL, 150.0,
                        fill_price=5.00, client_id="pmcc-long-1")
    await rec._on_fill(order, order, summary)

    pos = await db_repos.positions.get_by_symbol("test", "AAPL", strategy_id="pmcc")
    assert pos.state == PositionState.PMCC_LONG_OPEN
    assert pos.current_cycle_id is not None
    assert summary.cycles_opened == 1
    cyc = await db_repos.cycles.get(pos.current_cycle_id)
    assert cyc.initial_csp_strike == 150.0
    # debit 5.00 * 100 * 1 = 500 at risk; premium recorded as -500 (paid out).
    assert cyc.initial_capital_at_risk == pytest.approx(500.0)
    assert cyc.initial_csp_premium == pytest.approx(-500.0)


@pytest.mark.asyncio
async def test_pmcc_short_sell_transitions_to_both_open(db_repos):
    """From PMCC_LONG_OPEN, a SELL_TO_OPEN call → PMCC_BOTH_OPEN, same cycle."""
    broker = PaperBroker()
    cycle_id = await db_repos.cycles.insert(
        WheelCycle(account_id="test", symbol="AAPL", strategy_id="pmcc",
                   started_at=_utc(), n_orders=1)
    )
    await _seed_pmcc_position(db_repos, PositionState.PMCC_LONG_OPEN, cycle_id)
    rec = Reconciler(broker, db_repos, _config())
    summary = ReconcileSummary()
    order = _pmcc_order(OrderType.SELL_TO_OPEN, OptionType.CALL, 165.0,
                        fill_price=1.20, client_id="pmcc-short-1")
    await rec._on_fill(order, order, summary)

    pos = await db_repos.positions.get_by_symbol("test", "AAPL", strategy_id="pmcc")
    assert pos.state == PositionState.PMCC_BOTH_OPEN
    assert pos.current_cycle_id == cycle_id  # same cycle


@pytest.mark.asyncio
async def test_pmcc_short_close_returns_to_long_open_same_cycle(db_repos):
    """From PMCC_BOTH_OPEN, a BUY_TO_CLOSE call → PMCC_LONG_OPEN; the cycle
    persists (the long lives on)."""
    broker = PaperBroker()
    cycle_id = await db_repos.cycles.insert(
        WheelCycle(account_id="test", symbol="AAPL", strategy_id="pmcc",
                   started_at=_utc(), n_orders=2)
    )
    await _seed_pmcc_position(db_repos, PositionState.PMCC_BOTH_OPEN, cycle_id)
    rec = Reconciler(broker, db_repos, _config())
    summary = ReconcileSummary()
    order = _pmcc_order(OrderType.BUY_TO_CLOSE, OptionType.CALL, 165.0,
                        fill_price=0.40, client_id="pmcc-short-close-1")
    await rec._on_fill(order, order, summary)

    pos = await db_repos.positions.get_by_symbol("test", "AAPL", strategy_id="pmcc")
    assert pos.state == PositionState.PMCC_LONG_OPEN
    assert pos.current_cycle_id == cycle_id  # cycle NOT closed
    assert summary.cycles_closed == 0


@pytest.mark.asyncio
async def test_pmcc_multiple_shorts_one_cycle(db_repos):
    """Long open → sell short → buy back → sell another short. All one cycle —
    proves the long-persists-across-shorts invariant."""
    broker = PaperBroker()
    await _seed_pmcc_position(db_repos, PositionState.IDLE)
    rec = Reconciler(broker, db_repos, _config())
    summary = ReconcileSummary()

    long = _pmcc_order(OrderType.BUY_TO_OPEN, OptionType.CALL, 150.0,
                       fill_price=5.00, client_id="pmcc-long")
    await rec._on_fill(long, long, summary)
    pos = await db_repos.positions.get_by_symbol("test", "AAPL", strategy_id="pmcc")
    cycle_id = pos.current_cycle_id

    short1 = _pmcc_order(OrderType.SELL_TO_OPEN, OptionType.CALL, 165.0,
                         fill_price=1.20, client_id="pmcc-s1")
    await rec._on_fill(short1, short1, summary)
    close1 = _pmcc_order(OrderType.BUY_TO_CLOSE, OptionType.CALL, 165.0,
                         fill_price=0.40, client_id="pmcc-c1")
    await rec._on_fill(close1, close1, summary)
    short2 = _pmcc_order(OrderType.SELL_TO_OPEN, OptionType.CALL, 167.0,
                         fill_price=1.10, client_id="pmcc-s2")
    await rec._on_fill(short2, short2, summary)

    pos = await db_repos.positions.get_by_symbol("test", "AAPL", strategy_id="pmcc")
    assert pos.state == PositionState.PMCC_BOTH_OPEN
    assert pos.current_cycle_id == cycle_id   # still the original cycle
    assert summary.cycles_opened == 1
    assert summary.cycles_closed == 0


@pytest.mark.asyncio
async def test_pmcc_long_close_full_ends_cycle(db_repos):
    """SELL_TO_CLOSE long (no roll reason) → IDLE + cycle closed PMCC_FULL_CLOSED."""
    broker = PaperBroker()
    cycle_id = await db_repos.cycles.insert(
        WheelCycle(account_id="test", symbol="AAPL", strategy_id="pmcc",
                   started_at=_utc(), n_orders=1)
    )
    await _seed_pmcc_position(db_repos, PositionState.PMCC_LONG_OPEN, cycle_id)
    rec = Reconciler(broker, db_repos, _config())
    summary = ReconcileSummary()
    order = _pmcc_order(OrderType.SELL_TO_CLOSE, OptionType.CALL, 150.0,
                        fill_price=6.50, client_id="pmcc-long-close")
    await rec._on_fill(order, order, summary)

    pos = await db_repos.positions.get_by_symbol("test", "AAPL", strategy_id="pmcc")
    assert pos.state == PositionState.IDLE
    assert pos.current_cycle_id is None
    assert summary.cycles_closed == 1
    cyc = await db_repos.cycles.get(cycle_id)
    assert cyc.cycle_outcome == CycleOutcome.PMCC_FULL_CLOSED


@pytest.mark.asyncio
async def test_pmcc_long_close_roll_tags_rolled(db_repos):
    """SELL_TO_CLOSE long with trigger_reason 'pmcc_roll' → PMCC_LONG_ROLLED."""
    broker = PaperBroker()
    cycle_id = await db_repos.cycles.insert(
        WheelCycle(account_id="test", symbol="AAPL", strategy_id="pmcc",
                   started_at=_utc(), n_orders=1)
    )
    await _seed_pmcc_position(db_repos, PositionState.PMCC_LONG_OPEN, cycle_id)
    rec = Reconciler(broker, db_repos, _config())
    summary = ReconcileSummary()
    order = _pmcc_order(OrderType.SELL_TO_CLOSE, OptionType.CALL, 150.0,
                        fill_price=6.50, client_id="pmcc-roll",
                        trigger_reason="pmcc_roll_dte")
    await rec._on_fill(order, order, summary)

    cyc = await db_repos.cycles.get(cycle_id)
    assert cyc.cycle_outcome == CycleOutcome.PMCC_LONG_ROLLED


@pytest.mark.asyncio
async def test_pmcc_short_expiration_via_diff_one(db_repos):
    """In PMCC_BOTH_OPEN, broker shows no short call + no shares → the short
    expired worthless → _diff_one returns to PMCC_LONG_OPEN, cycle continues."""
    broker = PaperBroker()
    cycle_id = await db_repos.cycles.insert(
        WheelCycle(account_id="test", symbol="AAPL", strategy_id="pmcc",
                   started_at=_utc(), n_orders=2)
    )
    await _seed_pmcc_position(db_repos, PositionState.PMCC_BOTH_OPEN, cycle_id)
    rec = Reconciler(broker, db_repos, _config())
    summary = ReconcileSummary()
    local = await db_repos.positions.get_by_symbol("test", "AAPL", strategy_id="pmcc")
    # No broker rows → no short, no shares.
    await rec._diff_one("AAPL", local, [], summary)

    pos = await db_repos.positions.get_by_symbol("test", "AAPL", strategy_id="pmcc")
    assert pos.state == PositionState.PMCC_LONG_OPEN
    assert pos.current_cycle_id == cycle_id   # cycle preserved


@pytest.mark.asyncio
async def test_pmcc_short_assignment_flags_manual(db_repos):
    """In PMCC_BOTH_OPEN, shares appearing at the broker = short assignment →
    flag MANUAL_INTERVENTION (D3: covered-exercise math not auto-reconciled)."""
    broker = PaperBroker()
    await _seed_pmcc_position(db_repos, PositionState.PMCC_BOTH_OPEN)
    rec = Reconciler(broker, db_repos, _config())
    summary = ReconcileSummary()
    local = await db_repos.positions.get_by_symbol("test", "AAPL", strategy_id="pmcc")
    broker_row = Position(
        account_id="test", symbol="AAPL", state=PositionState.SHARES_HELD,
        shares=100, state_changed_at=_utc(),
    )
    await rec._diff_one("AAPL", local, [broker_row], summary)
    assert summary.manual_interventions == 1


@pytest.mark.asyncio
async def test_pmcc_full_pnl_across_long_and_shorts(db_repos):
    """Full PMCC cycle P&L: long debit + short credits/debits + long sale, all
    priced at 100x (TICKET-014.5 is_option discriminator). Proves PMCC P&L
    needs no special handling in _compute_cycle_pnl."""
    rec = Reconciler(PaperBroker(), db_repos, _config())
    cycle_id = await db_repos.cycles.insert(
        WheelCycle(account_id="test", symbol="AAPL", strategy_id="pmcc",
                   started_at=_utc(), n_orders=4)
    )
    for o in [
        _pmcc_order(OrderType.BUY_TO_OPEN, OptionType.CALL, 150.0,
                    fill_price=5.00, client_id="l", cycle_id=cycle_id),    # -500
        _pmcc_order(OrderType.SELL_TO_OPEN, OptionType.CALL, 165.0,
                    fill_price=1.00, client_id="s1", cycle_id=cycle_id),   # +100
        _pmcc_order(OrderType.BUY_TO_CLOSE, OptionType.CALL, 165.0,
                    fill_price=0.40, client_id="c1", cycle_id=cycle_id),   # -40
        _pmcc_order(OrderType.SELL_TO_CLOSE, OptionType.CALL, 150.0,
                    fill_price=6.50, client_id="lc", cycle_id=cycle_id),   # +650
    ]:
        await db_repos.orders.insert(o)
    pnl = await rec._compute_cycle_pnl(cycle_id)
    # -500 + 100 - 40 + 650 = +210
    assert pnl == pytest.approx(210.0)


# -- strategy-scoped share inference (2026-07-23 review fix) -------------------


@pytest.mark.asyncio
async def test_csp_shares_claimed_by_other_strategy_flags_not_assigns(db_repos):
    """monthly_wheel holds shares; weekly_wheel's CSP on the same symbol sees
    those shares at the broker. Pre-fix this booked a FALSE assignment into
    the weekly cycle (phantom cost basis + synthetic BUY_TO_OPEN). Ambiguous
    shares now go to MANUAL_INTERVENTION on the RIGHT strategy's row."""
    broker = PaperBroker()
    now = _utc()
    # monthly_wheel legitimately holds 100 shares.
    await db_repos.positions.insert(Position(
        account_id="test", symbol="F", strategy_id="monthly_wheel",
        state=PositionState.SHARES_HELD, shares=100, state_changed_at=now,
    ))
    # weekly_wheel has a CSP_OPEN on the same symbol.
    cycle_id = await db_repos.cycles.insert(WheelCycle(
        account_id="test", symbol="F", strategy_id="weekly_wheel",
        started_at=now,
    ))
    weekly_id = await db_repos.positions.insert(Position(
        account_id="test", symbol="F", strategy_id="weekly_wheel",
        state=PositionState.CSP_OPEN, shares=0, current_cycle_id=cycle_id,
        state_changed_at=now,
    ))
    weekly = await db_repos.positions.get_by_symbol(
        "test", "F", strategy_id="weekly_wheel")
    rec = Reconciler(broker, db_repos, _config())
    summary = ReconcileSummary()
    # Broker view: shares on the symbol, no short put.
    broker_row = Position(
        account_id="test", symbol="F", state=PositionState.SHARES_HELD,
        shares=100, state_changed_at=now,
    )
    await rec._diff_one("F", weekly, [broker_row], summary)

    weekly_after = await db_repos.positions.get_by_symbol(
        "test", "F", strategy_id="weekly_wheel")
    assert weekly_after.state == PositionState.MANUAL_INTERVENTION
    assert weekly_after.id == weekly_id
    # No false assignment: no shares booked, monthly row untouched.
    assert weekly_after.shares == 0
    monthly_after = await db_repos.positions.get_by_symbol(
        "test", "F", strategy_id="monthly_wheel")
    assert monthly_after.state == PositionState.SHARES_HELD


@pytest.mark.asyncio
async def test_csp_assignment_still_inferred_when_unambiguous(db_repos):
    """Single strategy on the symbol -> the normal assignment inference stands."""
    broker = PaperBroker()
    now = _utc()
    cycle_id = await db_repos.cycles.insert(WheelCycle(
        account_id="test", symbol="F", strategy_id="monthly_wheel",
        started_at=now, initial_csp_strike=9.5,
    ))
    await db_repos.orders.insert(Order(
        account_id="test", symbol="F", strategy_id="monthly_wheel",
        cycle_id=cycle_id, order_type=OrderType.SELL_TO_OPEN,
        contract_symbol="F250706P00009500", strike=9.5,
        expiration=date(2025, 7, 6), option_type=OptionType.PUT,
        quantity=1, fill_price=0.50, status=OrderStatus.FILLED,
        placed_at=now, client_order_id="csp-assign-test",
    ))
    await db_repos.positions.insert(Position(
        account_id="test", symbol="F", strategy_id="monthly_wheel",
        state=PositionState.CSP_OPEN, shares=0, current_cycle_id=cycle_id,
        state_changed_at=now,
    ))
    local = await db_repos.positions.get_by_symbol(
        "test", "F", strategy_id="monthly_wheel")
    rec = Reconciler(broker, db_repos, _config())
    summary = ReconcileSummary()
    broker_row = Position(
        account_id="test", symbol="F", state=PositionState.SHARES_HELD,
        shares=100, state_changed_at=now,
    )
    await rec._diff_one("F", local, [broker_row], summary)
    after = await db_repos.positions.get_by_symbol(
        "test", "F", strategy_id="monthly_wheel")
    assert after.state != PositionState.MANUAL_INTERVENTION  # assignment path ran
