"""Round-trip tests for the in-memory paper broker.

These also serve as the executable contract for the Broker ABC; downstream
unit tests in Sprint 3+ are expected to lean on the same helpers.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from core.broker import BrokerUnavailable, OrderRejected
from core.models import (
    OptionContract,
    OptionType,
    Order,
    OrderStatus,
    OrderType,
    PositionState,
    Quote,
)
from platforms.paper_broker import PaperBroker


def _make_csp_order(occ: str = "F250620P00010000", strike: float = 10.0, qty: int = 1) -> Order:
    return Order(
        account_id="PAPER-1",
        symbol="F",
        order_type=OrderType.SELL_TO_OPEN,
        contract_symbol=occ,
        strike=strike,
        expiration=date(2025, 6, 20),
        option_type=OptionType.PUT,
        quantity=qty,
        limit_price=0.50,
        status=OrderStatus.PENDING,
        placed_at=datetime.now(UTC).replace(tzinfo=None),
        client_order_id="cli-1",
    )


@pytest.mark.asyncio
async def test_account_starts_with_seed_cash():
    broker = PaperBroker(cash=10_000)
    acct = await broker.get_account()
    assert acct.cash == 10_000
    assert acct.buying_power == 10_000
    assert acct.equity == 10_000


@pytest.mark.asyncio
async def test_place_order_returns_broker_id_and_pending_status():
    broker = PaperBroker()
    placed = await broker.place_order(_make_csp_order())
    assert placed.broker_order_id is not None
    assert placed.status == OrderStatus.PENDING
    assert placed.raw_request is not None  # captured for audit trail


@pytest.mark.asyncio
async def test_place_order_rejects_zero_quantity():
    broker = PaperBroker()
    with pytest.raises(OrderRejected):
        await broker.place_order(_make_csp_order(qty=0))


@pytest.mark.asyncio
async def test_place_order_idempotent_on_client_order_id():
    """Sprint 4's router relies on this — same client_order_id = same broker order."""
    broker = PaperBroker()
    first = await broker.place_order(_make_csp_order())
    second = await broker.place_order(_make_csp_order())
    assert first.broker_order_id == second.broker_order_id


@pytest.mark.asyncio
async def test_fill_credits_premium_and_records_open_short():
    broker = PaperBroker(cash=10_000)
    placed = await broker.place_order(_make_csp_order())
    filled = await broker.fill_order(placed.broker_order_id, fill_price=0.50)

    assert filled.status == OrderStatus.FILLED
    assert filled.fill_price == 0.50

    acct = await broker.get_account()
    # 1 contract * 100 shares * $0.50 premium = $50 credit
    assert acct.cash == pytest.approx(10_050.0)

    positions = await broker.get_positions()
    assert len(positions) == 1
    assert positions[0].symbol == "F"
    assert positions[0].state == PositionState.CSP_OPEN


@pytest.mark.asyncio
async def test_assignment_delivers_shares_and_debits_strike():
    broker = PaperBroker(cash=10_000)
    placed = await broker.place_order(_make_csp_order())
    await broker.fill_order(placed.broker_order_id, fill_price=0.50)

    await broker.assign("F250620P00010000")

    acct = await broker.get_account()
    # +50 premium - 100*$10 strike = -950
    assert acct.cash == pytest.approx(9_050.0)

    positions = await broker.get_positions()
    assert len(positions) == 1
    pos = positions[0]
    assert pos.symbol == "F"
    assert pos.shares == 100
    assert pos.cost_basis == pytest.approx(10.0)
    assert pos.state == PositionState.SHARES_HELD


@pytest.mark.asyncio
async def test_cancel_pending_order():
    broker = PaperBroker()
    placed = await broker.place_order(_make_csp_order())
    await broker.cancel_order(placed.broker_order_id)
    orders = await broker.get_orders_since(datetime(2000, 1, 1))
    assert orders[0].status == OrderStatus.CANCELLED


@pytest.mark.asyncio
async def test_cancel_filled_order_is_noop():
    broker = PaperBroker()
    placed = await broker.place_order(_make_csp_order())
    await broker.fill_order(placed.broker_order_id, fill_price=0.50)
    await broker.cancel_order(placed.broker_order_id)  # should not raise or rewrite
    orders = await broker.get_orders_since(datetime(2000, 1, 1))
    assert orders[0].status == OrderStatus.FILLED


@pytest.mark.asyncio
async def test_get_orders_since_filters_by_timestamp():
    broker = PaperBroker()
    await broker.place_order(_make_csp_order())
    future = datetime.utcnow() + timedelta(hours=1)
    assert await broker.get_orders_since(future) == []


@pytest.mark.asyncio
async def test_quote_round_trip():
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F", bid=10.0, ask=10.04, last=10.02))
    q = await broker.get_quote("F")
    assert q.mid == pytest.approx(10.02)


@pytest.mark.asyncio
async def test_chain_filtering():
    broker = PaperBroker()
    chain = [
        OptionContract(
            underlying="F",
            occ_symbol="F250620P00010000",
            strike=10.0,
            expiration=date(2025, 6, 20),
            option_type=OptionType.PUT,
        ),
        OptionContract(
            underlying="F",
            occ_symbol="F250620C00010000",
            strike=10.0,
            expiration=date(2025, 6, 20),
            option_type=OptionType.CALL,
        ),
        OptionContract(
            underlying="F",
            occ_symbol="F250718P00010000",
            strike=10.0,
            expiration=date(2025, 7, 18),
            option_type=OptionType.PUT,
        ),
    ]
    broker.seed_chain("F", chain)
    puts_jun = await broker.get_option_chain(
        "F", expiration=date(2025, 6, 20), option_type=OptionType.PUT
    )
    assert len(puts_jun) == 1
    assert puts_jun[0].occ_symbol == "F250620P00010000"


@pytest.mark.asyncio
async def test_unavailable_flag_raises_for_retry_path():
    broker = PaperBroker()
    broker.set_unavailable(True)
    with pytest.raises(BrokerUnavailable):
        await broker.get_account()
