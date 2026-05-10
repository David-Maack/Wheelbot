"""Multi-leg order primitive — Sprint 9 sub-sprint 2.

Covers:
  - Broker ABC default raises NotImplementedError on adapters that haven't
    overridden it.
  - PaperBroker.place_multi_leg_order persists legs in raw_request, infers
    MULTI_LEG_OPEN/CLOSE order_type from leg actions, is idempotent on
    client_order_id, and rejects empty/zero-qty packages.
  - PaperBroker.fill_multi_leg credits cash by net price * 100 * qty,
    registers each open leg in _open_options, and clears any closed legs.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from core.broker import Broker, OrderRejected
from core.models import (
    Account,
    OptionContract,
    OptionType,
    Order,
    OrderLeg,
    OrderStatus,
    OrderType,
    Position,
    Quote,
)
from platforms.paper_broker import PaperBroker


def _put_leg(strike: float, action: OrderType) -> OrderLeg:
    return OrderLeg(
        contract_symbol=f"F250620P{int(strike * 1000):08d}",
        underlying="F",
        option_type=OptionType.PUT,
        strike=strike,
        expiration=date(2025, 6, 20),
        action=action,
    )


def _vertical_legs() -> list[OrderLeg]:
    """Bull put credit spread: short higher strike, long lower strike."""
    return [
        _put_leg(10.0, OrderType.SELL_TO_OPEN),
        _put_leg(9.0, OrderType.BUY_TO_OPEN),
    ]


@pytest.mark.asyncio
async def test_paper_broker_place_multi_leg_returns_pending_with_inferred_type():
    broker = PaperBroker(cash=10_000)
    placed = await broker.place_multi_leg_order(
        underlying="F",
        legs=_vertical_legs(),
        quantity=2,
        limit_price=0.30,
        client_order_id="mleg-1",
        strategy_id="put_spread",
    )
    assert placed.broker_order_id is not None
    assert placed.status == OrderStatus.PENDING
    assert placed.order_type == OrderType.MULTI_LEG_OPEN
    assert placed.strategy_id == "put_spread"
    assert placed.symbol == "F"
    assert placed.quantity == 2
    assert placed.raw_request is not None
    assert len(placed.raw_request["legs"]) == 2


@pytest.mark.asyncio
async def test_paper_broker_multi_leg_close_inferred_when_all_close_actions():
    broker = PaperBroker()
    placed = await broker.place_multi_leg_order(
        underlying="F",
        legs=[
            _put_leg(10.0, OrderType.BUY_TO_CLOSE),
            _put_leg(9.0, OrderType.SELL_TO_CLOSE),
        ],
        quantity=1,
        limit_price=-0.10,  # debit to close
        client_order_id="mleg-close-1",
    )
    assert placed.order_type == OrderType.MULTI_LEG_CLOSE


@pytest.mark.asyncio
async def test_paper_broker_multi_leg_idempotent_on_client_order_id():
    broker = PaperBroker()
    a = await broker.place_multi_leg_order(
        underlying="F",
        legs=_vertical_legs(),
        quantity=1,
        limit_price=0.25,
        client_order_id="dup-1",
    )
    b = await broker.place_multi_leg_order(
        underlying="F",
        legs=_vertical_legs(),
        quantity=1,
        limit_price=0.25,
        client_order_id="dup-1",
    )
    assert a.broker_order_id == b.broker_order_id


@pytest.mark.asyncio
async def test_paper_broker_multi_leg_rejects_empty_legs():
    broker = PaperBroker()
    with pytest.raises(OrderRejected):
        await broker.place_multi_leg_order(
            underlying="F",
            legs=[],
            quantity=1,
            limit_price=0.10,
        )


@pytest.mark.asyncio
async def test_paper_broker_multi_leg_rejects_zero_quantity():
    broker = PaperBroker()
    with pytest.raises(OrderRejected):
        await broker.place_multi_leg_order(
            underlying="F",
            legs=_vertical_legs(),
            quantity=0,
            limit_price=0.10,
        )


@pytest.mark.asyncio
async def test_fill_multi_leg_credits_cash_and_registers_legs():
    broker = PaperBroker(cash=10_000)
    placed = await broker.place_multi_leg_order(
        underlying="F",
        legs=_vertical_legs(),
        quantity=2,
        limit_price=0.30,
        client_order_id="fill-test-1",
    )
    filled = await broker.fill_multi_leg(placed.broker_order_id, fill_price=0.30)
    assert filled.status == OrderStatus.FILLED
    assert filled.fill_price == 0.30

    # Cash effect: 0.30 * 100 * 2 = $60 credit on top of seed.
    acct = await broker.get_account()
    assert acct.cash == pytest.approx(10_060.0)

    # Both legs registered in open_options.
    assert "F250620P00010000" in broker._open_options
    assert "F250620P00009000" in broker._open_options


@pytest.mark.asyncio
async def test_fill_multi_leg_close_clears_open_legs():
    broker = PaperBroker(cash=10_000)
    open_pkg = await broker.place_multi_leg_order(
        underlying="F",
        legs=_vertical_legs(),
        quantity=1,
        limit_price=0.40,
        client_order_id="open-1",
    )
    await broker.fill_multi_leg(open_pkg.broker_order_id, fill_price=0.40)
    assert "F250620P00010000" in broker._open_options

    close_pkg = await broker.place_multi_leg_order(
        underlying="F",
        legs=[
            _put_leg(10.0, OrderType.BUY_TO_CLOSE),
            _put_leg(9.0, OrderType.SELL_TO_CLOSE),
        ],
        quantity=1,
        limit_price=-0.10,
        client_order_id="close-1",
    )
    await broker.fill_multi_leg(close_pkg.broker_order_id, fill_price=-0.10)
    assert "F250620P00010000" not in broker._open_options
    assert "F250620P00009000" not in broker._open_options


class _StubBroker(Broker):
    @property
    def name(self) -> str:
        return "stub"

    async def get_account(self) -> Account:  # pragma: no cover - unused
        raise NotImplementedError

    async def get_positions(self) -> list[Position]:  # pragma: no cover - unused
        return []

    async def get_quote(self, symbol: str) -> Quote:  # pragma: no cover - unused
        raise NotImplementedError

    async def get_option_chain(  # pragma: no cover - unused
        self,
        underlying: str,
        expiration: date | None = None,
        option_type: OptionType | None = None,
    ) -> list[OptionContract]:
        return []

    async def place_order(self, order: Order) -> Order:  # pragma: no cover - unused
        raise NotImplementedError

    async def cancel_order(self, broker_order_id: str) -> None:  # pragma: no cover
        return None

    async def get_orders_since(self, since: datetime) -> list[Order]:  # pragma: no cover
        return []


@pytest.mark.asyncio
async def test_broker_abc_default_multi_leg_raises_not_implemented():
    broker = _StubBroker()
    with pytest.raises(NotImplementedError):
        await broker.place_multi_leg_order(
            underlying="F",
            legs=_vertical_legs(),
            quantity=1,
            limit_price=0.30,
        )
