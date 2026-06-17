"""TICKET-028 — sandbox execution smoke orchestration (mock broker, no network).

The live run is scripts/tastytrade_sandbox_smoke.py against the real sandbox;
here we just verify run_smoke()'s control flow: happy-path place/verify/cancel,
the market-hours skip, and the invalid_symbol hard-fail classification.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

from core.broker import OrderRejected
from core.models import OptionContract, OptionType, Order, OrderStatus, OrderType
from scripts.tastytrade_sandbox_smoke import run_smoke

_EXP = date(2026, 7, 17)
_NOW = datetime(2026, 6, 16, 12, 0, 0)


class _StubBroker:
    """Duck-typed broker for the smoke's call surface."""

    def __init__(self, *, place_exc=None, mleg_exc=None, positions=None):
        self._place_exc = place_exc
        self._mleg_exc = mleg_exc
        self._positions = positions or []
        self.cancelled: list[str] = []
        self._ids: list[str] = []

    async def get_option_chain(self, underlying, expiration=None, option_type=None):
        return [
            OptionContract(underlying="F", occ_symbol="F260717P00004000", strike=4.0,
                           expiration=_EXP, option_type=OptionType.PUT),
            OptionContract(underlying="F", occ_symbol="F260717P00005000", strike=5.0,
                           expiration=_EXP, option_type=OptionType.PUT),
        ]

    async def place_order(self, order):
        if self._place_exc:
            raise self._place_exc
        oid = f"ord-{len(self._ids) + 1}"
        self._ids.append(oid)
        return order.model_copy(update={"broker_order_id": oid, "status": OrderStatus.PENDING})

    async def place_multi_leg_order(self, **kw):
        if self._mleg_exc:
            raise self._mleg_exc
        oid = f"ord-{len(self._ids) + 1}"
        self._ids.append(oid)
        return Order(account_id="tastytrade", symbol="F", order_type=OrderType.SELL_TO_OPEN,
                     quantity=1, status=OrderStatus.PENDING, placed_at=_NOW, broker_order_id=oid)

    async def get_orders_since(self, since):
        return [Order(account_id="tastytrade", symbol="F", order_type=OrderType.SELL_TO_OPEN,
                      quantity=1, status=OrderStatus.PENDING, placed_at=_NOW, broker_order_id=oid)
                for oid in self._ids]

    async def cancel_order(self, oid):
        self.cancelled.append(oid)

    async def get_positions(self):
        return self._positions


@pytest.mark.asyncio
async def test_smoke_happy_path_places_verifies_cancels():
    b = _StubBroker()
    single, multi, positions = await run_smoke(b)
    assert "single-leg: PASS" in single and "in_orders_since=True" in single
    assert "multi-leg: PASS" in multi
    assert "positions: OK — 0 open" in positions
    assert len(b.cancelled) == 2  # both orders placed and cancelled


@pytest.mark.asyncio
async def test_smoke_reports_market_hours_block():
    exc = OrderRejected(
        "tastytrade rejected order: tif.day_invalid_intersession_options: "
        "Day orders will be accepted after 3:15pm CT."
    )
    b = _StubBroker(place_exc=exc, mleg_exc=exc)
    single, multi, _positions = await run_smoke(b)
    assert "MARKET-HOURS SKIP" in single
    assert "MARKET-HOURS SKIP" in multi
    assert b.cancelled == []  # nothing placed → nothing to cancel


@pytest.mark.asyncio
async def test_smoke_flags_invalid_symbol_as_hard_fail():
    exc = OrderRejected(
        "tastytrade rejected order: invalid_symbol: This order contains a leg "
        "with an invalid symbol."
    )
    b = _StubBroker(place_exc=exc)
    single, *_ = await run_smoke(b)
    assert "FAIL invalid_symbol" in single
