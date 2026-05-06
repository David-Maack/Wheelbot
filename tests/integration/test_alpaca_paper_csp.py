"""Integration test — spec §13 item 10.

Places a single-contract cash-secured put through the Broker abstraction
against Alpaca's paper endpoints, verifies the order surfaces via
get_orders_since, then cancels it.

Skip rules:
- Skipped automatically when ALPACA_API_KEY / ALPACA_API_SECRET are absent
  (so `pytest tests/` stays green on a fresh clone).
- Skipped when the alpaca-py package isn't installed
  (project extra: `pip install -e ".[broker,dev]"`).

Designed to NOT fill: limit price is set far below the bid so the order rests
on the book and we cancel it before market close. If you run this near close,
prefer to run during regular hours.
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, date, datetime, timedelta

import pytest

pytestmark = pytest.mark.integration

alpaca = pytest.importorskip("alpaca", reason="alpaca-py not installed; install '.[broker]'")

from core.config import load_secrets  # noqa: E402
from core.models import OptionType, Order, OrderStatus, OrderType  # noqa: E402

load_secrets()

if not (os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_API_SECRET")):
    pytest.skip("ALPACA_API_KEY/SECRET not set in env or config/secrets.env", allow_module_level=True)


@pytest.mark.asyncio
async def test_place_paper_csp_through_abstraction():
    from platforms.alpaca_broker import AlpacaBroker

    broker = AlpacaBroker()

    # Sanity: can talk to Alpaca at all.
    account = await broker.get_account()
    assert account.cash >= 0

    # Pick a near-dated put on F that's far OTM so a tiny limit price won't fill.
    today = date.today()
    target_expiry = today + timedelta(days=30)

    chain = await broker.get_option_chain("F", option_type=OptionType.PUT)
    if not chain:
        pytest.skip("Alpaca returned empty F put chain (market closed or feed delay)")

    # Closest-expiry contract, strike well below current bid (low-delta, won't fill cheap).
    chain.sort(key=lambda c: (abs((c.expiration - target_expiry).days), -c.strike))
    contract = next((c for c in chain if c.strike <= 5.0), chain[0])

    placed = await broker.place_order(
        Order(
            account_id="alpaca-paper",
            symbol="F",
            order_type=OrderType.SELL_TO_OPEN,
            contract_symbol=contract.occ_symbol,
            strike=contract.strike,
            expiration=contract.expiration,
            option_type=OptionType.PUT,
            quantity=1,
            limit_price=0.01,  # far below mid — should rest, not fill
            status=OrderStatus.PENDING,
            placed_at=datetime.now(UTC).replace(tzinfo=None),
            client_order_id=f"wheelbot-it-{uuid.uuid4().hex[:8]}",
        )
    )

    assert placed.broker_order_id, "expected a broker_order_id from Alpaca"
    assert placed.status in (
        OrderStatus.PENDING,
        OrderStatus.FILLED,  # vanishingly unlikely at $0.01 limit, but legal
        OrderStatus.REJECTED,  # if account lacks options approval
    )

    if placed.status == OrderStatus.REJECTED:
        pytest.skip("Alpaca paper account lacks options approval — order rejected")

    # Verify it round-trips through get_orders_since.
    cutoff = datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=5)
    recent = await broker.get_orders_since(cutoff)
    assert any(o.broker_order_id == placed.broker_order_id for o in recent), (
        f"placed order {placed.broker_order_id} not in recent orders"
    )

    # Always cancel — never leave a paper order hanging.
    await broker.cancel_order(placed.broker_order_id)
