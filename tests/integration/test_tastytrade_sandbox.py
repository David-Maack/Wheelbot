"""Integration test — spec §13 #28.

Talks to Tastytrade's sandbox (cert.tastyworks.com). Auto-skips when the
required env vars are absent — `pytest tests/` stays green on a fresh clone.

Required env (set via `python -m scripts.bootstrap_tastytrade --sandbox` or
manually in `config/secrets.env`):

    TASTYTRADE_PROVIDER_SECRET
    TASTYTRADE_REMEMBER_TOKEN
    TASTYTRADE_ACCOUNT_NUMBER         (optional — first account picked otherwise)
    TASTYTRADE_USE_SANDBOX=true

The test does not place fillable orders. It places a $0.01 limit CSP on F
that should rest on the book, verifies it shows up via get_orders_since,
then cancels it. **This is sandbox; it should never touch real money.**
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, date, datetime, timedelta

import pytest

pytestmark = pytest.mark.integration

# Hard skip if SDK absent.
pytest.importorskip("tastytrade", reason="tastytrade SDK not installed; install '.[broker]'")

from core.config import load_secrets  # noqa: E402
from core.models import OptionType, Order, OrderStatus, OrderType  # noqa: E402

load_secrets()

if not (
    os.environ.get("TASTYTRADE_PROVIDER_SECRET")
    and os.environ.get("TASTYTRADE_REMEMBER_TOKEN")
):
    pytest.skip(
        "TASTYTRADE_PROVIDER_SECRET / TASTYTRADE_REMEMBER_TOKEN not set",
        allow_module_level=True,
    )

# Belt + suspenders: refuse to run against prod even if creds are valid.
if os.environ.get("TASTYTRADE_USE_SANDBOX", "true").lower() not in ("true", "1", "yes"):
    pytest.skip(
        "TASTYTRADE_USE_SANDBOX is not 'true'; refusing to run against production",
        allow_module_level=True,
    )


@pytest.mark.asyncio
async def test_account_balances_round_trip():
    from platforms.tastytrade_broker import TastytradeBroker

    broker = TastytradeBroker(is_test=True)
    try:
        account = await broker.get_account()
        assert account.equity >= 0
    finally:
        await broker.aclose()


@pytest.mark.asyncio
async def test_place_paper_csp_through_abstraction():
    from platforms.tastytrade_broker import TastytradeBroker

    broker = TastytradeBroker(is_test=True)
    try:
        chain = await broker.get_option_chain("F", option_type=OptionType.PUT)
        if not chain:
            pytest.skip("Tastytrade sandbox returned empty F chain")

        target_expiry = date.today() + timedelta(days=30)
        chain.sort(key=lambda c: (abs((c.expiration - target_expiry).days), -c.strike))
        contract = next((c for c in chain if c.strike <= 5.0), chain[0])

        client_id = f"wheelbot-it-{uuid.uuid4().hex[:8]}"
        placed = await broker.place_order(
            Order(
                account_id="tastytrade",
                symbol="F",
                order_type=OrderType.SELL_TO_OPEN,
                contract_symbol=contract.occ_symbol,
                strike=contract.strike,
                expiration=contract.expiration,
                option_type=OptionType.PUT,
                quantity=1,
                limit_price=0.01,
                status=OrderStatus.PENDING,
                placed_at=datetime.now(UTC).replace(tzinfo=None),
                client_order_id=client_id,
            )
        )
        assert placed.broker_order_id, "expected broker_order_id from Tastytrade"

        if placed.status == OrderStatus.REJECTED:
            pytest.skip("Tastytrade sandbox rejected order (likely options approval)")

        cutoff = datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=5)
        recent = await broker.get_orders_since(cutoff)
        assert any(o.broker_order_id == placed.broker_order_id for o in recent)

        await broker.cancel_order(placed.broker_order_id)
    finally:
        await broker.aclose()
