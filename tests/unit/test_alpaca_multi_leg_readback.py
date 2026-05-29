"""AlpacaBroker._to_order handles MLEG parent orders without raw.symbol.

Regression for the bug that flagged 11 positions BROKER_DOWN in prod:
Alpaca returns multi-leg parent orders with symbol=None (symbols live on
legs). Our Order.symbol is required, so the reconciler crashed on every
tick once any MLEG order existed in the account.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core.models import OrderStatus, OrderType


def _stub_alpaca_imports():
    """Make AlpacaBroker importable without the alpaca-py SDK installed."""
    import sys

    fake_alpaca = SimpleNamespace()
    fake_trading_enums = SimpleNamespace(
        AssetClass=SimpleNamespace(US_OPTION="us_option", US_EQUITY="us_equity"),
        OrderClass=SimpleNamespace(MLEG="mleg", SIMPLE="simple"),
        OrderSide=SimpleNamespace(BUY="buy", SELL="sell"),
        TimeInForce=SimpleNamespace(DAY="day"),
        PositionIntent=SimpleNamespace(
            SELL_TO_OPEN="sell_to_open",
            BUY_TO_OPEN="buy_to_open",
            SELL_TO_CLOSE="sell_to_close",
            BUY_TO_CLOSE="buy_to_close",
        ),
        QueryOrderStatus=SimpleNamespace(ALL="all"),
        ContractType=SimpleNamespace(PUT="put", CALL="call"),
    )
    return fake_trading_enums


def _make_leg(symbol: str, side: str, position_intent: str):
    return SimpleNamespace(symbol=symbol, side=side, position_intent=position_intent)


def _make_mleg_parent(*, legs, filled_avg_price=None, status="filled"):
    return SimpleNamespace(
        id="mleg-abc-123",
        client_order_id="wb-deadbeef00000000",
        symbol=None,                       # ← the bug: parent has no symbol
        asset_class=None,                  # MLEG parent isn't a US_OPTION
        order_class="mleg",
        legs=legs,
        side=None,
        position_intent=None,
        qty="2",
        limit_price="1.42",
        filled_avg_price=filled_avg_price,
        status=status,
        submitted_at=datetime.now(UTC) - timedelta(minutes=5),
        filled_at=datetime.now(UTC) if filled_avg_price is not None else None,
        model_dump=lambda mode=None: {"order_class": "mleg"},
    )


@pytest.fixture
def broker(monkeypatch):
    """Construct AlpacaBroker without calling the SDK constructors."""
    # Avoid importing alpaca-py at all by stubbing the trading.enums module
    # before AlpacaBroker._to_order tries to import OrderClass/AssetClass.
    import sys

    fake_enums = _stub_alpaca_imports()
    fake_module = SimpleNamespace(
        AssetClass=fake_enums.AssetClass,
        OrderClass=fake_enums.OrderClass,
        OrderSide=fake_enums.OrderSide,
        TimeInForce=fake_enums.TimeInForce,
        PositionIntent=fake_enums.PositionIntent,
        QueryOrderStatus=fake_enums.QueryOrderStatus,
        ContractType=fake_enums.ContractType,
    )
    monkeypatch.setitem(sys.modules, "alpaca.trading.enums", fake_module)

    from platforms.alpaca_broker import AlpacaBroker

    inst = AlpacaBroker.__new__(AlpacaBroker)
    inst._account_id = "primary"
    return inst


def test_to_order_handles_mleg_parent_with_none_symbol(broker):
    """The reconciler-failure regression: MLEG parent with symbol=None must
    map to a valid Order tagged with the underlying from the first leg."""
    legs = [
        _make_leg("TSLA260612P00400000", "sell", "sell_to_open"),
        _make_leg("TSLA260612P00395000", "buy", "buy_to_open"),
    ]
    raw = _make_mleg_parent(legs=legs, filled_avg_price="1.42", status="filled")
    order = broker._to_order(raw)
    assert order.symbol == "TSLA"
    assert order.order_type == OrderType.MULTI_LEG_OPEN
    assert order.status == OrderStatus.FILLED
    assert order.quantity == 2
    # Alpaca reports MLEG fill_price as debit-positive; we flip to credit-positive.
    assert order.fill_price == -1.42
    assert order.limit_price == -1.42


def test_to_order_mleg_close_inferred_when_all_legs_close(broker):
    legs = [
        _make_leg("TSLA260612P00400000", "buy", "buy_to_close"),
        _make_leg("TSLA260612P00395000", "sell", "sell_to_close"),
    ]
    raw = _make_mleg_parent(legs=legs, filled_avg_price="0.30", status="filled")
    order = broker._to_order(raw)
    assert order.order_type == OrderType.MULTI_LEG_CLOSE


def test_to_order_legacy_single_leg_path_unchanged(broker):
    """Non-MLEG orders still work the same way."""
    raw = SimpleNamespace(
        id="alp-123",
        client_order_id="wb-abc",
        symbol="F250620P00010000",
        asset_class="us_option",
        order_class="simple",
        legs=None,
        side="sell",
        position_intent="sell_to_open",
        qty="1",
        limit_price="0.50",
        filled_avg_price=None,
        status="new",
        submitted_at=datetime.now(UTC),
        filled_at=None,
        model_dump=lambda mode=None: {},
    )
    order = broker._to_order(raw)
    assert order.symbol == "F"
    assert order.order_type == OrderType.SELL_TO_OPEN
    assert order.contract_symbol == "F250620P00010000"


def test_to_order_mleg_pending_no_fill_price(broker):
    """A pending MLEG order has no filled_avg_price — don't crash."""
    legs = [
        _make_leg("MSFT260618P00395000", "sell", "sell_to_open"),
        _make_leg("MSFT260618P00390000", "buy", "buy_to_open"),
    ]
    raw = _make_mleg_parent(legs=legs, filled_avg_price=None, status="new")
    order = broker._to_order(raw)
    assert order.symbol == "MSFT"
    assert order.status == OrderStatus.PENDING
    assert order.fill_price is None
    assert order.filled_at is None


class _FakeReq:
    """Stand-in for Alpaca's *Request models: records kwargs, dumps to dict."""

    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def model_dump(self, mode=None):
        return {k: str(v) for k, v in self._kwargs.items()}


def _stub_submit_modules(monkeypatch):
    """Stub the alpaca request/exception modules place_multi_leg_order imports."""
    import sys

    monkeypatch.setitem(
        sys.modules,
        "alpaca.common.exceptions",
        SimpleNamespace(APIError=type("APIError", (Exception,), {})),
    )
    monkeypatch.setitem(
        sys.modules,
        "alpaca.trading.requests",
        SimpleNamespace(
            LimitOrderRequest=_FakeReq,
            MarketOrderRequest=_FakeReq,
            OptionLegRequest=_FakeReq,
        ),
    )


async def test_place_multi_leg_order_negates_fill_price_at_submit(broker, monkeypatch):
    """Finding #2 regression: the SUBMIT path must negate filled_avg_price the
    same way the READ-BACK path does. Alpaca reports a credit spread's fill as a
    debit-positive number; storing it un-negated on a fast fill silently flips
    spread P&L depending on which path observed the order first."""
    from datetime import UTC, datetime

    from core.models import OptionType, OrderType

    _stub_submit_modules(monkeypatch)

    # A credit spread that fills instantly: Alpaca returns filled_avg_price as a
    # POSITIVE debit-style number (0.42). Our convention is credit-positive, so
    # the stored Order must carry -0.42.
    placed = SimpleNamespace(
        id="mleg-submit-1",
        status="filled",
        submitted_at=datetime.now(UTC),
        filled_at=datetime.now(UTC),
        filled_avg_price="0.42",
        model_dump=lambda mode=None: {"id": "mleg-submit-1"},
    )
    broker._trading = SimpleNamespace(submit_order=lambda req: placed)

    from core.models import OrderLeg as Leg

    legs = [
        Leg(
            contract_symbol="TSLA260612P00400000",
            underlying="TSLA",
            option_type=OptionType.PUT,
            strike=400.0,
            expiration=__import__("datetime").date(2026, 6, 12),
            action=OrderType.SELL_TO_OPEN,
        ),
        Leg(
            contract_symbol="TSLA260612P00395000",
            underlying="TSLA",
            option_type=OptionType.PUT,
            strike=395.0,
            expiration=__import__("datetime").date(2026, 6, 12),
            action=OrderType.BUY_TO_OPEN,
        ),
    ]

    order = await broker.place_multi_leg_order(
        underlying="TSLA",
        legs=legs,
        quantity=1,
        limit_price=0.50,           # our credit-positive convention
        client_order_id="wb-submit-test",
    )

    # fill negated to credit-positive convention, matching _to_order_multi_leg
    assert order.fill_price == -0.42
    # limit kept in our convention on the returned Order (negated only on the wire)
    assert order.limit_price == 0.50
    assert order.order_type == OrderType.MULTI_LEG_OPEN


async def test_place_multi_leg_order_zero_fill_not_dropped(broker, monkeypatch):
    """Finding #3 regression: a genuine 0.0 fill must survive as 0.0, not None
    (the old `if placed.filled_avg_price else None` falsy check ate zeros)."""
    from datetime import UTC, datetime

    from core.models import OptionType, OrderType
    from core.models import OrderLeg as Leg

    _stub_submit_modules(monkeypatch)

    placed = SimpleNamespace(
        id="mleg-submit-zero",
        status="filled",
        submitted_at=datetime.now(UTC),
        filled_at=datetime.now(UTC),
        filled_avg_price="0",          # a zero fill — must not become None
        model_dump=lambda mode=None: {"id": "mleg-submit-zero"},
    )
    broker._trading = SimpleNamespace(submit_order=lambda req: placed)

    legs = [
        Leg(
            contract_symbol="TSLA260612P00400000",
            underlying="TSLA",
            option_type=OptionType.PUT,
            strike=400.0,
            expiration=__import__("datetime").date(2026, 6, 12),
            action=OrderType.SELL_TO_OPEN,
        ),
    ]

    order = await broker.place_multi_leg_order(
        underlying="TSLA",
        legs=legs,
        quantity=1,
        limit_price=None,
        client_order_id="wb-submit-zero",
    )
    assert order.fill_price == 0.0
    assert order.fill_price is not None


def test_is_leg_child_skips_orders_without_our_client_order_id(broker):
    """Defensive filter in get_orders_since: legs (and manual broker orders)
    are skipped because they don't carry our `wb-` client_order_id prefix."""
    from platforms.alpaca_broker import _is_leg_child

    # Our orders carry wb-... — keep them.
    ours = SimpleNamespace(client_order_id="wb-abc123", order_class="simple")
    assert _is_leg_child(ours) is False

    # MLEG parent we placed — keep it.
    parent = SimpleNamespace(client_order_id="wb-deadbeef", order_class="mleg")
    assert _is_leg_child(parent) is False

    # Leg child — no client_order_id we recognize.
    leg = SimpleNamespace(client_order_id=None, order_class="simple")
    assert _is_leg_child(leg) is True

    # Manual order placed by a human at Alpaca UI — skip (not ours).
    manual = SimpleNamespace(client_order_id="some-other-id", order_class="simple")
    assert _is_leg_child(manual) is True
