"""PMCC close orchestrator — short-close triggers + long-roll DTE/delta triggers.

The long-roll delta trigger (2026-07-06 sprint) rolls the long when its
current BS delta decays below `long_roll_delta` — below ~0.70 the long stops
behaving like stock and the covered-call economics break. DTE stays the
quote-outage backstop.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from core.models import (
    OptionType,
    Order,
    OrderStatus,
    OrderType,
    Position,
    PositionState,
    Quote,
    WheelCycle,
)
from core.strategies import StrategyDefinition, _validate
from platforms.paper_broker import PaperBroker
from strategies.pmcc_close import propose_close_for_symbol

TODAY = date(2026, 7, 6)


def _config() -> dict:
    return {"account": {"id": "test", "broker": "paper"}}


def _utc() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _strategy(**overrides: Any) -> StrategyDefinition:
    params: dict[str, Any] = {
        "long_dte_min": 90, "long_dte_max": 180, "long_delta_target": 0.80,
        "short_dte_min": 7, "short_dte_max": 30, "short_delta_target": 0.27,
        "profit_close_pct_short": 50, "short_time_close_dte": 1,
        "long_roll_dte": 60, "long_roll_delta": 0.70,
        "max_capital_per_position_usd": 10000, "ivr_min": 0,
    }
    params.update(overrides)
    return StrategyDefinition(
        id="pmcc", display_name="PMCC", type="pmcc",
        enabled=True, max_concurrent=3, params=params,
    )


def _occ(strike: float, exp: date) -> str:
    return f"AAPL{exp:%y%m%d}C{int(strike * 1000):08d}"


async def _seed_cycle_and_position(db_repos, state: PositionState) -> int:
    cycle_id = await db_repos.cycles.insert(
        WheelCycle(account_id="test", symbol="AAPL", strategy_id="pmcc",
                   started_at=_utc(), n_orders=1)
    )
    await db_repos.positions.insert(
        Position(account_id="test", symbol="AAPL", strategy_id="pmcc",
                 state=state, shares=0, current_cycle_id=cycle_id,
                 state_changed_at=_utc())
    )
    return cycle_id


async def _seed_filled_order(
    db_repos, cycle_id: int, order_type: OrderType, strike: float,
    exp: date, fill_price: float, client_id: str,
) -> Order:
    order = Order(
        account_id="test", symbol="AAPL", strategy_id="pmcc",
        cycle_id=cycle_id, order_type=order_type,
        contract_symbol=_occ(strike, exp), strike=strike, expiration=exp,
        option_type=OptionType.CALL, quantity=1, fill_price=fill_price,
        status=OrderStatus.FILLED, placed_at=_utc(), client_order_id=client_id,
    )
    await db_repos.orders.insert(order)
    return order


def _seed_quotes(broker: PaperBroker, contract_symbol: str, opt_mid: float,
                 spot: float = 150.0) -> None:
    broker.seed_quote(Quote(symbol="AAPL", bid=spot - 0.05, ask=spot + 0.05))
    broker.seed_quote(Quote(symbol=contract_symbol,
                            bid=opt_mid - 0.05, ask=opt_mid + 0.05))


# -- long roll: DTE trigger --------------------------------------------------


@pytest.mark.asyncio
async def test_long_roll_fires_below_dte(db_repos):
    broker = PaperBroker()
    cycle_id = await _seed_cycle_and_position(db_repos, PositionState.PMCC_LONG_OPEN)
    exp = TODAY + timedelta(days=45)  # < long_roll_dte 60
    order = await _seed_filled_order(
        db_repos, cycle_id, OrderType.BUY_TO_OPEN, 120.0, exp, 33.0, "long-1")
    _seed_quotes(broker, order.contract_symbol, 31.0)

    p = await propose_close_for_symbol(
        broker, db_repos, "AAPL", _config(), today=TODAY, strategy=_strategy())
    assert p is not None
    assert p.order_type == OrderType.SELL_TO_CLOSE
    assert p.trigger_reason == "pmcc_roll_dte"


@pytest.mark.asyncio
async def test_long_roll_quiet_when_healthy(db_repos):
    """DTE above the floor AND delta still deep ITM → no proposal."""
    broker = PaperBroker()
    cycle_id = await _seed_cycle_and_position(db_repos, PositionState.PMCC_LONG_OPEN)
    exp = TODAY + timedelta(days=120)
    # Strike 120, spot 150, mid 31 → BS delta ~0.9, well above 0.70.
    order = await _seed_filled_order(
        db_repos, cycle_id, OrderType.BUY_TO_OPEN, 120.0, exp, 33.0, "long-1")
    _seed_quotes(broker, order.contract_symbol, 31.0)

    p = await propose_close_for_symbol(
        broker, db_repos, "AAPL", _config(), today=TODAY, strategy=_strategy())
    assert p is None


# -- long roll: delta trigger ------------------------------------------------


@pytest.mark.asyncio
async def test_long_roll_fires_on_delta_decay(db_repos):
    """DTE healthy but the long decayed to ATM (delta ~0.55) → delta roll."""
    broker = PaperBroker()
    cycle_id = await _seed_cycle_and_position(db_repos, PositionState.PMCC_LONG_OPEN)
    exp = TODAY + timedelta(days=120)
    # Strike 150 = spot: ATM call delta ~0.55 < 0.70 threshold.
    order = await _seed_filled_order(
        db_repos, cycle_id, OrderType.BUY_TO_OPEN, 150.0, exp, 12.0, "long-1")
    _seed_quotes(broker, order.contract_symbol, 9.0)

    p = await propose_close_for_symbol(
        broker, db_repos, "AAPL", _config(), today=TODAY, strategy=_strategy())
    assert p is not None
    assert p.order_type == OrderType.SELL_TO_CLOSE
    assert p.trigger_reason == "pmcc_roll_delta"


@pytest.mark.asyncio
async def test_long_roll_delta_off_when_param_absent(db_repos):
    """Same decayed position, but long_roll_delta unset → feature off, no roll."""
    broker = PaperBroker()
    cycle_id = await _seed_cycle_and_position(db_repos, PositionState.PMCC_LONG_OPEN)
    exp = TODAY + timedelta(days=120)
    order = await _seed_filled_order(
        db_repos, cycle_id, OrderType.BUY_TO_OPEN, 150.0, exp, 12.0, "long-1")
    _seed_quotes(broker, order.contract_symbol, 9.0)

    strategy = _strategy(long_roll_delta=None)
    p = await propose_close_for_symbol(
        broker, db_repos, "AAPL", _config(), today=TODAY, strategy=strategy)
    assert p is None


@pytest.mark.asyncio
async def test_long_roll_delta_unavailable_skips_tick(db_repos):
    """No quotes seeded → delta unreadable → skip (DTE backstop still governs)."""
    broker = PaperBroker()
    cycle_id = await _seed_cycle_and_position(db_repos, PositionState.PMCC_LONG_OPEN)
    exp = TODAY + timedelta(days=120)
    await _seed_filled_order(
        db_repos, cycle_id, OrderType.BUY_TO_OPEN, 150.0, exp, 12.0, "long-1")
    # No quotes at all: underlying quote raises BrokerUnavailable inside.

    p = await propose_close_for_symbol(
        broker, db_repos, "AAPL", _config(), today=TODAY, strategy=_strategy())
    assert p is None


# -- short close regression --------------------------------------------------


@pytest.mark.asyncio
async def test_short_close_profit_trigger_still_fires(db_repos):
    """PMCC_BOTH_OPEN + short mid decayed past 50% of premium → BUY_TO_CLOSE."""
    broker = PaperBroker()
    cycle_id = await _seed_cycle_and_position(db_repos, PositionState.PMCC_BOTH_OPEN)
    exp = TODAY + timedelta(days=14)
    short = await _seed_filled_order(
        db_repos, cycle_id, OrderType.SELL_TO_OPEN, 160.0, exp, 1.20, "short-1")
    _seed_quotes(broker, short.contract_symbol, 0.50)  # 0.50 <= 0.5 * 1.20

    p = await propose_close_for_symbol(
        broker, db_repos, "AAPL", _config(), today=TODAY, strategy=_strategy())
    assert p is not None
    assert p.order_type == OrderType.BUY_TO_CLOSE
    assert p.trigger_reason == "pmcc_short_profit"


# -- load-time validation ----------------------------------------------------


def test_validate_rejects_roll_delta_at_or_above_entry_delta():
    bad = _strategy(long_roll_delta=0.80)  # == long_delta_target
    with pytest.raises(ValueError, match="long_roll_delta"):
        _validate([bad])


def test_validate_accepts_roll_delta_below_entry_delta():
    _validate([_strategy()])  # 0.70 < 0.80 — no raise
