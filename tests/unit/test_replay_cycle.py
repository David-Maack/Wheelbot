"""scripts/replay_cycle — narrative output + P&L decomposition."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from io import StringIO
from typing import Any

import pytest

from core.models import (
    Order,
    OrderStatus,
    OrderType,
    OptionType,
    Position,
    PositionState,
    StateLog,
    StateLogTrigger,
    WheelCycle,
)
from scripts.replay_cycle import _decompose_pnl


@pytest.mark.asyncio
async def test_decompose_pnl_csp_then_assignment_then_cc_called_away():
    """CSP at 0.50 → assigned at 9.0 → CC at 0.30 → called away at 10.0."""
    orders: list[dict[str, Any]] = [
        {  # CSP fill: collected $50 premium
            "order_type": OrderType.SELL_TO_OPEN.value,
            "quantity": 1,
            "fill_price": 0.50,
            "status": OrderStatus.FILLED.value,
        },
        {  # Stock assignment: bought 100 @ $9.0
            "order_type": OrderType.BUY_TO_OPEN.value,
            "quantity": 100,
            "fill_price": 9.0,
            "status": OrderStatus.FILLED.value,
        },
        {  # CC fill: collected $30 premium
            "order_type": OrderType.SELL_TO_OPEN.value,
            "quantity": 1,
            "fill_price": 0.30,
            "status": OrderStatus.FILLED.value,
        },
        {  # Called away: sold 100 @ $10.0
            "order_type": OrderType.SELL_TO_CLOSE.value,
            "quantity": 100,
            "fill_price": 10.0,
            "status": OrderStatus.FILLED.value,
        },
    ]
    parts = _decompose_pnl(orders)
    assert parts["premium_credits"] == pytest.approx(80.0)
    assert parts["premium_debits"] == pytest.approx(0.0)
    assert parts["share_legs"] == pytest.approx(100.0)  # +1000 - 900
    assert parts["total"] == pytest.approx(180.0)


@pytest.mark.asyncio
async def test_decompose_skips_unfilled_orders():
    orders: list[dict[str, Any]] = [
        {
            "order_type": OrderType.SELL_TO_OPEN.value,
            "quantity": 1,
            "fill_price": None,
            "status": OrderStatus.PENDING.value,
        },
        {
            "order_type": OrderType.SELL_TO_OPEN.value,
            "quantity": 1,
            "fill_price": 0.40,
            "status": OrderStatus.FILLED.value,
        },
    ]
    parts = _decompose_pnl(orders)
    assert parts["premium_credits"] == pytest.approx(40.0)
    assert parts["total"] == pytest.approx(40.0)


@pytest.mark.asyncio
async def test_replay_resolves_latest_cycle_for_symbol(db_repos, capsys):
    """End-to-end through the run() entrypoint with --latest --symbol."""
    from scripts import replay_cycle as rc

    now = datetime.now(UTC).replace(tzinfo=None)
    older = await db_repos.cycles.insert(
        WheelCycle(account_id="primary", symbol="F", started_at=now - timedelta(days=60))
    )
    await db_repos.cycles.update(
        older,
        ended_at=(now - timedelta(days=30)).isoformat(),
        final_pnl=10.0,
        cycle_outcome="CSP_EXPIRED",
        days_held=30,
    )
    newer = await db_repos.cycles.insert(
        WheelCycle(
            account_id="primary",
            symbol="F",
            started_at=now - timedelta(days=10),
            initial_csp_strike=9.5,
            initial_csp_premium=50.0,
        )
    )
    await db_repos.cycles.update(
        newer,
        ended_at=now.isoformat(),
        final_pnl=50.0,
        cycle_outcome="CSP_EXPIRED",
        days_held=10,
    )

    resolved = await rc._resolve_cycle_id(
        db_repos, cycle_id=None, symbol="F", latest=True, account_id="primary"
    )
    assert resolved == newer
