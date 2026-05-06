"""scripts/manual_close — close-out flow."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any
from unittest.mock import patch

import pytest

from core.models import (
    Order,
    OrderStatus,
    OrderType,
    OptionType,
    Position,
    PositionState,
    UniverseEntry,
    WheelCycle,
)
from execution.router import OrderRouter
from platforms.paper_broker import PaperBroker
from scripts.manual_close import _open_short_for, _proposal_for_position, _select_targets


def _utc() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


@pytest.mark.asyncio
async def test_select_targets_filters_to_closable_states(db_repos):
    now = _utc()
    for sym, state in [
        ("A", PositionState.CSP_OPEN),
        ("B", PositionState.SHARES_HELD),
        ("C", PositionState.IDLE),
        ("D", PositionState.MANUAL_INTERVENTION),
    ]:
        await db_repos.positions.insert(
            Position(account_id="t", symbol=sym, state=state, shares=0, state_changed_at=now)
        )
    targets = await _select_targets(db_repos, "t", None)
    syms = sorted(p.symbol for p in targets)
    assert syms == ["A", "B"]


@pytest.mark.asyncio
async def test_select_targets_by_symbol_returns_only_that_one(db_repos):
    now = _utc()
    for sym, state in [("A", PositionState.CSP_OPEN), ("B", PositionState.CSP_OPEN)]:
        await db_repos.positions.insert(
            Position(account_id="t", symbol=sym, state=state, shares=0, state_changed_at=now)
        )
    targets = await _select_targets(db_repos, "t", "A")
    assert [p.symbol for p in targets] == ["A"]


@pytest.mark.asyncio
async def test_open_short_returns_latest_filled_short_for_cycle(db_repos):
    now = _utc()
    cid = await db_repos.cycles.insert(
        WheelCycle(account_id="t", symbol="F", started_at=now - timedelta(days=5))
    )
    await db_repos.orders.insert(
        Order(
            account_id="t",
            symbol="F",
            cycle_id=cid,
            order_type=OrderType.SELL_TO_OPEN,
            contract_symbol="F250706P00009500",
            strike=9.5,
            expiration=date(2025, 7, 6),
            option_type=OptionType.PUT,
            quantity=1,
            limit_price=0.50,
            fill_price=0.50,
            status=OrderStatus.FILLED,
            placed_at=now - timedelta(days=4),
            client_order_id="wb-old-1",
        )
    )
    pos = Position(
        account_id="t",
        symbol="F",
        state=PositionState.CSP_OPEN,
        shares=0,
        current_cycle_id=cid,
        state_changed_at=now,
    )
    short = await _open_short_for(db_repos, pos)
    assert short is not None
    assert short.contract_symbol == "F250706P00009500"


def test_proposal_for_csp_open_returns_buy_to_close():
    now = _utc()
    pos = Position(
        account_id="t", symbol="F", state=PositionState.CSP_OPEN,
        shares=0, current_cycle_id=1, state_changed_at=now,
    )
    short = Order(
        account_id="t",
        symbol="F",
        order_type=OrderType.SELL_TO_OPEN,
        contract_symbol="F250706P00009500",
        strike=9.5,
        expiration=date(2025, 7, 6),
        option_type=OptionType.PUT,
        quantity=1,
        fill_price=0.50,
        status=OrderStatus.FILLED,
        placed_at=now,
        client_order_id="wb-1",
    )
    p = _proposal_for_position(pos, short)
    assert p is not None
    assert p.order_type == OrderType.BUY_TO_CLOSE
    assert p.contract.occ_symbol == "F250706P00009500"


def test_proposal_for_shares_held_returns_sell_to_close():
    now = _utc()
    pos = Position(
        account_id="t", symbol="F", state=PositionState.SHARES_HELD,
        shares=100, cost_basis=9.0, state_changed_at=now,
    )
    p = _proposal_for_position(pos, None)
    assert p is not None
    assert p.order_type == OrderType.SELL_TO_CLOSE
    assert p.quantity == 100
