"""run_bot's roll executor — verifies BTC+STO on ROLL, BTC on CLOSE, no-op on LET_ASSIGN.

These cover the gap where the orchestrator returned a decision but nothing
ever placed the resulting orders.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from core.models import (
    OptionContract,
    OptionType,
    Order,
    OrderStatus,
    OrderType,
    Position,
    PositionState,
)
from scripts.run_bot import _execute_roll_action
from strategies.roll_advisor import RollAction, RollDecision
from strategies.roll_orchestrator import RollOutcome


def _utc():
    return datetime.now(UTC).replace(tzinfo=None)


def _short_contract():
    return OptionContract(
        underlying="F",
        occ_symbol="F250706P00009500",
        strike=9.5,
        expiration=date(2025, 7, 6),
        option_type=OptionType.PUT,
        bid=1.40, ask=1.60,
    )


def _new_contract():
    return OptionContract(
        underlying="F",
        occ_symbol="F250803P00009000",
        strike=9.0,
        expiration=date(2025, 8, 3),
        option_type=OptionType.PUT,
        bid=2.00, ask=2.10,
    )


def _short_order(qty: int = 1) -> Order:
    return Order(
        account_id="test",
        symbol="F",
        order_type=OrderType.SELL_TO_OPEN,
        contract_symbol="F250706P00009500",
        strike=9.5,
        expiration=date(2025, 7, 6),
        option_type=OptionType.PUT,
        quantity=qty,
        fill_price=0.50,
        status=OrderStatus.FILLED,
        placed_at=_utc(),
        client_order_id="wb-csp",
    )


def _position() -> Position:
    return Position(
        id=1,
        account_id="test", symbol="F",
        state=PositionState.CSP_OPEN,
        shares=0,
        current_cycle_id=1,
        state_changed_at=_utc(),
    )


class _StubRouter:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []  # (order_type, occ)

    async def place(self, proposal):
        self.calls.append((proposal.order_type.value, proposal.contract.occ_symbol))
        from types import SimpleNamespace
        return SimpleNamespace(placed=SimpleNamespace(broker_order_id="paper-x"))


@pytest.mark.asyncio
async def test_roll_executes_btc_then_sto():
    router = _StubRouter()
    outcome = RollOutcome(
        action=RollAction.ROLL,
        rule=RollDecision(
            action=RollAction.ROLL,
            rationale="credit roll",
            new_contract=_new_contract(),
            expected_credit_per_share=0.40,
        ),
        llm=None,
        halted=False,
        reason="rule_only",
    )
    await _execute_roll_action(
        router=router, outcome=outcome,
        position=_position(), short=_short_order(), short_contract=_short_contract(),
    )
    assert router.calls == [
        ("BUY_TO_CLOSE", "F250706P00009500"),
        ("SELL_TO_OPEN", "F250803P00009000"),
    ]


@pytest.mark.asyncio
async def test_close_executes_btc_only():
    router = _StubRouter()
    outcome = RollOutcome(
        action=RollAction.CLOSE,
        rule=RollDecision(action=RollAction.CLOSE, rationale="cut losses"),
        llm=None,
        halted=False,
        reason="rule_only",
    )
    await _execute_roll_action(
        router=router, outcome=outcome,
        position=_position(), short=_short_order(), short_contract=_short_contract(),
    )
    assert router.calls == [("BUY_TO_CLOSE", "F250706P00009500")]


@pytest.mark.asyncio
async def test_let_assign_is_a_noop():
    router = _StubRouter()
    outcome = RollOutcome(
        action=RollAction.LET_ASSIGN,
        rule=RollDecision(action=RollAction.LET_ASSIGN, rationale="take shares"),
        llm=None,
        halted=False,
        reason="rule_only",
    )
    await _execute_roll_action(
        router=router, outcome=outcome,
        position=_position(), short=_short_order(), short_contract=_short_contract(),
    )
    assert router.calls == []


@pytest.mark.asyncio
async def test_btc_router_failure_aborts_sto():
    """If the BTC leg raises, we don't proceed to STO (avoids a naked short)."""

    class _FailRouter:
        def __init__(self):
            self.calls = []

        async def place(self, proposal):
            self.calls.append(proposal.order_type.value)
            raise RuntimeError("broker down")

    router = _FailRouter()
    outcome = RollOutcome(
        action=RollAction.ROLL,
        rule=RollDecision(
            action=RollAction.ROLL,
            rationale="credit roll",
            new_contract=_new_contract(),
            expected_credit_per_share=0.40,
        ),
        llm=None,
        halted=False,
        reason="rule_only",
    )
    await _execute_roll_action(
        router=router, outcome=outcome,  # type: ignore[arg-type]
        position=_position(), short=_short_order(), short_contract=_short_contract(),
    )
    assert router.calls == ["BUY_TO_CLOSE"]  # STO not reached


@pytest.mark.asyncio
async def test_quantity_propagates_from_short_order():
    """A 3-contract short rolls 3 contracts, not 1."""
    router = _StubRouter()
    outcome = RollOutcome(
        action=RollAction.ROLL,
        rule=RollDecision(
            action=RollAction.ROLL,
            rationale="credit roll",
            new_contract=_new_contract(),
            expected_credit_per_share=0.40,
        ),
        llm=None,
        halted=False,
        reason="rule_only",
    )

    # Capture proposals to inspect quantity.
    captured = []

    class _Capture:
        async def place(self, proposal):
            captured.append(proposal.quantity)
            from types import SimpleNamespace
            return SimpleNamespace(placed=SimpleNamespace())

    await _execute_roll_action(
        router=_Capture(), outcome=outcome,  # type: ignore[arg-type]
        position=_position(), short=_short_order(qty=3), short_contract=_short_contract(),
    )
    assert captured == [3, 3]
