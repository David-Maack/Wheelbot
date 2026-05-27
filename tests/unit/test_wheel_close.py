"""Wheel profit-close orchestrator — CSP and CC threshold logic + roll handling."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from core.models import (
    Order,
    OrderStatus,
    OrderType,
    OptionType,
    Position,
    PositionState,
    Quote,
    WheelCycle,
)
from core.strategies import StrategyDefinition
from platforms.paper_broker import PaperBroker
from strategies.wheel_close import (
    propose_all_closes,
    propose_close_for_position,
)


# -- helpers ----------------------------------------------------------------


def _strategy(**params_overrides: Any) -> StrategyDefinition:
    base_params: dict[str, Any] = {
        "csp_profit_close_pct": 50,
        "cc_profit_close_pct": 50,
    }
    base_params.update(params_overrides)
    return StrategyDefinition(
        id="monthly_wheel",
        display_name="Monthly Wheel",
        type="wheel",
        enabled=True,
        max_concurrent=4,
        params=base_params,
    )


async def _seed_open_csp(
    db_repos, *, fill_price: float, symbol: str = "F", contract: str = "F250706P00010000"
) -> tuple[Position, int]:
    """Persist a CSP_OPEN position with a FILLED SELL_TO_OPEN order on a cycle."""
    now = datetime.now(UTC).replace(tzinfo=None)
    cycle_id = await db_repos.cycles.insert(
        WheelCycle(
            account_id="test",
            symbol=symbol,
            strategy_id="monthly_wheel",
            started_at=now,
            initial_csp_strike=10.0,
            initial_csp_premium=fill_price,
            initial_capital_at_risk=1000.0,
        )
    )
    await db_repos.orders.insert(
        Order(
            account_id="test",
            symbol=symbol,
            strategy_id="monthly_wheel",
            cycle_id=cycle_id,
            order_type=OrderType.SELL_TO_OPEN,
            contract_symbol=contract,
            strike=10.0,
            expiration=date(2025, 6, 1) + timedelta(days=35),
            option_type=OptionType.PUT,
            quantity=1,
            limit_price=fill_price,
            fill_price=fill_price,
            status=OrderStatus.FILLED,
            placed_at=now,
            filled_at=now,
        )
    )
    position = Position(
        account_id="test",
        symbol=symbol,
        strategy_id="monthly_wheel",
        state=PositionState.CSP_OPEN,
        shares=0,
        current_cycle_id=cycle_id,
        state_changed_at=now,
    )
    pos_id = await db_repos.positions.insert(position)
    position = position.model_copy(update={"id": pos_id})
    return position, cycle_id


async def _seed_open_cc(
    db_repos, *, fill_price: float, symbol: str = "F", contract: str = "F250706C00012000"
) -> Position:
    """Persist a CC_OPEN position with a FILLED SELL_TO_OPEN call order."""
    now = datetime.now(UTC).replace(tzinfo=None)
    cycle_id = await db_repos.cycles.insert(
        WheelCycle(
            account_id="test",
            symbol=symbol,
            strategy_id="monthly_wheel",
            started_at=now,
            initial_csp_strike=10.0,
            initial_csp_premium=0.40,
            initial_capital_at_risk=1000.0,
        )
    )
    await db_repos.orders.insert(
        Order(
            account_id="test",
            symbol=symbol,
            strategy_id="monthly_wheel",
            cycle_id=cycle_id,
            order_type=OrderType.SELL_TO_OPEN,
            contract_symbol=contract,
            strike=12.0,
            expiration=date(2025, 6, 1) + timedelta(days=35),
            option_type=OptionType.CALL,
            quantity=1,
            limit_price=fill_price,
            fill_price=fill_price,
            status=OrderStatus.FILLED,
            placed_at=now,
            filled_at=now,
        )
    )
    position = Position(
        account_id="test",
        symbol=symbol,
        strategy_id="monthly_wheel",
        state=PositionState.CC_OPEN,
        shares=100,
        cost_basis=9.60,
        current_cycle_id=cycle_id,
        state_changed_at=now,
    )
    pos_id = await db_repos.positions.insert(position)
    return position.model_copy(update={"id": pos_id})


# -- CSP profit-close --------------------------------------------------------


@pytest.mark.asyncio
async def test_csp_close_fires_when_mid_drops_below_threshold(db_repos):
    position, _ = await _seed_open_csp(db_repos, fill_price=1.00)
    broker = PaperBroker()
    # Mid 0.45 → 55% profit → under target 0.50 = (1 - 0.50) × 1.00.
    broker.seed_quote(Quote(symbol="F250706P00010000", bid=0.44, ask=0.46))

    proposal = await propose_close_for_position(
        broker, db_repos, position, strategy=_strategy(),
    )
    assert proposal is not None
    assert proposal.order_type == OrderType.BUY_TO_CLOSE
    assert proposal.contract.option_type == OptionType.PUT
    assert proposal.quantity == 1
    assert "wheel_close" in proposal.rationale
    assert "profit" in proposal.rationale


@pytest.mark.asyncio
async def test_csp_close_skips_when_mid_above_threshold(db_repos):
    position, _ = await _seed_open_csp(db_repos, fill_price=1.00)
    broker = PaperBroker()
    # Mid 0.60 → still 40% profit → above target 0.50 threshold.
    broker.seed_quote(Quote(symbol="F250706P00010000", bid=0.59, ask=0.61))

    proposal = await propose_close_for_position(
        broker, db_repos, position, strategy=_strategy(),
    )
    assert proposal is None


@pytest.mark.asyncio
async def test_csp_close_skips_when_no_quote_available(db_repos):
    position, _ = await _seed_open_csp(db_repos, fill_price=1.00)
    broker = PaperBroker()  # no quote seeded

    proposal = await propose_close_for_position(
        broker, db_repos, position, strategy=_strategy(),
    )
    assert proposal is None


# -- CC profit-close ---------------------------------------------------------


@pytest.mark.asyncio
async def test_cc_close_fires_when_mid_drops_below_threshold(db_repos):
    position = await _seed_open_cc(db_repos, fill_price=0.80)
    broker = PaperBroker()
    # Mid 0.35 → 56% profit → under target 0.40 = (1 - 0.50) × 0.80.
    broker.seed_quote(Quote(symbol="F250706C00012000", bid=0.34, ask=0.36))

    proposal = await propose_close_for_position(
        broker, db_repos, position, strategy=_strategy(),
    )
    assert proposal is not None
    assert proposal.order_type == OrderType.BUY_TO_CLOSE
    assert proposal.contract.option_type == OptionType.CALL
    assert "wheel_close" in proposal.rationale
    assert "profit" in proposal.rationale


# -- Threshold lookup --------------------------------------------------------


@pytest.mark.asyncio
async def test_csp_uses_csp_profit_close_pct_over_legacy(db_repos):
    """csp_profit_close_pct: 70 must override legacy profit_close_pct: 50."""
    position, _ = await _seed_open_csp(db_repos, fill_price=1.00)
    broker = PaperBroker()
    # Mid 0.45 → 55% profit. At 50% threshold this would fire; at 70% it shouldn't.
    broker.seed_quote(Quote(symbol="F250706P00010000", bid=0.44, ask=0.46))

    proposal = await propose_close_for_position(
        broker, db_repos, position,
        strategy=_strategy(profit_close_pct=50, csp_profit_close_pct=70),
    )
    assert proposal is None  # 70% threshold not met yet


@pytest.mark.asyncio
async def test_falls_back_to_legacy_profit_close_pct(db_repos):
    """When only the legacy profit_close_pct is set, both CSP and CC honor it."""
    position, _ = await _seed_open_csp(db_repos, fill_price=1.00)
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F250706P00010000", bid=0.44, ask=0.46))

    strategy_legacy = StrategyDefinition(
        id="monthly_wheel",
        display_name="Monthly Wheel",
        type="wheel",
        enabled=True,
        max_concurrent=4,
        params={"profit_close_pct": 50},  # legacy only
    )
    proposal = await propose_close_for_position(
        broker, db_repos, position, strategy=strategy_legacy,
    )
    assert proposal is not None  # 55% profit clears legacy 50% threshold


# -- Rolled position uses latest short premium --------------------------------


@pytest.mark.asyncio
async def test_uses_latest_short_premium_for_rolled_cycle(db_repos):
    """A rolled cycle has multiple SELL_TO_OPEN orders; the threshold compares
    against the LATEST short's premium, not the cycle's initial premium."""
    position, cycle_id = await _seed_open_csp(db_repos, fill_price=1.00)
    # Insert a second, later SELL_TO_OPEN on the same cycle (the roll) at $0.40.
    now = datetime.now(UTC).replace(tzinfo=None)
    later = now + timedelta(hours=1)
    await db_repos.orders.insert(
        Order(
            account_id="test",
            symbol="F",
            strategy_id="monthly_wheel",
            cycle_id=cycle_id,
            order_type=OrderType.SELL_TO_OPEN,
            contract_symbol="F250713P00009500",  # rolled to a different strike
            strike=9.5,
            expiration=date(2025, 6, 1) + timedelta(days=42),
            option_type=OptionType.PUT,
            quantity=1,
            limit_price=0.40,
            fill_price=0.40,
            status=OrderStatus.FILLED,
            placed_at=later,
            filled_at=later,
        )
    )
    broker = PaperBroker()
    # Quote the LATEST contract at 0.18 → 55% profit on $0.40 = under target $0.20.
    # If the orchestrator naively used the ORIGINAL 1.00 premium, the target
    # would be 0.50 and the trigger would NOT fire at 0.18 — but the latest is
    # the right reference, so it must fire.
    broker.seed_quote(Quote(symbol="F250713P00009500", bid=0.17, ask=0.19))

    proposal = await propose_close_for_position(
        broker, db_repos, position, strategy=_strategy(),
    )
    assert proposal is not None
    # Confirms we're using the rolled contract, not the original.
    assert proposal.contract.occ_symbol == "F250713P00009500"


# -- Walker ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_propose_all_closes_walks_active_wheel_positions(db_repos):
    """Two CSP_OPEN positions: one at threshold, one not. Walker proposes once."""
    pos_a, _ = await _seed_open_csp(
        db_repos, fill_price=1.00, symbol="F", contract="F250706P00010000",
    )
    pos_b, _ = await _seed_open_csp(
        db_repos, fill_price=0.50, symbol="G", contract="G250706P00010000",
    )
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F250706P00010000", bid=0.44, ask=0.46))  # 55% — fires
    broker.seed_quote(Quote(symbol="G250706P00010000", bid=0.40, ask=0.42))  # 18% — skip

    config = {"account": {"id": "test"}}
    proposals = await propose_all_closes(
        broker, db_repos, config, strategy=_strategy(),
    )
    assert len(proposals) == 1
    assert proposals[0].symbol == "F"


@pytest.mark.asyncio
async def test_csp_time_close_fires_when_dte_below_threshold(db_repos):
    """Sprint 13 sub-sprint 2: time-close on a CSP when DTE ≤ time_close_dte."""
    position, _ = await _seed_open_csp(db_repos, fill_price=1.00)
    broker = PaperBroker()
    # Mid 0.80 — well above profit target ($0.50), so profit trigger does NOT fire.
    # Time trigger should fire because DTE will be 5 (under 21).
    broker.seed_quote(Quote(symbol="F250706P00010000", bid=0.79, ask=0.81))

    # Today = expiration - 5 days → DTE = 5.
    today = date(2025, 6, 1) + timedelta(days=35 - 5)
    proposal = await propose_close_for_position(
        broker, db_repos, position,
        today=today, strategy=_strategy(time_close_dte=21),
    )
    assert proposal is not None
    assert proposal.order_type == OrderType.BUY_TO_CLOSE
    assert "time_close" in proposal.rationale
    assert "profit" not in proposal.rationale  # only time trigger fired


@pytest.mark.asyncio
async def test_cc_time_close_fires_when_dte_below_threshold(db_repos):
    """Time-close also works on a CC position."""
    position = await _seed_open_cc(db_repos, fill_price=0.80)
    broker = PaperBroker()
    # Mid 0.60 — above profit target ($0.40); profit doesn't fire.
    broker.seed_quote(Quote(symbol="F250706C00012000", bid=0.59, ask=0.61))

    today = date(2025, 6, 1) + timedelta(days=35 - 10)  # DTE = 10
    proposal = await propose_close_for_position(
        broker, db_repos, position,
        today=today, strategy=_strategy(time_close_dte=21),
    )
    assert proposal is not None
    assert "time_close" in proposal.rationale


@pytest.mark.asyncio
async def test_time_close_does_not_fire_when_dte_above_threshold(db_repos):
    """DTE well above threshold + no profit trigger → no close proposal."""
    position, _ = await _seed_open_csp(db_repos, fill_price=1.00)
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F250706P00010000", bid=0.79, ask=0.81))

    # Today = expiration - 30 days → DTE = 30, above threshold 21.
    today = date(2025, 6, 1) + timedelta(days=35 - 30)
    proposal = await propose_close_for_position(
        broker, db_repos, position,
        today=today, strategy=_strategy(time_close_dte=21),
    )
    assert proposal is None


@pytest.mark.asyncio
async def test_time_close_disabled_when_param_unset(db_repos):
    """No time_close_dte in params (e.g. weekly_wheel) → time trigger never fires."""
    position, _ = await _seed_open_csp(db_repos, fill_price=1.00)
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F250706P00010000", bid=0.79, ask=0.81))

    # DTE 1 — well past the standard 21-day threshold.
    today = date(2025, 6, 1) + timedelta(days=35 - 1)
    strategy_no_time_close = StrategyDefinition(
        id="weekly_wheel",
        display_name="Weekly Wheel",
        type="wheel",
        enabled=True,
        max_concurrent=4,
        params={"csp_profit_close_pct": 50},  # no time_close_dte at all
    )
    proposal = await propose_close_for_position(
        broker, db_repos, position,
        today=today, strategy=strategy_no_time_close,
    )
    # Profit didn't trigger (0.80 > 0.50 target) and time is unset → no proposal.
    assert proposal is None


@pytest.mark.asyncio
async def test_csp_stop_loss_fires_at_threshold(db_repos):
    """Sprint 14: CSP stop-loss when current mid hits 2× original premium.

    Regression for the 2026-05-27 KMI canary — premium $0.46, current $1.35
    (2.93× of original) → with csp_stop_loss_mult=2.0, mid ≥ $0.92 fires
    the close immediately."""
    position, _ = await _seed_open_csp(db_repos, fill_price=0.46)
    broker = PaperBroker()
    # Mid 0.95 → 2.07× original → exceeds 2.0 threshold.
    # Not in profit (debit > target $0.23) and DTE 35 so time-close inactive.
    broker.seed_quote(Quote(symbol="F250706P00010000", bid=0.94, ask=0.96))

    proposal = await propose_close_for_position(
        broker, db_repos, position,
        today=date(2025, 6, 1),
        strategy=_strategy(csp_stop_loss_mult=2.0),
    )
    assert proposal is not None
    assert proposal.order_type == OrderType.BUY_TO_CLOSE
    assert "stop_loss" in proposal.rationale
    assert "profit" not in proposal.rationale


@pytest.mark.asyncio
async def test_csp_stop_loss_does_not_fire_below_threshold(db_repos):
    """Mid below the 2× threshold and not in profit → no proposal."""
    position, _ = await _seed_open_csp(db_repos, fill_price=0.46)
    broker = PaperBroker()
    # Mid 0.80 → 1.74× — below 2.0 threshold, above profit target.
    broker.seed_quote(Quote(symbol="F250706P00010000", bid=0.79, ask=0.81))

    proposal = await propose_close_for_position(
        broker, db_repos, position,
        today=date(2025, 6, 1),
        strategy=_strategy(csp_stop_loss_mult=2.0),
    )
    assert proposal is None


@pytest.mark.asyncio
async def test_cc_does_not_have_stop_loss(db_repos):
    """CC positions are intentionally excluded from stop-loss — called-away
    is the wheel's profitable outcome, not a loss. Even at 5× original
    premium, the close orchestrator should not propose a stop-loss close."""
    position = await _seed_open_cc(db_repos, fill_price=0.80)
    broker = PaperBroker()
    # Mid 5.00 — way over a 2× stop, but not in profit and DTE > 21 so no
    # other trigger fires. Should return None.
    broker.seed_quote(Quote(symbol="F250706C00012000", bid=4.95, ask=5.05))

    proposal = await propose_close_for_position(
        broker, db_repos, position,
        today=date(2025, 6, 1),
        strategy=_strategy(csp_stop_loss_mult=2.0),
    )
    # CC is being called-away territory — no stop close.
    assert proposal is None


@pytest.mark.asyncio
async def test_csp_stop_loss_disabled_when_mult_zero(db_repos):
    """csp_stop_loss_mult=0 disables the feature for this strategy."""
    position, _ = await _seed_open_csp(db_repos, fill_price=0.46)
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F250706P00010000", bid=2.00, ask=2.02))  # 4× original

    proposal = await propose_close_for_position(
        broker, db_repos, position,
        today=date(2025, 6, 1),
        strategy=_strategy(csp_stop_loss_mult=0),
    )
    # No stop trigger; not in profit; DTE 35 so no time trigger. Returns None.
    assert proposal is None


@pytest.mark.asyncio
async def test_propose_all_closes_skips_non_open_states(db_repos):
    """SPREAD_OPEN, MANUAL_INTERVENTION, IDLE etc. must not be considered."""
    position, _ = await _seed_open_csp(db_repos, fill_price=1.00)
    # Move it out of CSP_OPEN.
    conn = await db_repos.db.connect()
    await conn.execute(
        "UPDATE positions SET state = 'MANUAL_INTERVENTION' WHERE id = ?",
        (position.id,),
    )
    await conn.commit()

    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F250706P00010000", bid=0.10, ask=0.12))  # deep profit

    proposals = await propose_all_closes(
        broker, db_repos, {"account": {"id": "test"}}, strategy=_strategy(),
    )
    assert proposals == []
