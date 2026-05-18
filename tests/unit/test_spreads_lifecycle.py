"""Multi-leg spread lifecycle: reconciler fills + expiration + closes + P&L."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from core.models import (
    OptionContract,
    OptionType,
    Order,
    OrderLeg,
    OrderStatus,
    OrderType,
    Position,
    PositionState,
    Quote,
    UniverseEntry,
    WheelCycle,
)
from core.strategies import StrategyDefinition
from execution.reconciler import Reconciler
from execution.router import OrderRouter
from platforms.paper_broker import PaperBroker
from strategies.spreads import (
    MultiLegProposal,
    propose_close_for_symbol,
)


# -- helpers ----------------------------------------------------------------


def _config() -> dict:
    return {
        "account": {"id": "test", "broker": "paper"},
        "wheel": {
            "buying_power_floor_pct": 5,
            "max_position_pct_of_account": 50,
            "max_concurrent_positions": 4,
            "open_interest_min": 0,
            "volume_min": 0,
            "bid_ask_spread_max_pct": 100.0,
        },
        "regime": {"enabled": False},
        "execution": {
            "dry_run": False,
            "retry_max_attempts": 3,
            "retry_initial_backoff_seconds": 0,
            "retry_max_backoff_seconds": 0,
        },
    }


def _universe() -> dict:
    return {
        "tickers": [UniverseEntry(symbol="F", name="Ford", tier=1, overrides={})],
        "banned": [],
        "banned_rules": [],
    }


def _strategy(**overrides: Any) -> StrategyDefinition:
    params: dict[str, Any] = {
        "dte_min": 30,
        "dte_max": 45,
        "short_delta_min": 0.20,
        "short_delta_max": 0.30,
        "spread_width_dollars": 1.0,
        "min_credit_pct_of_width": 25.0,
        "profit_close_pct": 50,
        "time_close_dte": 7,
    }
    params.update(overrides)
    return StrategyDefinition(
        id="put_spread",
        display_name="Bull Put Spread",
        type="vertical_spread",
        enabled=True,
        max_concurrent=4,
        params=params,
    )


def _spread_legs() -> list[OrderLeg]:
    today = date(2025, 6, 1)
    return [
        OrderLeg(
            contract_symbol="F250706P00010000",
            underlying="F",
            option_type=OptionType.PUT,
            strike=10.0,
            expiration=today + timedelta(days=35),
            action=OrderType.SELL_TO_OPEN,
        ),
        OrderLeg(
            contract_symbol="F250706P00009000",
            underlying="F",
            option_type=OptionType.PUT,
            strike=9.0,
            expiration=today + timedelta(days=35),
            action=OrderType.BUY_TO_OPEN,
        ),
    ]


def _spread_proposal(qty: int = 2) -> MultiLegProposal:
    return MultiLegProposal(
        symbol="F",
        legs=_spread_legs(),
        net_credit_per_spread=0.30,
        max_loss_per_spread=70.0,
        width_dollars=1.0,
        quantity=qty,
        rationale="put_spread test",
        strategy_id="put_spread",
    )


async def _noop_sleep(_: float) -> None:
    return None


@pytest.fixture(autouse=True)
def _stub_earnings(monkeypatch):
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)


# -- reconciler: multi-leg open fill --------------------------------------


@pytest.mark.asyncio
async def test_reconciler_multi_leg_open_fill_transitions_to_spread_open(db_repos):
    broker = PaperBroker(cash=20_000)
    router = OrderRouter(broker, db_repos, _config(), _universe())
    result = await router.place_multi_leg(
        _spread_proposal(), sleep=_noop_sleep, today=date(2025, 6, 1),
    )
    assert result.placed is not None
    # Fill the package at the limit price.
    await broker.fill_multi_leg(result.placed.broker_order_id, fill_price=0.30)

    reconciler = Reconciler(broker, db_repos, _config())
    summary = await reconciler.reconcile_once()
    assert summary.fills_processed == 1
    assert summary.cycles_opened == 1

    pos = await db_repos.positions.get_by_symbol(
        "test", "F", strategy_id="put_spread"
    )
    assert pos is not None
    assert pos.state == PositionState.SPREAD_OPEN
    assert pos.current_cycle_id is not None

    cycle = await db_repos.cycles.get(pos.current_cycle_id)
    assert cycle is not None
    assert cycle.strategy_id == "put_spread"
    # initial_csp_premium = 0.30 * 100 * 2 = 60.0
    assert cycle.initial_csp_premium == pytest.approx(60.0)
    # capital_at_risk = (1.0 - 0.30) * 100 * 2 = 140.0
    assert cycle.initial_capital_at_risk == pytest.approx(140.0)


# -- reconciler: multi-leg close fill -------------------------------------


@pytest.mark.asyncio
async def test_reconciler_multi_leg_close_fill_closes_cycle(db_repos):
    broker = PaperBroker(cash=20_000)
    router = OrderRouter(broker, db_repos, _config(), _universe())

    open_result = await router.place_multi_leg(
        _spread_proposal(qty=1), sleep=_noop_sleep, today=date(2025, 6, 1),
    )
    await broker.fill_multi_leg(open_result.placed.broker_order_id, fill_price=0.30)
    reconciler = Reconciler(broker, db_repos, _config())
    await reconciler.reconcile_once()

    # Now place + fill a close at $0.10 debit.
    close_legs = [
        OrderLeg(
            contract_symbol=leg.contract_symbol,
            underlying=leg.underlying,
            option_type=leg.option_type,
            strike=leg.strike,
            expiration=leg.expiration,
            action=(
                OrderType.BUY_TO_CLOSE
                if leg.action == OrderType.SELL_TO_OPEN
                else OrderType.SELL_TO_CLOSE
            ),
        )
        for leg in _spread_legs()
    ]
    close_proposal = MultiLegProposal(
        symbol="F",
        legs=close_legs,
        net_credit_per_spread=-0.10,  # signed: paid 0.10 to close
        max_loss_per_spread=0.0,
        width_dollars=0.0,
        quantity=1,
        rationale="close test",
        strategy_id="put_spread",
        order_type=OrderType.MULTI_LEG_CLOSE,
    )
    close_result = await router.place_multi_leg(
        close_proposal, sleep=_noop_sleep, today=date(2025, 6, 5),
    )
    await broker.fill_multi_leg(close_result.placed.broker_order_id, fill_price=-0.10)

    summary = await reconciler.reconcile_once()
    assert summary.fills_processed == 1
    assert summary.cycles_closed == 1

    pos = await db_repos.positions.get_by_symbol(
        "test", "F", strategy_id="put_spread"
    )
    assert pos is not None
    assert pos.state == PositionState.SPREAD_CLOSED

    cycle = await db_repos.cycles.get(pos.current_cycle_id)
    assert cycle is not None
    assert cycle.ended_at is not None
    assert cycle.cycle_outcome == "SPREAD_CLOSED_PROFIT"
    # P&L: open credit +30 - close debit 10 = +20
    assert cycle.final_pnl == pytest.approx(20.0)


# -- reconciler: order cancellation restores position state ----------------


@pytest.mark.asyncio
async def test_reconciler_cancelled_open_drops_position_back_to_idle(db_repos):
    """MULTI_LEG_OPEN cancelled at broker → SPREAD_PENDING must return to IDLE.

    Regression for the live-bot GOOGL situation: open attempt cancelled,
    position was stuck SPREAD_PENDING indefinitely, blocking new entries.
    """
    broker = PaperBroker(cash=20_000)
    router = OrderRouter(broker, db_repos, _config(), _universe())
    result = await router.place_multi_leg(
        _spread_proposal(qty=1), sleep=_noop_sleep, today=date(2025, 6, 1),
    )
    assert result.placed is not None
    # Cancel at the broker without filling.
    await broker.cancel_order(result.placed.broker_order_id)

    reconciler = Reconciler(broker, db_repos, _config())
    summary = await reconciler.reconcile_once()
    assert summary.cancellations_processed == 1
    assert summary.fills_processed == 0

    pos = await db_repos.positions.get_by_symbol(
        "test", "F", strategy_id="put_spread"
    )
    assert pos is not None
    assert pos.state == PositionState.IDLE
    assert pos.current_cycle_id is None


@pytest.mark.asyncio
async def test_reconciler_cancelled_close_restores_spread_open(db_repos):
    """MULTI_LEG_CLOSE cancelled at broker → return to SPREAD_OPEN (still hold legs).

    Regression for the live-bot NVDA situation: close attempt cancelled,
    position stuck SPREAD_PENDING blocked the close orchestrator from
    retrying — bot couldn't manage the spread for 4 days.
    """
    broker = PaperBroker(cash=20_000)
    router = OrderRouter(broker, db_repos, _config(), _universe())

    # 1. Open + fill.
    open_result = await router.place_multi_leg(
        _spread_proposal(qty=1), sleep=_noop_sleep, today=date(2025, 6, 1),
    )
    await broker.fill_multi_leg(open_result.placed.broker_order_id, fill_price=0.30)
    reconciler = Reconciler(broker, db_repos, _config())
    await reconciler.reconcile_once()

    pos_before = await db_repos.positions.get_by_symbol(
        "test", "F", strategy_id="put_spread"
    )
    assert pos_before.state == PositionState.SPREAD_OPEN
    cycle_id_before = pos_before.current_cycle_id
    assert cycle_id_before is not None

    # 2. Place a close that won't fill, then cancel it.
    close_legs = [
        OrderLeg(
            contract_symbol=leg.contract_symbol,
            underlying=leg.underlying,
            option_type=leg.option_type,
            strike=leg.strike,
            expiration=leg.expiration,
            action=(
                OrderType.BUY_TO_CLOSE
                if leg.action == OrderType.SELL_TO_OPEN
                else OrderType.SELL_TO_CLOSE
            ),
        )
        for leg in _spread_legs()
    ]
    close_proposal = MultiLegProposal(
        symbol="F",
        legs=close_legs,
        net_credit_per_spread=-0.10,
        max_loss_per_spread=0.0,
        width_dollars=0.0,
        quantity=1,
        rationale="close test",
        strategy_id="put_spread",
        order_type=OrderType.MULTI_LEG_CLOSE,
    )
    close_result = await router.place_multi_leg(
        close_proposal, sleep=_noop_sleep, today=date(2025, 6, 5),
    )
    assert close_result.placed is not None
    await broker.cancel_order(close_result.placed.broker_order_id)

    # 3. Reconcile → SPREAD_PENDING should restore to SPREAD_OPEN, cycle intact.
    summary = await reconciler.reconcile_once()
    assert summary.cancellations_processed == 1
    pos_after = await db_repos.positions.get_by_symbol(
        "test", "F", strategy_id="put_spread"
    )
    assert pos_after.state == PositionState.SPREAD_OPEN
    assert pos_after.current_cycle_id == cycle_id_before  # cycle preserved


@pytest.mark.asyncio
async def test_reconciler_cancel_skips_when_position_not_in_spread_pending(db_repos):
    """If the position has been moved to MANUAL_INTERVENTION or any other
    state by another rule, a stale cancellation must not clobber it."""
    broker = PaperBroker(cash=20_000)
    router = OrderRouter(broker, db_repos, _config(), _universe())
    result = await router.place_multi_leg(
        _spread_proposal(qty=1), sleep=_noop_sleep, today=date(2025, 6, 1),
    )
    # Manually move the position out of SPREAD_PENDING before reconcile.
    conn = await db_repos.db.connect()
    await conn.execute(
        "UPDATE positions SET state = 'MANUAL_INTERVENTION' WHERE symbol = 'F'"
    )
    await conn.commit()

    await broker.cancel_order(result.placed.broker_order_id)
    reconciler = Reconciler(broker, db_repos, _config())
    summary = await reconciler.reconcile_once()
    assert summary.cancellations_processed == 0  # guard fired

    pos = await db_repos.positions.get_by_symbol(
        "test", "F", strategy_id="put_spread"
    )
    assert pos.state == PositionState.MANUAL_INTERVENTION  # untouched


# -- reconciler: spread expiration (broker shows nothing) ------------------


@pytest.mark.asyncio
async def test_reconciler_spread_expiration_max_profit(db_repos):
    broker = PaperBroker(cash=20_000)
    router = OrderRouter(broker, db_repos, _config(), _universe())
    open_result = await router.place_multi_leg(
        _spread_proposal(qty=1), sleep=_noop_sleep, today=date(2025, 6, 1),
    )
    await broker.fill_multi_leg(open_result.placed.broker_order_id, fill_price=0.30)
    reconciler = Reconciler(broker, db_repos, _config())
    await reconciler.reconcile_once()

    # Simulate expiration: broker drops both legs from open_options.
    broker._open_options.pop("F250706P00010000", None)
    broker._open_options.pop("F250706P00009000", None)

    summary = await reconciler.reconcile_once()
    assert summary.expirations_processed == 1
    assert summary.cycles_closed == 1

    pos = await db_repos.positions.get_by_symbol(
        "test", "F", strategy_id="put_spread"
    )
    assert pos is not None
    assert pos.state == PositionState.IDLE
    assert pos.current_cycle_id is None


# -- close orchestrator ----------------------------------------------------


@pytest.mark.asyncio
async def test_close_orchestrator_triggers_on_profit_target(db_repos):
    broker = PaperBroker(cash=20_000)
    router = OrderRouter(broker, db_repos, _config(), _universe())
    open_result = await router.place_multi_leg(
        _spread_proposal(qty=1), sleep=_noop_sleep, today=date(2025, 6, 1),
    )
    await broker.fill_multi_leg(open_result.placed.broker_order_id, fill_price=0.30)
    reconciler = Reconciler(broker, db_repos, _config())
    await reconciler.reconcile_once()

    # Quote each leg cheap → debit-to-close ≤ 0.15 (50% of 0.30 credit).
    broker.seed_quote(Quote(symbol="F250706P00010000", bid=0.12, ask=0.14))
    broker.seed_quote(Quote(symbol="F250706P00009000", bid=0.02, ask=0.04))

    proposal = await propose_close_for_symbol(
        broker, db_repos, "F", _config(),
        today=date(2025, 6, 10), strategy=_strategy(),
    )
    assert proposal is not None
    assert proposal.order_type == OrderType.MULTI_LEG_CLOSE
    # Buy short at 0.13 mid, sell long at 0.03 mid → debit 0.10 → net credit -0.10
    assert proposal.net_credit_per_spread == pytest.approx(-0.10)
    actions = sorted(str(leg.action) for leg in proposal.legs)
    assert actions == sorted([OrderType.BUY_TO_CLOSE.value, OrderType.SELL_TO_CLOSE.value])


@pytest.mark.asyncio
async def test_close_orchestrator_triggers_on_dte(db_repos):
    broker = PaperBroker(cash=20_000)
    router = OrderRouter(broker, db_repos, _config(), _universe())
    open_result = await router.place_multi_leg(
        _spread_proposal(qty=1), sleep=_noop_sleep, today=date(2025, 6, 1),
    )
    await broker.fill_multi_leg(open_result.placed.broker_order_id, fill_price=0.30)
    reconciler = Reconciler(broker, db_repos, _config())
    await reconciler.reconcile_once()

    # Quotes that DON'T trigger profit-close (debit > target), but the date
    # we ask for is inside the time_close_dte window.
    broker.seed_quote(Quote(symbol="F250706P00010000", bid=0.40, ask=0.42))
    broker.seed_quote(Quote(symbol="F250706P00009000", bid=0.10, ask=0.12))
    expiration = date(2025, 6, 1) + timedelta(days=35)  # 2025-07-06
    today = expiration - timedelta(days=5)

    proposal = await propose_close_for_symbol(
        broker, db_repos, "F", _config(), today=today, strategy=_strategy(),
    )
    assert proposal is not None
    assert "time_close" in proposal.rationale


@pytest.mark.asyncio
async def test_close_orchestrator_skips_when_no_trigger(db_repos):
    broker = PaperBroker(cash=20_000)
    router = OrderRouter(broker, db_repos, _config(), _universe())
    open_result = await router.place_multi_leg(
        _spread_proposal(qty=1), sleep=_noop_sleep, today=date(2025, 6, 1),
    )
    await broker.fill_multi_leg(open_result.placed.broker_order_id, fill_price=0.30)
    reconciler = Reconciler(broker, db_repos, _config())
    await reconciler.reconcile_once()

    # Quotes that miss profit target.
    broker.seed_quote(Quote(symbol="F250706P00010000", bid=0.40, ask=0.42))
    broker.seed_quote(Quote(symbol="F250706P00009000", bid=0.10, ask=0.12))

    # Date well before time-close window.
    proposal = await propose_close_for_symbol(
        broker, db_repos, "F", _config(),
        today=date(2025, 6, 5), strategy=_strategy(),
    )
    assert proposal is None


# -- PnL math --------------------------------------------------------------


@pytest.mark.asyncio
async def test_compute_cycle_pnl_handles_multi_leg_credit_and_debit(db_repos):
    """Open at +0.40 credit, close at -0.15 debit, qty 3 → P&L = (0.40 - 0.15) * 100 * 3 = 75."""
    cycle_id = await db_repos.cycles.insert(
        WheelCycle(
            account_id="test",
            symbol="F",
            strategy_id="put_spread",
            started_at=datetime.now(UTC).replace(tzinfo=None),
            initial_capital_at_risk=180.0,
        )
    )
    placed_at = datetime.now(UTC).replace(tzinfo=None)
    await db_repos.orders.insert(
        Order(
            account_id="test",
            symbol="F",
            strategy_id="put_spread",
            cycle_id=cycle_id,
            order_type=OrderType.MULTI_LEG_OPEN,
            quantity=3,
            limit_price=0.40,
            fill_price=0.40,
            status=OrderStatus.FILLED,
            placed_at=placed_at,
            filled_at=placed_at,
        )
    )
    await db_repos.orders.insert(
        Order(
            account_id="test",
            symbol="F",
            strategy_id="put_spread",
            cycle_id=cycle_id,
            order_type=OrderType.MULTI_LEG_CLOSE,
            quantity=3,
            limit_price=-0.15,
            fill_price=-0.15,
            status=OrderStatus.FILLED,
            placed_at=placed_at,
            filled_at=placed_at,
        )
    )
    reconciler = Reconciler(PaperBroker(), db_repos, _config())
    pnl = await reconciler._compute_cycle_pnl(cycle_id)
    assert pnl == pytest.approx(75.0)
