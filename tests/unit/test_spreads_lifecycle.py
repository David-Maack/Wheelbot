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


@pytest.mark.asyncio
async def test_reconciler_selfheals_stuck_spread_pending_to_open(db_repos):
    """TICKET-029: a spread whose PENDING→OPEN transition was missed (reconcile
    cursor reset mid-rebuild, or a missed close-cancel) is stranded in
    SPREAD_PENDING. The close orchestrator only walks SPREAD_OPEN, so it would
    never be managed and never stopped out (META/GOOGL did exactly this). With
    no order in flight and the legs still at the broker, the reconciler
    self-heals it back to SPREAD_OPEN so the stop-loss can run again."""
    broker = PaperBroker(cash=20_000)
    router = OrderRouter(broker, db_repos, _config(), _universe())
    open_result = await router.place_multi_leg(
        _spread_proposal(qty=1), sleep=_noop_sleep, today=date(2025, 6, 1),
    )
    await broker.fill_multi_leg(open_result.placed.broker_order_id, fill_price=0.30)
    reconciler = Reconciler(broker, db_repos, _config())
    await reconciler.reconcile_once()  # normal path → SPREAD_OPEN

    pos = await db_repos.positions.get_by_symbol("test", "F", strategy_id="put_spread")
    assert pos.state == PositionState.SPREAD_OPEN
    cycle_id = pos.current_cycle_id

    # Strand it: force SPREAD_PENDING while the legs are still open at the broker
    # and no order is in flight (the open FILLED earlier) — the exact stuck shape.
    await db_repos.positions.update(pos.id, state=PositionState.SPREAD_PENDING.value)

    await reconciler.reconcile_once()
    healed = await db_repos.positions.get_by_symbol("test", "F", strategy_id="put_spread")
    assert healed.state == PositionState.SPREAD_OPEN
    assert healed.current_cycle_id == cycle_id


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

    # Capture the cycle id while the spread is open — after close it's cleared
    # from the position (cycle pointer reset so the dashboard doesn't read a
    # dead cycle during the next pending window).
    pos_open = await db_repos.positions.get_by_symbol("test", "F", strategy_id="put_spread")
    opened_cycle_id = pos_open.current_cycle_id
    assert opened_cycle_id is not None

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
    # Cycle pointer cleared on close (Bug B fix).
    assert pos.current_cycle_id is None

    cycle = await db_repos.cycles.get(opened_cycle_id)
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
    """MULTI_LEG_CLOSE lifecycle keeps the position in SPREAD_OPEN throughout.

    2026-07-23 review fix strengthened the original NVDA regression: a close
    placement no longer moves the position to SPREAD_PENDING at all (the
    close orchestrator only walks SPREAD_OPEN, so the PENDING hop dead-ended
    a missed stop-out until the DAY order expired). The position stays
    SPREAD_OPEN while the close works, so the orchestrator keeps
    re-proposing and the pre-submission stale-replace re-prices; a cancelled
    close therefore needs no restore.
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
    # The close placement itself must NOT pend the position (the fix).
    pos_mid = await db_repos.positions.get_by_symbol(
        "test", "F", strategy_id="put_spread"
    )
    assert pos_mid.state == PositionState.SPREAD_OPEN
    await broker.cancel_order(close_result.placed.broker_order_id)

    # 3. Reconcile → still SPREAD_OPEN, cycle intact; no restore was needed
    # so the cancellation is a defensive no-op (counter stays 0).
    summary = await reconciler.reconcile_once()
    assert summary.cancellations_processed == 0
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
    # Order price is MARKETABLE: buy short at ask 0.14, sell long at bid 0.02 →
    # debit 0.12 → net credit -0.12 (the 0.10 mid still drives the trigger).
    assert proposal.net_credit_per_spread == pytest.approx(-0.12)
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
async def test_close_orchestrator_stop_loss_triggers_at_2x_credit_bull_put(db_repos):
    """Bull put stop-loss: close when debit-to-close ≥ 2× original credit.

    Sprint 12 sub-sprint 2 — the change that would have capped TSLA's
    drawdown at ~2× credit instead of letting it ride toward max loss.
    """
    broker = PaperBroker(cash=20_000)
    router = OrderRouter(broker, db_repos, _config(), _universe())
    # Open at 0.30 net credit per spread.
    open_result = await router.place_multi_leg(
        _spread_proposal(qty=1), sleep=_noop_sleep, today=date(2025, 6, 1),
    )
    await broker.fill_multi_leg(open_result.placed.broker_order_id, fill_price=0.30)
    reconciler = Reconciler(broker, db_repos, _config())
    await reconciler.reconcile_once()

    # Quote each leg such that net debit-to-close = 0.62 (>= 2× 0.30 = 0.60).
    # Profit close target is 0.15 (we'd need debit ≤ 0.15) — not triggered.
    # Time close at DTE 7 — not triggered (we're far from expiration).
    # Stop-loss should fire.
    broker.seed_quote(Quote(symbol="F250706P00010000", bid=0.71, ask=0.73))
    broker.seed_quote(Quote(symbol="F250706P00009000", bid=0.10, ask=0.12))

    proposal = await propose_close_for_symbol(
        broker, db_repos, "F", _config(),
        today=date(2025, 6, 10),
        strategy=_strategy(stop_loss_mult=2.0),
    )
    assert proposal is not None
    assert proposal.order_type == OrderType.MULTI_LEG_CLOSE
    assert "stop_loss" in proposal.rationale
    # Trigger uses the 0.61 mid debit (≥ 0.60 threshold). Order price is
    # MARKETABLE: buy short at ask 0.73, sell long at bid 0.10 → debit 0.63 →
    # net credit -0.63.
    assert proposal.net_credit_per_spread == pytest.approx(-0.63)


@pytest.mark.asyncio
async def test_close_orchestrator_stop_loss_triggers_at_1_5x_credit_bear_call(db_repos):
    """Bear call uses a tighter 1.5× multiplier than bull puts (asymmetric melt-up risk).

    Same lifecycle path as put_spread — direction-agnostic close machinery.
    """
    # Build call legs for a bear_call_spread.
    from datetime import timedelta as td
    today = date(2025, 6, 1)
    bear_call_legs = [
        OrderLeg(
            contract_symbol="F250706C00010000",
            underlying="F",
            option_type=OptionType.CALL,
            strike=10.0,
            expiration=today + td(days=35),
            action=OrderType.SELL_TO_OPEN,
        ),
        OrderLeg(
            contract_symbol="F250706C00011000",
            underlying="F",
            option_type=OptionType.CALL,
            strike=11.0,
            expiration=today + td(days=35),
            action=OrderType.BUY_TO_OPEN,
        ),
    ]
    broker = PaperBroker(cash=20_000)
    router = OrderRouter(broker, db_repos, _config(), _universe())
    open_proposal = MultiLegProposal(
        symbol="F",
        legs=bear_call_legs,
        net_credit_per_spread=0.40,
        max_loss_per_spread=60.0,
        width_dollars=1.0,
        quantity=1,
        rationale="bear_call test",
        strategy_id="bear_call_spread",
        direction="bear_call",
    )
    open_result = await router.place_multi_leg(open_proposal, sleep=_noop_sleep, today=today)
    await broker.fill_multi_leg(open_result.placed.broker_order_id, fill_price=0.40)
    reconciler = Reconciler(broker, db_repos, _config())
    await reconciler.reconcile_once()

    # Net debit-to-close = 0.62 ≥ 1.5 × 0.40 = 0.60 → stop fires.
    # At 2.0× threshold (0.80) it would NOT fire, confirming the asymmetric multiplier.
    broker.seed_quote(Quote(symbol="F250706C00010000", bid=0.71, ask=0.73))
    broker.seed_quote(Quote(symbol="F250706C00011000", bid=0.10, ask=0.12))

    bear_call_strategy = StrategyDefinition(
        id="bear_call_spread",
        display_name="Bear Call Spread",
        type="vertical_spread",
        enabled=True,
        max_concurrent=4,
        params={
            "direction": "bear_call",
            "dte_min": 30, "dte_max": 45,
            "short_delta_min": 0.20, "short_delta_max": 0.30,
            "spread_width_dollars": 1.0,
            "profit_close_pct": 35,
            "time_close_dte": 21,
            "stop_loss_mult": 1.5,
        },
    )
    proposal = await propose_close_for_symbol(
        broker, db_repos, "F", _config(),
        today=date(2025, 6, 10),
        strategy=bear_call_strategy,
    )
    assert proposal is not None
    assert proposal.order_type == OrderType.MULTI_LEG_CLOSE
    assert proposal.direction == "bear_call"
    assert "stop_loss" in proposal.rationale


@pytest.mark.asyncio
async def test_close_orchestrator_stop_loss_does_not_trigger_below_threshold(db_repos):
    """If the debit-to-close is below stop_loss_mult × credit, no stop trigger."""
    broker = PaperBroker(cash=20_000)
    router = OrderRouter(broker, db_repos, _config(), _universe())
    open_result = await router.place_multi_leg(
        _spread_proposal(qty=1), sleep=_noop_sleep, today=date(2025, 6, 1),
    )
    await broker.fill_multi_leg(open_result.placed.broker_order_id, fill_price=0.30)
    reconciler = Reconciler(broker, db_repos, _config())
    await reconciler.reconcile_once()

    # Net debit-to-close = 0.50. At 2× credit threshold (0.60), this is below
    # the stop. Above the profit target of 0.15 too. Outside time window.
    # → no trigger at all.
    broker.seed_quote(Quote(symbol="F250706P00010000", bid=0.59, ask=0.61))
    broker.seed_quote(Quote(symbol="F250706P00009000", bid=0.09, ask=0.11))

    proposal = await propose_close_for_symbol(
        broker, db_repos, "F", _config(),
        today=date(2025, 6, 10),
        strategy=_strategy(stop_loss_mult=2.0),
    )
    assert proposal is None


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


# -- TICKET-014 precursor fixes -------------------------------------------


def _condor_legs() -> list[OrderLeg]:
    """Symmetric iron condor on F with $5 wings: 90/95P + 105/110C @ 35 DTE."""
    today = date(2025, 6, 1)
    exp = today + timedelta(days=35)
    return [
        OrderLeg(
            contract_symbol="F250706P00090000",
            underlying="F", option_type=OptionType.PUT, strike=90.0,
            expiration=exp, action=OrderType.BUY_TO_OPEN,    # long put
        ),
        OrderLeg(
            contract_symbol="F250706P00095000",
            underlying="F", option_type=OptionType.PUT, strike=95.0,
            expiration=exp, action=OrderType.SELL_TO_OPEN,   # short put
        ),
        OrderLeg(
            contract_symbol="F250706C00105000",
            underlying="F", option_type=OptionType.CALL, strike=105.0,
            expiration=exp, action=OrderType.SELL_TO_OPEN,   # short call
        ),
        OrderLeg(
            contract_symbol="F250706C00110000",
            underlying="F", option_type=OptionType.CALL, strike=110.0,
            expiration=exp, action=OrderType.BUY_TO_OPEN,    # long call
        ),
    ]


@pytest.mark.asyncio
async def test_open_cycle_for_spread_uses_explicit_width_dollars(db_repos):
    """Precursor #1: when raw_request['width_dollars'] is present, reconciler
    uses it verbatim instead of computing max(strikes)-min(strikes).

    For a 4-leg iron condor 90/95P + 105/110C the legs-derived width is
    20 (outer span), but the wing width is 5. capital_at_risk should be
    (5 - 0.30) * 100 * 1 = 470.0, not (20 - 0.30) * 100 * 1 = 1970.0."""
    legs = _condor_legs()
    raw_request = {
        "underlying": "F",
        "legs": [leg.model_dump(mode="json") for leg in legs],
        "quantity": 1,
        "limit_price": 0.30,
        "width_dollars": 5.0,    # the precursor-fix field
    }
    placed_at = datetime.now(UTC).replace(tzinfo=None)
    order = await db_repos.orders.insert(
        Order(
            account_id="test",
            symbol="F",
            strategy_id="iron_condor",
            order_type=OrderType.MULTI_LEG_OPEN,
            contract_symbol=legs[1].contract_symbol,  # short put as canonical
            strike=95.0,
            expiration=legs[0].expiration,
            option_type=OptionType.PUT,
            quantity=1,
            limit_price=0.30,
            fill_price=-0.30,
            status=OrderStatus.FILLED,
            placed_at=placed_at,
            filled_at=placed_at,
            raw_request=raw_request,
        )
    )
    reconciler = Reconciler(PaperBroker(), db_repos, _config())
    full_order = await db_repos.orders.get(order)
    cycle_id = await reconciler._open_cycle_for_spread(
        full_order, fill_price=0.30, summary=type("S", (), {"cycles_opened": 0})(),
    )
    cycle = await db_repos.cycles.get(cycle_id)
    # Explicit width 5: (5.0 - 0.30) * 100 * 1 = 470.0
    assert cycle.initial_capital_at_risk == pytest.approx(470.0)


@pytest.mark.asyncio
async def test_open_cycle_for_spread_falls_back_to_legs_when_no_width(db_repos):
    """Backwards-compat: orders placed before the router started stamping
    width_dollars still produce a cycle (using the legs-derived width).
    This locks the fallback path so old orders don't crash."""
    legs = _spread_legs()
    raw_request = {
        "underlying": "F",
        "legs": [leg.model_dump(mode="json") for leg in legs],
        "quantity": 1,
        "limit_price": 0.30,
        # NO width_dollars key.
    }
    placed_at = datetime.now(UTC).replace(tzinfo=None)
    order_id = await db_repos.orders.insert(
        Order(
            account_id="test",
            symbol="F",
            strategy_id="put_spread",
            order_type=OrderType.MULTI_LEG_OPEN,
            contract_symbol=legs[0].contract_symbol,
            strike=10.0,
            expiration=legs[0].expiration,
            option_type=OptionType.PUT,
            quantity=1,
            limit_price=0.30,
            fill_price=-0.30,
            status=OrderStatus.FILLED,
            placed_at=placed_at,
            filled_at=placed_at,
            raw_request=raw_request,
        )
    )
    reconciler = Reconciler(PaperBroker(), db_repos, _config())
    full_order = await db_repos.orders.get(order_id)
    cycle_id = await reconciler._open_cycle_for_spread(
        full_order, fill_price=0.30, summary=type("S", (), {"cycles_opened": 0})(),
    )
    cycle = await db_repos.cycles.get(cycle_id)
    # Legs-derived width: max(10, 9) - min(10, 9) = 1; (1 - 0.30) * 100 * 1 = 70
    assert cycle.initial_capital_at_risk == pytest.approx(70.0)


@pytest.mark.asyncio
async def test_close_direction_set_by_strategy_id_for_iron_condor(db_repos):
    """Precursor #3: propose_close_for_symbol should set direction=iron_condor
    when strategy.id == 'iron_condor', NOT bear_call (which is what the old
    leg-type inference produced as soon as any call leg was seen).

    Closes bypass the regime gate so this is a label bug, not a fill bug,
    but the rationale string and downstream display would show the wrong
    direction. Locks the fix."""
    from strategies.spreads import DIRECTION_IRON_CONDOR

    broker = PaperBroker(cash=20_000)
    router = OrderRouter(broker, db_repos, _config(), _universe())
    condor_proposal = MultiLegProposal(
        symbol="F",
        legs=_condor_legs(),
        net_credit_per_spread=0.50,
        max_loss_per_spread=450.0,
        width_dollars=5.0,
        quantity=1,
        rationale="iron condor test",
        strategy_id="iron_condor",
        direction=DIRECTION_IRON_CONDOR,
    )
    open_result = await router.place_multi_leg(
        condor_proposal, sleep=_noop_sleep, today=date(2025, 6, 1),
    )
    await broker.fill_multi_leg(open_result.placed.broker_order_id, fill_price=0.50)
    reconciler = Reconciler(broker, db_repos, _config())
    await reconciler.reconcile_once()

    # Seed mids that trigger profit close (debit-to-close < 75% of 0.50).
    for occ, bid, ask in [
        ("F250706P00090000", 0.04, 0.06),    # long put cheap
        ("F250706P00095000", 0.08, 0.10),    # short put cheap
        ("F250706C00105000", 0.08, 0.10),    # short call cheap
        ("F250706C00110000", 0.04, 0.06),    # long call cheap
    ]:
        broker.seed_quote(Quote(symbol=occ, bid=bid, ask=ask))

    condor_strategy = StrategyDefinition(
        id="iron_condor",
        display_name="Iron Condor",
        type="iron_condor",
        enabled=True,
        max_concurrent=2,
        params={
            "profit_close_pct": 25,
            "time_close_dte": 21,
            "spread_width_dollars": 5.0,
            "min_credit_pct_of_width": 30.0,
            "dte_min": 30, "dte_max": 45,
            "short_delta_min": 0.10, "short_delta_max": 0.20,
        },
    )
    close_proposal = await propose_close_for_symbol(
        broker, db_repos, "F", _config(),
        today=date(2025, 6, 10), strategy=condor_strategy,
    )
    assert close_proposal is not None
    # The fix: direction comes from strategy.id, not leg-type inference.
    assert close_proposal.direction == DIRECTION_IRON_CONDOR


@pytest.mark.asyncio
async def test_router_stamps_width_dollars_onto_raw_request(db_repos):
    """Precursor #1 end-to-end: routing a MultiLegProposal results in an
    Order whose raw_request carries the proposal's width_dollars."""
    broker = PaperBroker(cash=20_000)
    router = OrderRouter(broker, db_repos, _config(), _universe())
    proposal = MultiLegProposal(
        symbol="F",
        legs=_spread_legs(),
        net_credit_per_spread=0.30,
        max_loss_per_spread=70.0,
        width_dollars=1.0,
        quantity=1,
        rationale="test",
        strategy_id="put_spread",
    )
    result = await router.place_multi_leg(
        proposal, sleep=_noop_sleep, today=date(2025, 6, 1),
    )
    assert result.placed is not None
    persisted = await db_repos.orders.get_by_client_id(result.placed.client_order_id)
    assert persisted is not None
    assert persisted.raw_request is not None
    assert persisted.raw_request.get("width_dollars") == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_close_debit_clamped_at_wing_width(db_repos):
    """2026-07-23 review fix: a marketable close on blown-out quotes must
    never pay more than the wing width — TSLA 360/365 closed at a $6.09
    debit on a $5 structure (-$470 realized vs -$365 theoretical max)."""
    broker = PaperBroker(cash=20_000)
    router = OrderRouter(broker, db_repos, _config(), _universe())
    open_result = await router.place_multi_leg(
        _spread_proposal(qty=1), sleep=_noop_sleep, today=date(2025, 6, 1),
    )
    await broker.fill_multi_leg(open_result.placed.broker_order_id, fill_price=0.30)
    reconciler = Reconciler(broker, db_repos, _config())
    await reconciler.reconcile_once()

    # $1-wide 10/9 put spread. Quotes blown out: marketable debit would be
    # ask(short) - bid(long) = 1.40 - 0.10 = 1.30 > width 1.00.
    broker.seed_quote(Quote(symbol="F250706P00010000", bid=1.30, ask=1.40))
    broker.seed_quote(Quote(symbol="F250706P00009000", bid=0.10, ask=0.12))

    proposal = await propose_close_for_symbol(
        broker, db_repos, "F", _config(),
        today=date(2025, 6, 10),
        strategy=_strategy(stop_loss_mult=2.0),
    )
    assert proposal is not None
    assert "stop_loss" in proposal.rationale
    # Clamped to the width, not the 1.30 marketable debit.
    assert proposal.net_credit_per_spread == pytest.approx(-1.0)
