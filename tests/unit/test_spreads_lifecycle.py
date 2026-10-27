"""Multi-leg spread lifecycle: reconciler fills + expiration + closes + P&L.

zero_dte is the only multi-leg strategy since the 2026-10-27 cleanup, so the
packages here carry its strategy id; the reconciler path is the same one its
same-day bull put spreads fill through.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from core.models import (
    OptionType,
    Order,
    OrderLeg,
    OrderStatus,
    OrderType,
    PositionState,
    Quote,
    UniverseEntry,
    WheelCycle,
)
from core.strategies import StrategyDefinition
from execution.reconciler import Reconciler
from execution.router import OrderRouter
from platforms.paper_broker import PaperBroker
from strategies.spreads import MultiLegProposal
from strategies.zero_dte import propose_close_for_symbol

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


def _zero_dte_strategy() -> StrategyDefinition:
    return StrategyDefinition(
        id="zero_dte", display_name="0DTE", type="zero_dte",
        enabled=True, max_concurrent=2,
        params={"profit_close_pct": 50, "flatten_et": "15:00"},
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
        rationale="zero_dte test",
        strategy_id="zero_dte",
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
        "test", "F", strategy_id="zero_dte"
    )
    assert pos is not None
    assert pos.state == PositionState.SPREAD_OPEN
    assert pos.current_cycle_id is not None

    cycle = await db_repos.cycles.get(pos.current_cycle_id)
    assert cycle is not None
    assert cycle.strategy_id == "zero_dte"
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

    pos = await db_repos.positions.get_by_symbol("test", "F", strategy_id="zero_dte")
    assert pos.state == PositionState.SPREAD_OPEN
    cycle_id = pos.current_cycle_id

    # Strand it: force SPREAD_PENDING while the legs are still open at the broker
    # and no order is in flight (the open FILLED earlier) — the exact stuck shape.
    await db_repos.positions.update(pos.id, state=PositionState.SPREAD_PENDING.value)

    await reconciler.reconcile_once()
    healed = await db_repos.positions.get_by_symbol("test", "F", strategy_id="zero_dte")
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
    pos_open = await db_repos.positions.get_by_symbol("test", "F", strategy_id="zero_dte")
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
        strategy_id="zero_dte",
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
        "test", "F", strategy_id="zero_dte"
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
        "test", "F", strategy_id="zero_dte"
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
        "test", "F", strategy_id="zero_dte"
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
        strategy_id="zero_dte",
        order_type=OrderType.MULTI_LEG_CLOSE,
    )
    close_result = await router.place_multi_leg(
        close_proposal, sleep=_noop_sleep, today=date(2025, 6, 5),
    )
    assert close_result.placed is not None
    # The close placement itself must NOT pend the position (the fix).
    pos_mid = await db_repos.positions.get_by_symbol(
        "test", "F", strategy_id="zero_dte"
    )
    assert pos_mid.state == PositionState.SPREAD_OPEN
    await broker.cancel_order(close_result.placed.broker_order_id)

    # 3. Reconcile → still SPREAD_OPEN, cycle intact; no restore was needed
    # so the cancellation is a defensive no-op (counter stays 0).
    summary = await reconciler.reconcile_once()
    assert summary.cancellations_processed == 0
    pos_after = await db_repos.positions.get_by_symbol(
        "test", "F", strategy_id="zero_dte"
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
        "test", "F", strategy_id="zero_dte"
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
        "test", "F", strategy_id="zero_dte"
    )
    assert pos is not None
    assert pos.state == PositionState.IDLE
    assert pos.current_cycle_id is None


# -- PnL math --------------------------------------------------------------


@pytest.mark.asyncio
async def test_compute_cycle_pnl_handles_multi_leg_credit_and_debit(db_repos):
    """Open at +0.40 credit, close at -0.15 debit, qty 3 → P&L = (0.40 - 0.15) * 100 * 3 = 75."""
    cycle_id = await db_repos.cycles.insert(
        WheelCycle(
            account_id="test",
            symbol="F",
            strategy_id="zero_dte",
            started_at=datetime.now(UTC).replace(tzinfo=None),
            initial_capital_at_risk=180.0,
        )
    )
    placed_at = datetime.now(UTC).replace(tzinfo=None)
    await db_repos.orders.insert(
        Order(
            account_id="test",
            symbol="F",
            strategy_id="zero_dte",
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
            strategy_id="zero_dte",
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
    """A 4-leg package on F with $5 wings: 90/95P + 105/110C @ 35 DTE."""
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

    For a 4-leg package 90/95P + 105/110C the legs-derived width is 20
    (outer span), but the wing width is 5. capital_at_risk should be
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
            strategy_id="zero_dte",
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
            strategy_id="zero_dte",
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
        strategy_id="zero_dte",
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
    debit on a $5 structure (-$470 realized vs -$365 theoretical max).
    zero_dte's 15:00 flatten goes through the same clamp."""
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

    from zoneinfo import ZoneInfo

    flatten = datetime(2025, 6, 10, 15, 0, tzinfo=ZoneInfo("America/New_York"))
    proposal = await propose_close_for_symbol(
        broker, db_repos, "F", _config(), strategy=_zero_dte_strategy(), now=flatten,
    )
    assert proposal is not None
    assert "HARD FLATTEN" in proposal.rationale
    # Clamped to the width, not the 1.30 marketable debit.
    assert proposal.net_credit_per_spread == pytest.approx(-1.0)
