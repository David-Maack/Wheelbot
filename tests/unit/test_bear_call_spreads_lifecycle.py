"""Bear call spread end-to-end: orchestrator dispatch, reconciler, close path.

The reconciler / multi-leg state machine is exercised in
test_spreads_lifecycle.py with put-side legs. These tests cover the new
dispatch points: orchestrator selects the call selector when
strategy.params.direction == "bear_call", and the close orchestrator
infers direction="bear_call" from the leg option_type.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from core.models import (
    OptionContract,
    OptionType,
    OrderLeg,
    OrderType,
    PositionState,
    Quote,
    UniverseEntry,
)
from core.strategies import StrategyDefinition
from execution.reconciler import Reconciler
from execution.router import OrderRouter
from platforms.paper_broker import PaperBroker
from strategies.spreads import (
    MultiLegProposal,
    propose_close_for_symbol,
    propose_for_symbol,
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
        "direction": "bear_call",
        "dte_min": 30,
        "dte_max": 45,
        "short_delta_min": 0.20,
        "short_delta_max": 0.30,
        "spread_width_dollars": 1.0,
        "min_credit_pct_of_width": 25.0,
        "profit_close_pct": 50,
        "time_close_dte": 7,
        "max_capital_per_spread_usd": 250,
    }
    params.update(overrides)
    return StrategyDefinition(
        id="bear_call_spread",
        display_name="Bear Call Spread",
        type="vertical_spread",
        enabled=True,
        max_concurrent=4,
        params=params,
    )


def _call(strike: float, *, bid: float, ask: float, delta: float = 0.25) -> OptionContract:
    today = date(2025, 6, 1)
    occ = f"F250706C{int(strike * 1000):08d}"
    return OptionContract(
        underlying="F",
        occ_symbol=occ,
        strike=strike,
        expiration=today + timedelta(days=35),
        option_type=OptionType.CALL,
        bid=bid,
        ask=ask,
        delta=delta,
        open_interest=1000,
        volume=200,
    )


def _bear_call_legs() -> list[OrderLeg]:
    today = date(2025, 6, 1)
    return [
        OrderLeg(
            contract_symbol="F250706C00010000",
            underlying="F",
            option_type=OptionType.CALL,
            strike=10.0,
            expiration=today + timedelta(days=35),
            action=OrderType.SELL_TO_OPEN,
        ),
        OrderLeg(
            contract_symbol="F250706C00011000",
            underlying="F",
            option_type=OptionType.CALL,
            strike=11.0,
            expiration=today + timedelta(days=35),
            action=OrderType.BUY_TO_OPEN,
        ),
    ]


def _bear_call_proposal(qty: int = 1) -> MultiLegProposal:
    return MultiLegProposal(
        symbol="F",
        legs=_bear_call_legs(),
        net_credit_per_spread=0.30,
        max_loss_per_spread=70.0,
        width_dollars=1.0,
        quantity=qty,
        rationale="bear_call_spread test",
        strategy_id="bear_call_spread",
        direction="bear_call",
    )


async def _noop_sleep(_: float) -> None:
    return None


@pytest.fixture(autouse=True)
def _stub_earnings(monkeypatch):
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)


class _FakePositionsRepo:
    """Minimal stand-in for propose_for_symbol — no in-flight position."""

    async def get_by_symbol(self, account_id, symbol, strategy_id=None):
        return None


class _FakeRepos:
    def __init__(self):
        self.positions = _FakePositionsRepo()


# -- Orchestrator dispatch --------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_dispatches_to_bear_call_selector_when_direction_set():
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F", bid=10.0, ask=10.04))
    broker.seed_chain(
        "F",
        [
            _call(10.0, bid=0.39, ask=0.41, delta=0.25),  # short
            _call(11.0, bid=0.10, ask=0.12, delta=0.10),  # long
        ],
    )

    proposal = await propose_for_symbol(
        broker,
        _FakeRepos(),
        "F",
        config={"account": {"id": "test"}},
        universe=_universe(),
        today=date(2025, 6, 1),
        strategy=_strategy(),
    )

    assert proposal is not None
    assert proposal.direction == "bear_call"
    assert proposal.strategy_id == "bear_call_spread"
    # Short at the lower strike, long at the higher strike — call-spread layout.
    short_leg = proposal.legs[0]
    long_leg = proposal.legs[1]
    assert short_leg.action == OrderType.SELL_TO_OPEN
    assert short_leg.option_type == OptionType.CALL
    assert short_leg.strike == 10.0
    assert long_leg.option_type == OptionType.CALL
    assert long_leg.strike == 11.0
    assert "bear_call" in proposal.rationale


@pytest.mark.asyncio
async def test_orchestrator_defaults_to_bull_put_when_direction_unset():
    """Backwards-compat: existing put_spread configs (no direction key) keep working."""
    from core.models import OptionType as OT

    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F", bid=10.0, ask=10.04))

    # Seed a put-side chain — no direction key in params.
    def _put(strike, bid, ask, delta=-0.25):
        return OptionContract(
            underlying="F",
            occ_symbol=f"F250706P{int(strike * 1000):08d}",
            strike=strike,
            expiration=date(2025, 6, 1) + timedelta(days=35),
            option_type=OT.PUT,
            bid=bid,
            ask=ask,
            delta=delta,
            open_interest=1000,
            volume=200,
        )

    broker.seed_chain("F", [_put(10.0, 0.39, 0.41), _put(9.0, 0.10, 0.12, delta=-0.10)])

    legacy_strategy = StrategyDefinition(
        id="put_spread",
        display_name="Bull Put Spread",
        type="vertical_spread",
        enabled=True,
        max_concurrent=4,
        params={
            "dte_min": 30, "dte_max": 45,
            "short_delta_min": 0.20, "short_delta_max": 0.30,
            "spread_width_dollars": 1.0, "min_credit_pct_of_width": 25.0,
        },  # no `direction` key
    )
    proposal = await propose_for_symbol(
        broker, _FakeRepos(), "F",
        config={"account": {"id": "test"}},
        universe=_universe(),
        today=date(2025, 6, 1),
        strategy=legacy_strategy,
    )
    assert proposal is not None
    assert proposal.direction == "bull_put"


# -- Reconciler with call legs (smoke test, full path lives in test_spreads_lifecycle) --


@pytest.mark.asyncio
async def test_reconciler_handles_bear_call_open_fill(db_repos):
    broker = PaperBroker(cash=20_000)
    router = OrderRouter(broker, db_repos, _config(), _universe())
    result = await router.place_multi_leg(
        _bear_call_proposal(qty=1), sleep=_noop_sleep, today=date(2025, 6, 1),
    )
    assert result.placed is not None
    await broker.fill_multi_leg(result.placed.broker_order_id, fill_price=0.30)

    reconciler = Reconciler(broker, db_repos, _config())
    summary = await reconciler.reconcile_once()
    assert summary.fills_processed == 1
    assert summary.cycles_opened == 1

    pos = await db_repos.positions.get_by_symbol(
        "test", "F", strategy_id="bear_call_spread"
    )
    assert pos is not None
    assert pos.state == PositionState.SPREAD_OPEN
    cycle = await db_repos.cycles.get(pos.current_cycle_id)
    assert cycle is not None
    assert cycle.strategy_id == "bear_call_spread"


# -- Close orchestrator infers direction from leg option_type ---------------


@pytest.mark.asyncio
async def test_close_orchestrator_infers_bear_call_direction_from_legs(db_repos):
    broker = PaperBroker(cash=20_000)
    router = OrderRouter(broker, db_repos, _config(), _universe())
    open_result = await router.place_multi_leg(
        _bear_call_proposal(qty=1), sleep=_noop_sleep, today=date(2025, 6, 1),
    )
    await broker.fill_multi_leg(open_result.placed.broker_order_id, fill_price=0.30)
    reconciler = Reconciler(broker, db_repos, _config())
    await reconciler.reconcile_once()

    # Quote each leg cheap → debit-to-close hits the 50% profit target.
    broker.seed_quote(Quote(symbol="F250706C00010000", bid=0.12, ask=0.14))
    broker.seed_quote(Quote(symbol="F250706C00011000", bid=0.02, ask=0.04))

    proposal = await propose_close_for_symbol(
        broker, db_repos, "F", _config(),
        today=date(2025, 6, 10), strategy=_strategy(),
    )
    assert proposal is not None
    assert proposal.order_type == OrderType.MULTI_LEG_CLOSE
    # Direction inference from leg option_type.
    assert proposal.direction == "bear_call"
    assert "bear_call_close" in proposal.rationale
    # Order price is MARKETABLE: buy short at ask 0.14, sell long at bid 0.02 →
    # debit 0.12 → net credit -0.12 (the 0.10 mid still drives the trigger).
    assert proposal.net_credit_per_spread == pytest.approx(-0.12)
    actions = sorted(str(leg.action) for leg in proposal.legs)
    assert actions == sorted(
        [OrderType.BUY_TO_CLOSE.value, OrderType.SELL_TO_CLOSE.value]
    )
