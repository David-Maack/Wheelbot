"""Bull put spread selector + orchestrator + router multi-leg path."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from core.models import (
    OptionContract,
    OptionType,
    OrderStatus,
    OrderType,
    Position,
    PositionState,
    Quote,
    UniverseEntry,
)
from core.strategies import StrategyDefinition
from execution.router import OrderRouter
from platforms.paper_broker import PaperBroker
from strategies.spread_selector import select_bull_put_spread
from strategies.spreads import MultiLegProposal, propose_for_symbol


def _strategy(**params_overrides: Any) -> StrategyDefinition:
    base_params: dict[str, Any] = {
        "dte_min": 30,
        "dte_max": 45,
        "short_delta_min": 0.20,
        "short_delta_max": 0.30,
        "spread_width_dollars": 1.0,
        "min_credit_pct_of_width": 25.0,
        "open_interest_min": 100,
        "volume_min": 50,
        "bid_ask_spread_max_pct": 10.0,
    }
    base_params.update(params_overrides)
    return StrategyDefinition(
        id="put_spread",
        display_name="Bull Put Spread",
        type="vertical_spread",
        enabled=True,
        max_concurrent=4,
        params=base_params,
    )


def _put(strike: float, *, bid: float, ask: float, delta: float = -0.25) -> OptionContract:
    today = date(2025, 6, 1)
    occ = f"F250706P{int(strike * 1000):08d}"
    return OptionContract(
        underlying="F",
        occ_symbol=occ,
        strike=strike,
        expiration=today + timedelta(days=35),
        option_type=OptionType.PUT,
        bid=bid,
        ask=ask,
        delta=delta,
        open_interest=1000,
        volume=200,
    )


def _universe() -> dict:
    return {
        "tickers": [UniverseEntry(symbol="F", name="Ford", tier=1, overrides={})],
        "banned": [],
        "banned_rules": [],
    }


class _FakePositionsRepo:
    def __init__(self, positions: dict[tuple[str, str], Position] | None = None):
        self._by_key = positions or {}

    async def get_by_symbol(
        self, account_id: str, symbol: str, strategy_id: str | None = None
    ) -> Position | None:
        return self._by_key.get((account_id, symbol))


class _FakeRepos:
    def __init__(self, positions: dict[tuple[str, str], Position] | None = None):
        self.positions = _FakePositionsRepo(positions)


# -- Selector ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_selector_picks_short_high_long_low_with_target_width():
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F", bid=10.0, ask=10.04))
    broker.seed_chain(
        "F",
        [
            _put(10.0, bid=0.39, ask=0.41, delta=-0.25),  # short candidate
            _put(9.0, bid=0.10, ask=0.12, delta=-0.10),   # long target
        ],
    )
    candidate = await select_bull_put_spread(
        broker, "F", _strategy().params, today=date(2025, 6, 1)
    )
    assert candidate is not None
    assert candidate.short.strike == 10.0
    assert candidate.long.strike == 9.0
    assert candidate.width_dollars == pytest.approx(1.0)
    # short mid 0.40, long mid 0.11 → credit 0.29; max loss = (1.0 - 0.29) * 100 = 71.
    assert candidate.net_credit_per_spread == pytest.approx(0.29)
    assert candidate.max_loss_per_spread == pytest.approx(71.0)


@pytest.mark.asyncio
async def test_selector_rejects_when_credit_below_min_pct():
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F", bid=10.0, ask=10.04))
    broker.seed_chain(
        "F",
        [
            _put(10.0, bid=0.30, ask=0.32, delta=-0.25),  # short mid 0.31
            _put(9.0, bid=0.20, ask=0.22, delta=-0.10),   # long mid 0.21 → credit 0.10 → 10% < 25%
        ],
    )
    candidate = await select_bull_put_spread(
        broker, "F", _strategy().params, today=date(2025, 6, 1)
    )
    assert candidate is None


@pytest.mark.asyncio
async def test_selector_falls_back_to_nearest_strike_when_exact_width_missing():
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F", bid=10.0, ask=10.04))
    broker.seed_chain(
        "F",
        [
            _put(10.0, bid=0.39, ask=0.41, delta=-0.25),
            # No 9.0 strike — use 8.5 (target was 9.0 with width=1.0).
            _put(8.5, bid=0.05, ask=0.07, delta=-0.05),
        ],
    )
    # Credit at the wider 1.5-strike fallback is 0.34 → 22.7% < 25% default.
    # Loosen min_credit_pct so the fallback strike survives the credit gate.
    candidate = await select_bull_put_spread(
        broker, "F", _strategy(min_credit_pct_of_width=20.0).params,
        today=date(2025, 6, 1),
    )
    assert candidate is not None
    assert candidate.long.strike == 8.5
    assert candidate.width_dollars == pytest.approx(1.5)


@pytest.mark.asyncio
async def test_selector_rejects_when_ivr_below_min():
    """Sprint 12 sub-sprint 5: IVR < ivr_min should block entry."""
    class _StubIvr:
        async def iv_rank(self, symbol):
            return 22.0  # below the 30 threshold

    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F", bid=10.0, ask=10.04))
    broker.seed_chain(
        "F",
        [
            _put(10.0, bid=0.39, ask=0.41, delta=-0.25),
            _put(9.0, bid=0.10, ask=0.12, delta=-0.10),
        ],
    )
    candidate = await select_bull_put_spread(
        broker, "F",
        _strategy(ivr_min=30).params,
        today=date(2025, 6, 1), ivr=_StubIvr(),
    )
    assert candidate is None


@pytest.mark.asyncio
async def test_selector_passes_when_ivr_unavailable():
    """No iv_history yet → iv_rank returns None → skip the filter."""
    class _StubIvr:
        async def iv_rank(self, symbol):
            return None

    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F", bid=10.0, ask=10.04))
    broker.seed_chain(
        "F",
        [
            _put(10.0, bid=0.39, ask=0.41, delta=-0.25),
            _put(9.0, bid=0.10, ask=0.12, delta=-0.10),
        ],
    )
    candidate = await select_bull_put_spread(
        broker, "F",
        _strategy(ivr_min=30).params,
        today=date(2025, 6, 1), ivr=_StubIvr(),
    )
    assert candidate is not None


@pytest.mark.asyncio
async def test_selector_returns_none_when_no_short_passes():
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F", bid=10.0, ask=10.04))
    broker.seed_chain(
        "F",
        [
            # delta out of band
            _put(10.0, bid=0.39, ask=0.41, delta=-0.05),
            _put(9.0, bid=0.10, ask=0.12, delta=-0.02),
        ],
    )
    candidate = await select_bull_put_spread(
        broker, "F", _strategy().params, today=date(2025, 6, 1)
    )
    assert candidate is None


# -- Orchestrator -----------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_builds_multi_leg_proposal_for_idle_position():
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F", bid=10.0, ask=10.04))
    broker.seed_chain(
        "F",
        [
            _put(10.0, bid=0.39, ask=0.41, delta=-0.25),
            _put(9.0, bid=0.10, ask=0.12, delta=-0.10),
        ],
    )
    repos = _FakeRepos()
    proposal = await propose_for_symbol(
        broker,
        repos,
        "F",
        config={"account": {"id": "test"}},
        universe=_universe(),
        today=date(2025, 6, 1),
        strategy=_strategy(),
    )
    assert proposal is not None
    assert isinstance(proposal, MultiLegProposal)
    assert proposal.symbol == "F"
    assert len(proposal.legs) == 2
    short_leg = proposal.legs[0]
    long_leg = proposal.legs[1]
    assert short_leg.action == OrderType.SELL_TO_OPEN
    assert long_leg.action == OrderType.BUY_TO_OPEN
    assert short_leg.strike == 10.0
    assert long_leg.strike == 9.0
    assert proposal.strategy_id == "put_spread"
    assert proposal.quantity >= 1
    # 2026-07-01 audit: bull-put OPENS are news-checked like a CSP (they carry
    # the same negative-catalyst exposure); previously profile=None bypassed
    # the check entirely.
    assert proposal.news_check_profile == "bullish_csp"


@pytest.mark.asyncio
async def test_orchestrator_skips_when_position_already_open():
    broker = PaperBroker()
    broker.seed_chain(
        "F",
        [
            _put(10.0, bid=0.39, ask=0.41, delta=-0.25),
            _put(9.0, bid=0.10, ask=0.12, delta=-0.10),
        ],
    )
    pos = Position(
        account_id="test",
        symbol="F",
        strategy_id="put_spread",
        state=PositionState.SPREAD_OPEN,
        shares=0,
        state_changed_at=datetime.now(UTC).replace(tzinfo=None),
    )
    repos = _FakeRepos({("test", "F"): pos})
    proposal = await propose_for_symbol(
        broker, repos, "F",
        config={"account": {"id": "test"}},
        universe=_universe(),
        today=date(2025, 6, 1),
        strategy=_strategy(),
    )
    assert proposal is None


@pytest.mark.asyncio
async def test_orchestrator_sizing_respects_capital_cap():
    """max_capital_per_spread_usd / max_loss → quantity."""
    broker = PaperBroker()
    broker.seed_chain(
        "F",
        [
            _put(10.0, bid=0.39, ask=0.41, delta=-0.25),
            _put(9.0, bid=0.10, ask=0.12, delta=-0.10),
        ],
    )
    # max loss is 71. Cap of 250 → 3 contracts.
    strat = _strategy(max_capital_per_spread_usd=250)
    proposal = await propose_for_symbol(
        broker, _FakeRepos(), "F",
        config={"account": {"id": "test"}},
        universe=_universe(),
        today=date(2025, 6, 1),
        strategy=strat,
    )
    assert proposal is not None
    assert proposal.quantity == 3


@pytest.mark.asyncio
async def test_orchestrator_skips_when_one_spread_exceeds_cap():
    """Finding #13: if a single spread's defined risk exceeds the per-spread
    capital cap, the orchestrator must SKIP (return None) rather than flooring
    to 1 contract and blowing through the limit."""
    broker = PaperBroker()
    broker.seed_chain(
        "F",
        [
            _put(10.0, bid=0.39, ask=0.41, delta=-0.25),
            _put(9.0, bid=0.10, ask=0.12, delta=-0.10),
        ],
    )
    # max loss is 71; cap of 50 < 71 → cannot afford even one → skip.
    strat = _strategy(max_capital_per_spread_usd=50)
    proposal = await propose_for_symbol(
        broker, _FakeRepos(), "F",
        config={"account": {"id": "test"}},
        universe=_universe(),
        today=date(2025, 6, 1),
        strategy=strat,
    )
    assert proposal is None


# -- Router multi-leg path --------------------------------------------------


def _router_config() -> dict:
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


def _spread_proposal() -> MultiLegProposal:
    short = _put(10.0, bid=0.39, ask=0.41, delta=-0.25)
    long = _put(9.0, bid=0.10, ask=0.12, delta=-0.10)
    from core.models import OrderLeg

    legs = [
        OrderLeg(
            contract_symbol=short.occ_symbol,
            underlying="F",
            option_type=OptionType.PUT,
            strike=10.0,
            expiration=short.expiration,
            action=OrderType.SELL_TO_OPEN,
        ),
        OrderLeg(
            contract_symbol=long.occ_symbol,
            underlying="F",
            option_type=OptionType.PUT,
            strike=9.0,
            expiration=long.expiration,
            action=OrderType.BUY_TO_OPEN,
        ),
    ]
    return MultiLegProposal(
        symbol="F",
        legs=legs,
        net_credit_per_spread=0.29,
        max_loss_per_spread=71.0,
        width_dollars=1.0,
        quantity=2,
        rationale="put_spread test",
        strategy_id="put_spread",
    )


async def _noop_sleep(seconds: float) -> None:
    return None


@pytest.fixture(autouse=True)
def _stub_earnings(monkeypatch):
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)


@pytest.mark.asyncio
async def test_router_places_multi_leg_via_paper_broker(db_repos):
    broker = PaperBroker(cash=20_000)
    router = OrderRouter(broker, db_repos, _router_config(), _universe())
    result = await router.place_multi_leg(
        _spread_proposal(), sleep=_noop_sleep, today=date(2025, 6, 1)
    )
    assert result.placed is not None
    assert result.placed.broker_order_id is not None
    assert result.placed.order_type == OrderType.MULTI_LEG_OPEN
    assert result.placed.quantity == 2
    # Net credit 0.29 minus the default 0.05 open_slippage → 0.24 limit.
    assert result.placed.limit_price == pytest.approx(0.24)
    pos = await db_repos.positions.get_by_symbol("test", "F", strategy_id="put_spread")
    assert pos is not None
    assert pos.state == PositionState.SPREAD_PENDING


@pytest.mark.asyncio
async def test_multi_leg_open_applies_slippage_to_limit(db_repos):
    """Open slippage concession is configurable."""
    broker = PaperBroker(cash=20_000)
    cfg = _router_config()
    cfg["execution"]["open_slippage"] = 0.10
    router = OrderRouter(broker, db_repos, cfg, _universe())
    result = await router.place_multi_leg(
        _spread_proposal(), sleep=_noop_sleep, today=date(2025, 6, 1)
    )
    # 0.29 net credit − 0.10 slippage → 0.19.
    assert result.placed.limit_price == pytest.approx(0.19)


@pytest.mark.asyncio
async def test_multi_leg_close_applies_slippage_to_limit(db_repos):
    """A spread CLOSE concedes close_slippage too so the buyback crosses —
    net_credit (a debit, negative) minus close_slippage = more negative."""
    from core.models import OrderLeg
    short = _put(10.0, bid=0.39, ask=0.41, delta=-0.25)
    long = _put(9.0, bid=0.10, ask=0.12, delta=-0.10)
    close = MultiLegProposal(
        symbol="F",
        legs=[
            OrderLeg(
                contract_symbol=short.occ_symbol, underlying="F",
                option_type=OptionType.PUT, strike=10.0,
                expiration=short.expiration, action=OrderType.BUY_TO_CLOSE,
            ),
            OrderLeg(
                contract_symbol=long.occ_symbol, underlying="F",
                option_type=OptionType.PUT, strike=9.0,
                expiration=long.expiration, action=OrderType.SELL_TO_CLOSE,
            ),
        ],
        net_credit_per_spread=-0.10,   # $0.10 debit to close
        max_loss_per_spread=71.0,
        width_dollars=1.0,
        quantity=1,
        rationale="put_spread close test",
        strategy_id="put_spread",
        order_type=OrderType.MULTI_LEG_CLOSE,
    )
    broker = PaperBroker(cash=20_000)
    cfg = _router_config()
    cfg["execution"]["close_slippage"] = 0.05
    router = OrderRouter(broker, db_repos, cfg, _universe())
    result = await router.place_multi_leg(close, sleep=_noop_sleep, today=date(2025, 6, 1))
    # -0.10 debit − 0.05 slippage → -0.15 (willing to pay a bit more to exit).
    assert result.placed.limit_price == pytest.approx(-0.15)


@pytest.mark.asyncio
async def test_router_multi_leg_idempotent_on_resubmit(db_repos):
    """Double-submit safety on multi-leg: second call within the stale-pending
    window skips submission locally instead of relying on broker dedup."""
    broker = PaperBroker(cash=20_000)
    router = OrderRouter(broker, db_repos, _router_config(), _universe())
    first = await router.place_multi_leg(
        _spread_proposal(), sleep=_noop_sleep, today=date(2025, 6, 1)
    )
    second = await router.place_multi_leg(
        _spread_proposal(), sleep=_noop_sleep, today=date(2025, 6, 1)
    )
    assert first.placed is not None
    assert second.placed is None
    assert second.skipped_duplicate_pending is True
    rows = await db_repos.orders.list_recent("test")
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_router_multi_leg_risk_failure_short_circuits(db_repos):
    broker = PaperBroker(cash=50)  # tiny account → BP floor blocks
    cfg = _router_config()
    cfg["wheel"]["buying_power_floor_pct"] = 90  # very high floor → fail
    router = OrderRouter(broker, db_repos, cfg, _universe())
    result = await router.place_multi_leg(
        _spread_proposal(), sleep=_noop_sleep, today=date(2025, 6, 1)
    )
    assert result.risk_failed is True
    assert result.placed is None
    pos = await db_repos.positions.get_by_symbol("test", "F", strategy_id="put_spread")
    assert pos is None


@pytest.mark.asyncio
async def test_router_multi_leg_dry_run_does_not_place_or_persist(db_repos):
    broker = PaperBroker(cash=20_000)
    cfg = _router_config()
    cfg["execution"]["dry_run"] = True
    router = OrderRouter(broker, db_repos, cfg, _universe())
    result = await router.place_multi_leg(
        _spread_proposal(), sleep=_noop_sleep, today=date(2025, 6, 1)
    )
    assert result.dry_run is True
    assert result.placed is None
    pos = await db_repos.positions.get_by_symbol("test", "F", strategy_id="put_spread")
    assert pos is None
