"""Wheel orchestrator strategy-aware integration smoke.

Verifies that:
- propose_all uses the strategy's params (dte band, delta band)
- the produced proposal carries strategy_id
- position lookup is filtered by strategy_id (so two strategies on the same symbol don't collide)
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from core.models import (
    OptionContract,
    OptionType,
    Position,
    PositionState,
    Quote,
    UniverseEntry,
)
from core.strategies import StrategyDefinition
from data.ivr import IVRProvider
from platforms.paper_broker import PaperBroker
from strategies.wheel import propose_all


class _NullIvr:
    async def stats(self, symbol):
        return None
    async def iv_rank(self, symbol):
        return None
    async def iv_percentile(self, symbol):
        return None


def _put(occ: str, strike: float, days_out: int, *, delta: float, mid: float) -> OptionContract:
    today = date(2025, 6, 1)
    spread = max(mid * 0.02, 0.01)
    return OptionContract(
        underlying="F", occ_symbol=occ, strike=strike,
        expiration=today + timedelta(days=days_out),
        option_type=OptionType.PUT,
        bid=mid - spread / 2, ask=mid + spread / 2,
        delta=delta,
        open_interest=1000, volume=200,
    )


def _config() -> dict:
    return {
        "account": {"id": "test", "broker": "paper"},
        "wheel": {
            "csp_delta_min": 0.20, "csp_delta_max": 0.30,
            "dte_min": 30, "dte_max": 45,
            "open_interest_min": 100, "volume_min": 50,
            "bid_ask_spread_max_pct": 10.0,
        },
    }


def _universe(strategies: list[str]) -> dict:
    return {
        "tickers": [UniverseEntry(symbol="F", name="Ford", tier=1, strategies=strategies)],
        "banned": [], "banned_rules": [],
    }


@pytest.mark.asyncio
async def test_proposal_carries_strategy_id(db_repos):
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F", bid=10.0, ask=10.04))
    broker.seed_chain("F", [_put("F-30d", 9.5, 35, delta=-0.25, mid=0.40)])
    strategy = StrategyDefinition(
        id="monthly_wheel", display_name="Monthly", type="wheel",
        enabled=True, max_concurrent=4,
        params={"dte_min": 30, "dte_max": 45, "csp_delta_min": 0.20, "csp_delta_max": 0.30},
    )
    proposals = await propose_all(
        broker, db_repos, _config(), _universe(["monthly_wheel"]),
        _NullIvr(), today=date(2025, 6, 1), strategy=strategy,
    )
    assert len(proposals) == 1
    assert proposals[0].strategy_id == "monthly_wheel"


@pytest.mark.asyncio
async def test_strategy_dte_band_overrides_base_config(db_repos):
    """Weekly strategy should pick a 7-14 DTE contract; the 35-DTE one in the
    chain is filtered out even though base config says 30-45."""
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F", bid=10.0, ask=10.04))
    broker.seed_chain(
        "F",
        [
            _put("F-7d", 9.5, 10, delta=-0.20, mid=0.20),     # weekly band
            _put("F-30d", 9.5, 35, delta=-0.25, mid=0.40),    # monthly band
        ],
    )
    weekly = StrategyDefinition(
        id="weekly_wheel", display_name="Weekly", type="wheel",
        enabled=True, max_concurrent=4,
        params={"dte_min": 7, "dte_max": 14, "csp_delta_min": 0.15, "csp_delta_max": 0.30},
    )
    proposals = await propose_all(
        broker, db_repos, _config(), _universe(["weekly_wheel"]),
        _NullIvr(), today=date(2025, 6, 1), strategy=weekly,
    )
    assert len(proposals) == 1
    assert proposals[0].contract.occ_symbol == "F-7d"


@pytest.mark.asyncio
async def test_two_strategies_on_same_symbol_dont_collide(db_repos):
    """monthly_wheel has F as CSP_OPEN; weekly_wheel sees F as IDLE. The lookup
    must be (account_id, symbol, strategy_id), not just (account_id, symbol)."""
    now = datetime.now(UTC).replace(tzinfo=None)
    await db_repos.positions.insert(
        Position(
            account_id="test", symbol="F", strategy_id="monthly_wheel",
            state=PositionState.CSP_OPEN, shares=0,
            state_changed_at=now,
        )
    )

    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F", bid=10.0, ask=10.04))
    broker.seed_chain("F", [_put("F-7d", 9.5, 10, delta=-0.20, mid=0.20)])

    weekly = StrategyDefinition(
        id="weekly_wheel", display_name="Weekly", type="wheel",
        enabled=True, max_concurrent=4,
        params={"dte_min": 7, "dte_max": 14, "csp_delta_min": 0.15, "csp_delta_max": 0.30},
    )
    proposals = await propose_all(
        broker, db_repos, _config(),
        _universe(["weekly_wheel"]),  # only weekly tagged
        _NullIvr(), today=date(2025, 6, 1), strategy=weekly,
    )
    # weekly_wheel doesn't see F as CSP_OPEN — it's a fresh slot for this strategy.
    assert len(proposals) == 1
    assert proposals[0].strategy_id == "weekly_wheel"
