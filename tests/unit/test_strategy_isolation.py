"""scripts/run_bot._propose_and_route — per-strategy exception isolation
(2026-07-23 review fix).

Pre-fix, one strategy's propose pass raising aborted the whole tick, skipping
every downstream strategy's stops/closes (swing, last in config order, lost
the most). Now the failure logs + notifies and the loop continues.
"""

from __future__ import annotations

import pytest

from core.strategies import StrategyDefinition
from data.ivr import IVRProvider
from execution.router import OrderRouter
from platforms.paper_broker import PaperBroker
from scripts.run_bot import _propose_and_route


def _config() -> dict:
    return {"account": {"id": "test", "broker": "paper"}, "wheel": {}}


def _universe() -> dict:
    return {"tickers": [], "banned": [], "banned_rules": []}


def _strategy(id_: str, type_: str) -> StrategyDefinition:
    return StrategyDefinition(
        id=id_, display_name=id_, type=type_, enabled=True,
        max_concurrent=2, params={},
    )


@pytest.mark.asyncio
async def test_one_strategy_exception_does_not_skip_the_rest(db_repos, monkeypatch):
    calls: list[str] = []

    async def _boom(*args, **kwargs):
        raise RuntimeError("chain fetch exploded")

    async def _spread_closes(*args, **kwargs):
        calls.append("spread_closes")
        return []

    monkeypatch.setattr("scripts.run_bot.propose_all_wheel_closes", _boom)
    monkeypatch.setattr("scripts.run_bot.propose_all_spread_closes", _spread_closes)

    broker = PaperBroker(cash=20_000)
    router = OrderRouter(broker, db_repos, _config(), _universe())
    strategies = [
        _strategy("monthly_wheel", "wheel"),          # raises
        _strategy("put_spread", "vertical_spread"),   # must still run
    ]
    # Must not raise — the wheel failure is isolated.
    await _propose_and_route(
        broker=broker, repos=db_repos, router=router,
        ivr=IVRProvider(db_repos.iv_history),
        config=_config(), universe=_universe(), strategies=strategies,
        delta_unavailable_counters={},
    )
    assert calls == ["spread_closes"]
