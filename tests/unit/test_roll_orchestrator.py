"""strategies/roll_orchestrator — combine rule + LLM, halt on disagreement."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from core.models import (
    OptionContract,
    OptionType,
    UniverseEntry,
)
from core.notify import NullNotifier, set_dispatcher
from platforms.paper_broker import PaperBroker
from strategies.roll_advisor import RollAction, RollContext
from strategies.roll_orchestrator import evaluate as orch_evaluate


@pytest.fixture(autouse=True)
def _capture_notifier():
    from core.notify import Event

    class Capture(NullNotifier):
        def __init__(self):
            self.events: list[Event] = []

        async def send(self, event):
            self.events.append(event)

    cap = Capture()
    set_dispatcher(cap)
    yield cap
    set_dispatcher(NullNotifier())


def _put_short(delta: float = -0.55) -> OptionContract:
    today = date(2025, 6, 1)
    return OptionContract(
        underlying="F",
        occ_symbol="F250608P00010000",
        strike=10.0,
        expiration=today + timedelta(days=7),
        option_type=OptionType.PUT,
        delta=delta,
        bid=0.49,
        ask=0.51,
    )


def _ctx() -> RollContext:
    return RollContext(
        symbol="F",
        short_contract=_put_short(),
        short_quantity=1,
        short_premium_collected_per_share=0.50,
        current_short_mid=1.50,
        underlying_price=9.5,
    )


def _config() -> dict:
    return {
        "account": {"id": "test"},
        "wheel": {
            "csp_delta_min": 0.20, "csp_delta_max": 0.30,
            "dte_min": 30, "dte_max": 45,
            "roll_trigger_delta": 0.50,
        },
    }


def _universe() -> dict:
    return {"tickers": [UniverseEntry(symbol="F", name="Ford", tier=1, overrides={})], "banned": [], "banned_rules": []}


@pytest.mark.asyncio
async def test_below_trigger_returns_no_action(db_repos):
    broker = PaperBroker()
    ctx = RollContext(
        symbol="F",
        short_contract=_put_short(delta=-0.30),  # below trigger
        short_quantity=1,
        short_premium_collected_per_share=0.50,
        current_short_mid=0.55,
        underlying_price=10.0,
    )
    outcome = await orch_evaluate(
        broker=broker, repos=db_repos, ctx=ctx, position_id=None,
        config=_config(), universe=_universe(), today=date(2025, 6, 1),
    )
    assert outcome.action is None
    assert outcome.halted is False


@pytest.mark.asyncio
async def test_rule_decides_the_roll(db_repos):
    broker = PaperBroker()
    today = date(2025, 6, 1)
    broker.seed_chain("F", [
        OptionContract(
            underlying="F", occ_symbol="ROLL1", strike=9.5,
            expiration=today + timedelta(days=35), option_type=OptionType.PUT,
            delta=-0.25, bid=2.00, ask=2.05, open_interest=1000, volume=200,
        )
    ])
    outcome = await orch_evaluate(
        broker=broker, repos=db_repos, ctx=_ctx(), position_id=None,
        config=_config(), universe=_universe(), today=today,
    )
    assert outcome.action == RollAction.ROLL
    assert outcome.halted is False
    assert outcome.reason == "rule"
