"""strategies/roll_orchestrator — combine rule + LLM, halt on disagreement."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from core.models import (
    OptionContract,
    OptionType,
    Position,
    PositionState,
    UniverseEntry,
)
from core.notify import NullNotifier, set_dispatcher
from intelligence.roll_advisor_llm import LlmRollResult
from platforms.paper_broker import PaperBroker
from strategies.roll_advisor import RollAction, RollContext, RollDecision
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


def _config(llm_enabled: bool = False) -> dict:
    return {
        "account": {"id": "test"},
        "wheel": {
            "csp_delta_min": 0.20, "csp_delta_max": 0.30,
            "dte_min": 30, "dte_max": 45,
            "roll_trigger_delta": 0.50,
        },
        "intelligence": {"llm_roll_advisor_enabled": llm_enabled, "roll_advisor_models": ["m1", "m2"]},
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
        broker=broker, repos=db_repos, anthropic=None, ctx=ctx, position_id=None,
        config=_config(), universe=_universe(), today=date(2025, 6, 1),
    )
    assert outcome.action is None
    assert outcome.halted is False


@pytest.mark.asyncio
async def test_rule_only_when_llm_disabled(db_repos):
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
        broker=broker, repos=db_repos, anthropic=None, ctx=_ctx(), position_id=None,
        config=_config(llm_enabled=False), universe=_universe(), today=today,
    )
    assert outcome.action == RollAction.ROLL
    assert outcome.halted is False
    assert outcome.reason == "rule_only"


class _StubAnthropic:
    """Returns the same JSON for both models."""

    def __init__(self, decisions_per_model: dict[str, str]):
        self._mapping = decisions_per_model

    async def call(self, *, decision_type, model, system, user_payload, max_output_tokens=384, context=None):
        return {
            "parsed": {"decision": self._mapping[model], "rationale": "test"},
            "raw_text": "",
            "tokens_in": 1, "tokens_out": 1, "cost_usd": 0.0001, "decision_id": 1,
        }


@pytest.mark.asyncio
async def test_agreement_between_rule_and_llm(db_repos, _capture_notifier):
    broker = PaperBroker()
    today = date(2025, 6, 1)
    broker.seed_chain("F", [
        OptionContract(
            underlying="F", occ_symbol="ROLL1", strike=9.5,
            expiration=today + timedelta(days=35), option_type=OptionType.PUT,
            delta=-0.25, bid=2.00, ask=2.05, open_interest=1000, volume=200,
        )
    ])
    anthropic = _StubAnthropic({"m1": "ROLL", "m2": "ROLL"})
    outcome = await orch_evaluate(
        broker=broker, repos=db_repos, anthropic=anthropic,  # type: ignore[arg-type]
        ctx=_ctx(), position_id=None,
        config=_config(llm_enabled=True), universe=_universe(), today=today,
    )
    assert outcome.action == RollAction.ROLL
    assert outcome.halted is False
    assert outcome.reason == "agreed"
    assert _capture_notifier.events == []


@pytest.mark.asyncio
async def test_disagreement_halts_position_and_notifies(db_repos, _capture_notifier):
    broker = PaperBroker()
    today = date(2025, 6, 1)
    broker.seed_chain("F", [
        OptionContract(
            underlying="F", occ_symbol="ROLL1", strike=9.5,
            expiration=today + timedelta(days=35), option_type=OptionType.PUT,
            delta=-0.25, bid=2.00, ask=2.05, open_interest=1000, volume=200,
        )
    ])
    pos_id = await db_repos.positions.insert(
        Position(
            account_id="test", symbol="F", state=PositionState.CSP_OPEN,
            shares=0, current_cycle_id=None,
            state_changed_at=datetime.now(UTC).replace(tzinfo=None),
        )
    )
    anthropic = _StubAnthropic({"m1": "LET_ASSIGN", "m2": "LET_ASSIGN"})
    outcome = await orch_evaluate(
        broker=broker, repos=db_repos, anthropic=anthropic,  # type: ignore[arg-type]
        ctx=_ctx(), position_id=pos_id,
        config=_config(llm_enabled=True), universe=_universe(), today=today,
    )
    assert outcome.action is None
    assert outcome.halted is True

    pos = await db_repos.positions.get(pos_id)
    assert pos.state == PositionState.MANUAL_INTERVENTION

    assert any(e.event_type == "roll.disagreement" for e in _capture_notifier.events)
