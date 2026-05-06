"""strategies/roll_advisor — rule-based decisions."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from core.models import OptionContract, OptionType, UniverseEntry
from platforms.paper_broker import PaperBroker
from strategies.roll_advisor import RollAction, RollContext, evaluate_roll


def _put_short(strike: float = 10.0, days: int = 7, delta: float = -0.55) -> OptionContract:
    today = date(2025, 6, 1)
    return OptionContract(
        underlying="F",
        occ_symbol="F250608P00010000",
        strike=strike,
        expiration=today + timedelta(days=days),
        option_type=OptionType.PUT,
        delta=delta,
        bid=0.49,
        ask=0.51,
    )


def _put_chain_candidate(strike: float, days_out: int, mid: float, delta: float = -0.25):
    today = date(2025, 6, 1)
    return OptionContract(
        underlying="F",
        occ_symbol=f"FROLL{strike}",
        strike=strike,
        expiration=today + timedelta(days=days_out),
        option_type=OptionType.PUT,
        delta=delta,
        bid=mid - 0.01,
        ask=mid + 0.01,
        open_interest=1000,
        volume=200,
    )


def _config(**wheel) -> dict:
    base = {
        "account": {"id": "test", "broker": "paper"},
        "wheel": {
            "csp_delta_min": 0.20,
            "csp_delta_max": 0.30,
            "cc_delta_min": 0.20,
            "cc_delta_max": 0.30,
            "dte_min": 30,
            "dte_max": 45,
            "open_interest_min": 100,
            "volume_min": 50,
            "bid_ask_spread_max_pct": 10.0,
            "roll_trigger_delta": 0.50,
            "roll_only_for_credit": True,
        },
    }
    base["wheel"].update(wheel)
    return base


def _universe() -> dict:
    return {"tickers": [UniverseEntry(symbol="F", name="Ford", tier=1, overrides={})], "banned": [], "banned_rules": []}


@pytest.mark.asyncio
async def test_below_trigger_returns_none():
    broker = PaperBroker()
    ctx = RollContext(
        symbol="F",
        short_contract=_put_short(delta=-0.30),  # below trigger
        short_quantity=1,
        short_premium_collected_per_share=0.50,
        current_short_mid=0.55,
        underlying_price=10.0,
    )
    decision = await evaluate_roll(broker=broker, ctx=ctx, config=_config(), universe=_universe(), today=date(2025, 6, 1))
    assert decision is None


@pytest.mark.asyncio
async def test_credit_roll_picked_when_available():
    broker = PaperBroker()
    broker.seed_chain(
        "F",
        [
            _put_chain_candidate(strike=9.5, days_out=35, mid=2.00, delta=-0.25),
            _put_chain_candidate(strike=10.0, days_out=35, mid=2.20, delta=-0.27),
        ],
    )
    ctx = RollContext(
        symbol="F",
        short_contract=_put_short(),  # current cost-to-close mid 1.50/share
        short_quantity=1,
        short_premium_collected_per_share=0.50,
        current_short_mid=1.50,
        underlying_price=9.5,
    )
    decision = await evaluate_roll(broker=broker, ctx=ctx, config=_config(), universe=_universe(), today=date(2025, 6, 1))
    assert decision is not None
    assert decision.action == RollAction.ROLL
    assert decision.new_contract is not None
    assert decision.expected_credit_per_share is not None and decision.expected_credit_per_share > 0


@pytest.mark.asyncio
async def test_let_assign_when_no_credit_roll_for_put():
    broker = PaperBroker()
    # Only debit-roll candidates available.
    broker.seed_chain(
        "F",
        [_put_chain_candidate(strike=9.5, days_out=35, mid=0.30, delta=-0.20)],
    )
    ctx = RollContext(
        symbol="F",
        short_contract=_put_short(),
        short_quantity=1,
        short_premium_collected_per_share=0.50,
        current_short_mid=2.00,  # huge debit to close → no credit roll
        underlying_price=8.0,
    )
    decision = await evaluate_roll(broker=broker, ctx=ctx, config=_config(), universe=_universe(), today=date(2025, 6, 1))
    assert decision is not None
    assert decision.action == RollAction.LET_ASSIGN


@pytest.mark.asyncio
async def test_close_when_no_credit_roll_for_call():
    broker = PaperBroker()
    today = date(2025, 6, 1)
    cc_short = OptionContract(
        underlying="F",
        occ_symbol="F250608C00010000",
        strike=10.0,
        expiration=today + timedelta(days=7),
        option_type=OptionType.CALL,
        delta=0.55,
        bid=0.49,
        ask=0.51,
    )
    # No credit-roll candidates.
    broker.seed_chain("F", [
        OptionContract(
            underlying="F",
            occ_symbol="FROLL_CALL",
            strike=11.0,
            expiration=today + timedelta(days=35),
            option_type=OptionType.CALL,
            delta=0.25,
            bid=0.10,
            ask=0.12,
            open_interest=1000,
            volume=200,
        )
    ])
    ctx = RollContext(
        symbol="F",
        short_contract=cc_short,
        short_quantity=1,
        short_premium_collected_per_share=0.30,
        current_short_mid=1.20,
        underlying_price=11.0,
    )
    decision = await evaluate_roll(broker=broker, ctx=ctx, config=_config(), universe=_universe(), today=date(2025, 6, 1))
    assert decision is not None
    assert decision.action == RollAction.CLOSE
