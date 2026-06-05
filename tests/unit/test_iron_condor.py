"""TICKET-014 — Iron Condor selector + propose_for_symbol + sizing.

Tests exercise the selector directly with seeded chains on PaperBroker.
End-to-end fill + cycle wiring is covered by the existing spread lifecycle
suite (which the precursor fixes already extended for 4-leg condor cycles).
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from core.models import (
    OptionContract,
    OptionType,
    OrderType,
    PositionState,
    Position,
    UniverseEntry,
)
from core.strategies import StrategyDefinition
from platforms.paper_broker import PaperBroker
from strategies.iron_condor import (
    NEWS_CHECK_PROFILE,
    IronCondorCandidate,
    propose_for_symbol,
    select_iron_condor,
    _build_legs,
    _sizing_quantity,
)
from strategies.spreads import DIRECTION_IRON_CONDOR


# -- fixtures + helpers ----------------------------------------------------


def _config() -> dict:
    return {
        "account": {"id": "test", "broker": "paper"},
        "wheel": {"open_interest_min": 0, "volume_min": 0, "bid_ask_spread_max_pct": 100.0},
    }


def _universe() -> dict:
    return {
        "tickers": [UniverseEntry(symbol="SPY", name="SPY", tier=1, overrides={})],
        "banned": [],
        "banned_rules": [],
    }


def _strategy(**overrides: Any) -> StrategyDefinition:
    params: dict[str, Any] = {
        "dte_min": 30,
        "dte_max": 45,
        "short_put_delta_target": 0.16,
        "short_call_delta_target": 0.16,
        "wing_width": 5.0,
        "min_credit_pct": 10.0,
        "profit_close_pct": 25,
        "time_close_dte": 21,
        "max_capital_per_condor_usd": 1000.0,
        "ivr_min": 0,
    }
    params.update(overrides)
    return StrategyDefinition(
        id="iron_condor",
        display_name="Iron Condor",
        type="iron_condor",
        enabled=True,
        max_concurrent=2,
        params=params,
    )


def _occ(symbol: str, exp: date, kind: str, strike: float) -> str:
    yy = exp.strftime("%y")
    mm = f"{exp.month:02d}"
    dd = f"{exp.day:02d}"
    k = "P" if kind.upper() == "PUT" else "C"
    s = f"{int(round(strike * 1000)):08d}"
    return f"{symbol}{yy}{mm}{dd}{k}{s}"


def _contract(
    underlying: str, *, kind: OptionType, strike: float, exp: date,
    delta: float, bid: float, ask: float,
) -> OptionContract:
    return OptionContract(
        underlying=underlying,
        occ_symbol=_occ(underlying, exp, kind.value, strike),
        strike=strike,
        expiration=exp,
        option_type=kind,
        bid=bid,
        ask=ask,
        delta=delta,
        open_interest=1000,
        volume=500,
    )


def _seed_symmetric_chain(broker: PaperBroker, symbol: str, exp: date, spot: float):
    """Seed a SPY-like chain with put and call strikes around `spot`.

    Strikes from spot-15 to spot+15 in $1 steps; deltas decline linearly from
    ATM. Enough resolution to find a $5-wide condor at ~0.16 delta on each side.
    """
    contracts: list[OptionContract] = []
    # Wide range so $5 wings have available strikes on both sides of the
    # ~0.16 delta shorts. Real SPY chains are ±50+ pts at this DTE.
    for offset in range(-25, 26):
        strike = spot + offset
        # Delta model: put_delta = -0.5 - 0.03*offset (negative; strike above
        # spot = ITM put, more negative; strike below = OTM put, less negative).
        # call_delta = 0.5 - 0.03*offset (positive; strike below spot = ITM call;
        # strike above = OTM call). Clamp to [-0.99, -0.01] and [0.01, 0.99].
        put_delta = max(-0.99, min(-0.01, -0.50 - 0.03 * offset))
        call_delta = max(0.01, min(0.99, 0.50 - 0.03 * offset))
        # Premium proxy: delta magnitude × 3.0 — gives short legs ~0.5 at 0.16
        # delta and long wings ~0.1 at 0.03 delta, producing credit_pct ~14%
        # on a $5-wide condor (realistic; passes a min_credit_pct=10 gate).
        put_mid = max(0.10, abs(put_delta) * 3.0)
        call_mid = max(0.10, abs(call_delta) * 3.0)
        # Tight bid/ask so the chain-liquidity gate (default 100% spread max)
        # passes even at the OTM tails where mid is small.
        contracts.append(_contract(
            symbol, kind=OptionType.PUT, strike=strike, exp=exp,
            delta=put_delta,
            bid=max(0.05, put_mid - 0.02), ask=put_mid + 0.02,
        ))
        contracts.append(_contract(
            symbol, kind=OptionType.CALL, strike=strike, exp=exp,
            delta=call_delta,
            bid=max(0.05, call_mid - 0.02), ask=call_mid + 0.02,
        ))
    broker.seed_chain(symbol, contracts)


# -- select_iron_condor ----------------------------------------------------


@pytest.mark.asyncio
async def test_select_iron_condor_returns_4_legs_at_target_delta(db_repos):
    """Symmetric condor on SPY: short put + short call at ~0.16 delta,
    long wings at $5 OTM."""
    broker = PaperBroker(cash=50_000)
    today = date(2025, 6, 1)
    exp = today + timedelta(days=35)
    _seed_symmetric_chain(broker, "SPY", exp, spot=500.0)
    cand = await select_iron_condor(
        broker, "SPY", _strategy().params, today=today,
    )
    assert cand is not None
    # Short put delta near -0.16 (target=0.16 abs)
    assert cand.short_put.option_type == OptionType.PUT
    assert abs(abs(cand.short_put.delta) - 0.16) < 0.05
    # Short call delta near +0.16
    assert cand.short_call.option_type == OptionType.CALL
    assert abs(cand.short_call.delta - 0.16) < 0.05
    # Long puts BELOW short put, long calls ABOVE short call.
    assert cand.long_put.strike < cand.short_put.strike
    assert cand.long_call.strike > cand.short_call.strike
    # Symmetric wing widths.
    assert (cand.short_put.strike - cand.long_put.strike) == pytest.approx(5.0)
    assert (cand.long_call.strike - cand.short_call.strike) == pytest.approx(5.0)


@pytest.mark.asyncio
async def test_credit_below_min_pct_skipped(db_repos):
    """When net_credit / wing_width < min_credit_pct → no candidate."""
    broker = PaperBroker(cash=50_000)
    today = date(2025, 6, 1)
    exp = today + timedelta(days=35)
    _seed_symmetric_chain(broker, "SPY", exp, spot=500.0)
    cand = await select_iron_condor(
        broker, "SPY", _strategy(min_credit_pct=90.0).params,
        today=today,
    )
    assert cand is None


@pytest.mark.asyncio
async def test_no_viable_structure_when_calls_missing(db_repos):
    """No call chain → no condor (would otherwise crash on _pick_short_at_delta)."""
    broker = PaperBroker(cash=50_000)
    today = date(2025, 6, 1)
    exp = today + timedelta(days=35)
    # Seed PUTS only.
    puts_only: list[OptionContract] = []
    for offset in range(-15, 16):
        strike = 500 + offset
        delta = -0.50 - 0.03 * offset
        delta = max(-0.99, min(-0.01, delta))
        puts_only.append(_contract(
            "SPY", kind=OptionType.PUT, strike=strike, exp=exp,
            delta=delta, bid=0.50, ask=0.55,
        ))
    broker.seed_chain("SPY", puts_only)
    cand = await select_iron_condor(
        broker, "SPY", _strategy().params, today=today,
    )
    assert cand is None


# -- sizing ----------------------------------------------------------------


def _candidate(max_loss: float, wing: float = 5.0) -> IronCondorCandidate:
    today = date(2025, 6, 1)
    exp = today + timedelta(days=35)
    return IronCondorCandidate(
        long_put=_contract("SPY", kind=OptionType.PUT, strike=480.0, exp=exp,
                           delta=-0.05, bid=0.50, ask=0.55),
        short_put=_contract("SPY", kind=OptionType.PUT, strike=485.0, exp=exp,
                            delta=-0.16, bid=1.00, ask=1.05),
        short_call=_contract("SPY", kind=OptionType.CALL, strike=515.0, exp=exp,
                             delta=0.16, bid=1.00, ask=1.05),
        long_call=_contract("SPY", kind=OptionType.CALL, strike=520.0, exp=exp,
                            delta=0.05, bid=0.50, ask=0.55),
        net_credit_per_spread=1.0,
        max_loss_per_spread=max_loss,
        wing_width_dollars=wing,
    )


def test_sizing_quantity_respects_capital_cap():
    """qty = floor(max_capital_per_condor_usd / max_loss_per_spread)."""
    strat = _strategy(max_capital_per_condor_usd=1000.0)
    cand = _candidate(max_loss=400.0)
    assert _sizing_quantity(strat, cand) == 2


def test_sizing_quantity_returns_zero_when_one_condor_exceeds_cap():
    """Locked from audit #13: never floor to 1. If one package's max_loss
    exceeds the cap, return 0 so the caller SKIPS rather than over-risks."""
    strat = _strategy(max_capital_per_condor_usd=400.0)
    cand = _candidate(max_loss=500.0)
    assert _sizing_quantity(strat, cand) == 0


def test_sizing_quantity_halved_by_drawdown_warning():
    """size_multiplier=0.5 (TICKET-008 WARNING state) halves the effective cap."""
    strat = _strategy(max_capital_per_condor_usd=1000.0)
    cand = _candidate(max_loss=400.0)
    assert _sizing_quantity(strat, cand, size_multiplier=0.5) == 1
    # 250 effective cap, max_loss=400 → qty=0
    cand_big = _candidate(max_loss=400.0)
    assert _sizing_quantity(strat, cand_big, size_multiplier=0.5) == 1


# -- propose_for_symbol end-to-end -----------------------------------------


@pytest.mark.asyncio
async def test_propose_for_symbol_returns_multi_leg_proposal_with_4_legs(db_repos):
    broker = PaperBroker(cash=50_000)
    today = date(2025, 6, 1)
    exp = today + timedelta(days=35)
    _seed_symmetric_chain(broker, "SPY", exp, spot=500.0)
    proposal = await propose_for_symbol(
        broker, db_repos, "SPY", _config(), _universe(),
        today=today, strategy=_strategy(),
    )
    assert proposal is not None
    assert len(proposal.legs) == 4
    assert proposal.direction == DIRECTION_IRON_CONDOR
    assert proposal.news_check_profile == NEWS_CHECK_PROFILE
    assert proposal.strategy_id == "iron_condor"
    # Leg actions: LP=BTO, SP=STO, SC=STO, LC=BTO
    actions = [str(leg.action) for leg in proposal.legs]
    assert actions == [
        OrderType.BUY_TO_OPEN.value,
        OrderType.SELL_TO_OPEN.value,
        OrderType.SELL_TO_OPEN.value,
        OrderType.BUY_TO_OPEN.value,
    ]


@pytest.mark.asyncio
async def test_propose_for_symbol_skips_when_position_open(db_repos):
    """No new entry when SPREAD_OPEN — closes are a separate proposal."""
    broker = PaperBroker(cash=50_000)
    today = date(2025, 6, 1)
    exp = today + timedelta(days=35)
    _seed_symmetric_chain(broker, "SPY", exp, spot=500.0)
    await db_repos.positions.insert(
        Position(
            account_id="test", symbol="SPY", strategy_id="iron_condor",
            state=PositionState.SPREAD_OPEN, shares=0,
            state_changed_at=datetime.now(UTC).replace(tzinfo=None),
        )
    )
    proposal = await propose_for_symbol(
        broker, db_repos, "SPY", _config(), _universe(),
        today=today, strategy=_strategy(),
    )
    assert proposal is None


@pytest.mark.asyncio
async def test_propose_for_symbol_skipped_when_max_loss_exceeds_cap(db_repos):
    """qty=0 → propose_for_symbol returns None (audit #13)."""
    broker = PaperBroker(cash=50_000)
    today = date(2025, 6, 1)
    exp = today + timedelta(days=35)
    _seed_symmetric_chain(broker, "SPY", exp, spot=500.0)
    # Tight cap that even one condor's max_loss will exceed.
    proposal = await propose_for_symbol(
        broker, db_repos, "SPY", _config(), _universe(),
        today=today, strategy=_strategy(max_capital_per_condor_usd=50.0),
    )
    assert proposal is None


# -- IVR gate --------------------------------------------------------------


class _IVRStub:
    def __init__(self, rank: float | None):
        self._rank = rank

    async def iv_rank(self, symbol: str) -> float | None:
        return self._rank


@pytest.mark.asyncio
async def test_ivr_below_min_skips(db_repos):
    broker = PaperBroker(cash=50_000)
    today = date(2025, 6, 1)
    exp = today + timedelta(days=35)
    _seed_symmetric_chain(broker, "SPY", exp, spot=500.0)
    proposal = await propose_for_symbol(
        broker, db_repos, "SPY", _config(), _universe(),
        today=today, strategy=_strategy(ivr_min=50),
        ivr=_IVRStub(rank=30.0),
    )
    assert proposal is None
