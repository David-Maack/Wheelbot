"""TICKET-015 — PMCC selectors, proposers, risk rules, news wiring (Phase B).

Reconciler lifecycle (Phase A) is covered in test_reconciler.py.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from core.models import (
    OptionContract,
    OptionType,
    Order,
    OrderStatus,
    OrderType,
    Position,
    PositionState,
    UniverseEntry,
    WheelCycle,
)
from core.strategies import StrategyDefinition
from platforms.paper_broker import PaperBroker
from strategies.pmcc import (
    long_breakeven,
    propose_for_symbol,
    select_long_call,
    select_short_call,
)


# -- helpers ---------------------------------------------------------------


def _config() -> dict:
    return {
        "account": {"id": "test", "broker": "paper"},
        "wheel": {"open_interest_min": 0, "volume_min": 0, "bid_ask_spread_max_pct": 100.0},
    }


def _universe() -> dict:
    return {
        "tickers": [UniverseEntry(symbol="AAPL", name="Apple", tier=1, overrides={})],
        "banned": [], "banned_rules": [],
    }


def _strategy(**overrides: Any) -> StrategyDefinition:
    params: dict[str, Any] = {
        "long_dte_min": 90, "long_dte_max": 180, "long_delta_target": 0.80,
        "short_dte_min": 7, "short_dte_max": 30, "short_delta_target": 0.27,
        "profit_close_pct_short": 50, "short_time_close_dte": 1, "long_roll_dte": 30,
        "max_capital_per_position_usd": 10000, "ivr_min": 0,
    }
    params.update(overrides)
    return StrategyDefinition(
        id="pmcc", display_name="PMCC", type="pmcc",
        enabled=True, max_concurrent=3, params=params,
    )


def _utc() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _occ(strike: float, exp: date) -> str:
    return f"AAPL{exp:%y%m%d}C{int(strike*1000):08d}"


def _call(strike: float, exp: date, delta: float, mid: float) -> OptionContract:
    return OptionContract(
        underlying="AAPL", occ_symbol=_occ(strike, exp), strike=strike,
        expiration=exp, option_type=OptionType.CALL, delta=delta,
        bid=mid - 0.05, ask=mid + 0.05, open_interest=2000, volume=800,
    )


def _seed_long_chain(broker: PaperBroker, today: date):
    """LEAP calls 120 DTE, deltas declining as strike rises. Spot ~150."""
    exp = today + timedelta(days=120)
    contracts = []
    # Deep ITM (low strike) = high delta + high premium.
    for strike, delta, mid in [
        (120.0, 0.92, 33.0),
        (130.0, 0.86, 24.0),
        (140.0, 0.80, 16.0),
        (150.0, 0.55, 9.0),
        (160.0, 0.35, 5.0),
    ]:
        contracts.append(_call(strike, exp, delta, mid))
    broker.seed_chain("AAPL", contracts)
    return exp


def _seed_short_chain(broker: PaperBroker, today: date):
    """Near-term calls 14 DTE around 0.27 delta. Spot ~150."""
    exp = today + timedelta(days=14)
    contracts = []
    for strike, delta, mid in [
        (155.0, 0.40, 2.5),
        (160.0, 0.27, 1.2),
        (165.0, 0.18, 0.6),
        (170.0, 0.10, 0.3),
    ]:
        contracts.append(_call(strike, exp, delta, mid))
    broker.seed_chain("AAPL", contracts)
    return exp


# -- select_long_call ------------------------------------------------------


@pytest.mark.asyncio
async def test_select_long_call_picks_80_delta_within_cap(db_repos):
    broker = PaperBroker()
    today = date(2025, 6, 1)
    _seed_long_chain(broker, today)
    long = await select_long_call(broker, "AAPL", _strategy().params, today=today)
    assert long is not None
    # Deepest ITM (highest delta) whose debit fits the $10k cap. The 120-strike
    # (delta 0.92) costs 33*100=3300 ≤ 10000, so it's the deepest within cap.
    assert long.strike == 120.0


@pytest.mark.asyncio
async def test_select_long_call_respects_capital_cap(db_repos):
    broker = PaperBroker()
    today = date(2025, 6, 1)
    _seed_long_chain(broker, today)
    # Tight $1500 cap. In-band deltas [0.70, 0.95] are the 120/130/140 strikes
    # costing 3300/2400/1600 — all exceed $1500. The cheaper 150/160 strikes
    # are out of the long delta band. So nothing qualifies → None.
    long = await select_long_call(
        broker, "AAPL", _strategy(max_capital_per_position_usd=1500).params, today=today,
    )
    assert long is None


# -- select_short_call (the hard rule) -------------------------------------


@pytest.mark.asyncio
async def test_select_short_above_breakeven(db_repos):
    broker = PaperBroker()
    today = date(2025, 6, 1)
    _seed_short_chain(broker, today)
    # Long strike 140 + debit 16 → breakeven 156. Short must be > 156.
    short = await select_short_call(broker, "AAPL", _strategy().params, 156.0, today=today)
    assert short is not None
    assert short.strike > 156.0
    # Closest-to-0.27-delta above breakeven is the 160-strike (delta 0.27).
    assert short.strike == 160.0


@pytest.mark.asyncio
async def test_short_below_breakeven_rejected(db_repos):
    broker = PaperBroker()
    today = date(2025, 6, 1)
    _seed_short_chain(broker, today)
    # Breakeven 171 — above every available short strike → None.
    short = await select_short_call(broker, "AAPL", _strategy().params, 171.0, today=today)
    assert short is None


def test_long_breakeven_math():
    o = Order(
        account_id="t", symbol="AAPL", order_type=OrderType.BUY_TO_OPEN,
        strike=140.0, fill_price=16.0, option_type=OptionType.CALL,
        quantity=1, status=OrderStatus.FILLED, placed_at=_utc(),
    )
    assert long_breakeven(o) == pytest.approx(156.0)


# -- propose_for_symbol dispatch -------------------------------------------


@pytest.mark.asyncio
async def test_propose_long_when_idle(db_repos):
    broker = PaperBroker()
    today = date(2025, 6, 1)
    _seed_long_chain(broker, today)
    # No position → IDLE → propose the long BUY_TO_OPEN with bullish_long news.
    proposal = await propose_for_symbol(
        broker, db_repos, "AAPL", _config(), _universe(),
        today=today, strategy=_strategy(),
    )
    assert proposal is not None
    assert proposal.order_type == OrderType.BUY_TO_OPEN
    assert proposal.contract.option_type == OptionType.CALL
    assert proposal.strategy_id == "pmcc"
    assert proposal.news_check_profile == "bullish_long"


@pytest.mark.asyncio
async def test_propose_short_when_long_open(db_repos):
    broker = PaperBroker()
    today = date(2025, 6, 1)
    _seed_short_chain(broker, today)
    # Seed a PMCC_LONG_OPEN position + a filled long order (strike 140, debit 16
    # → breakeven 156).
    cycle_id = await db_repos.cycles.insert(
        WheelCycle(account_id="test", symbol="AAPL", strategy_id="pmcc",
                   started_at=_utc(), n_orders=1)
    )
    await db_repos.orders.insert(
        Order(account_id="test", symbol="AAPL", strategy_id="pmcc", cycle_id=cycle_id,
              order_type=OrderType.BUY_TO_OPEN, contract_symbol="AAPL_long",
              strike=140.0, expiration=today + timedelta(days=120),
              option_type=OptionType.CALL, quantity=1, fill_price=16.0,
              status=OrderStatus.FILLED, placed_at=_utc(), client_order_id="l")
    )
    await db_repos.positions.insert(
        Position(account_id="test", symbol="AAPL", strategy_id="pmcc",
                 state=PositionState.PMCC_LONG_OPEN, shares=0,
                 current_cycle_id=cycle_id, state_changed_at=_utc())
    )
    proposal = await propose_for_symbol(
        broker, db_repos, "AAPL", _config(), _universe(),
        today=today, strategy=_strategy(),
    )
    assert proposal is not None
    assert proposal.order_type == OrderType.SELL_TO_OPEN
    assert proposal.contract.strike > 156.0   # above breakeven
    # Short sells are NOT news-checked.
    assert proposal.news_check_profile is None


@pytest.mark.asyncio
async def test_propose_none_when_both_open(db_repos):
    broker = PaperBroker()
    today = date(2025, 6, 1)
    _seed_short_chain(broker, today)
    await db_repos.positions.insert(
        Position(account_id="test", symbol="AAPL", strategy_id="pmcc",
                 state=PositionState.PMCC_BOTH_OPEN, shares=0,
                 state_changed_at=_utc())
    )
    proposal = await propose_for_symbol(
        broker, db_repos, "AAPL", _config(), _universe(),
        today=today, strategy=_strategy(),
    )
    assert proposal is None


# -- risk rules ------------------------------------------------------------


def _pmcc_proposal(order_type: OrderType, strike: float, mid: float) -> "Proposal":
    from strategies.wheel import Proposal
    today = date(2025, 6, 1)
    exp = today + timedelta(days=120 if order_type == OrderType.BUY_TO_OPEN else 14)
    contract = _call(strike, exp, 0.80 if order_type == OrderType.BUY_TO_OPEN else 0.27, mid)
    return Proposal(
        symbol="AAPL", contract=contract, order_type=order_type,
        quantity=1, rationale="pmcc test", strategy_id="pmcc",
        news_check_profile="bullish_long" if order_type == OrderType.BUY_TO_OPEN else None,
    )


def _risk_config(pmcc_cap: float = 10000) -> dict:
    cfg = _config()
    cfg["wheel"] = {
        "buying_power_floor_pct": 5, "max_position_pct_of_account": 90,
        "max_concurrent_positions": 4, "open_interest_min": 0, "volume_min": 0,
        "bid_ask_spread_max_pct": 100.0,
    }
    cfg["regime"] = {"enabled": False}
    # The cap is read from the strategies registry in production (not the
    # wheel block) — mirror that here so the rule under test hits the real
    # source path.
    cfg["strategies"] = [
        {"id": "pmcc", "params": {"max_capital_per_position_usd": pmcc_cap}},
    ]
    return cfg


@pytest.mark.asyncio
async def test_pmcc_position_cap_blocks_oversized_long(db_repos):
    from risk.limits import RiskCheckResult, RiskGate
    gate = RiskGate(PaperBroker(cash=100_000), db_repos, _risk_config(pmcc_cap=5000), _universe())
    # Long mid 60 → debit 6000; cap 5000 → fail.
    proposal = _pmcc_proposal(OrderType.BUY_TO_OPEN, 130.0, 60.0)
    result = RiskCheckResult(proposal=proposal)
    gate._rule_pmcc_position_cap(result, proposal, {})
    statuses = {r.rule: r.status for r in result.results}
    assert statuses["pmcc_position_cap"] == "fail"


@pytest.mark.asyncio
async def test_pmcc_position_cap_passes_within_cap(db_repos):
    from risk.limits import RiskCheckResult, RiskGate
    gate = RiskGate(PaperBroker(cash=100_000), db_repos, _risk_config(pmcc_cap=10000), _universe())
    proposal = _pmcc_proposal(OrderType.BUY_TO_OPEN, 130.0, 30.0)  # debit 3000
    result = RiskCheckResult(proposal=proposal)
    gate._rule_pmcc_position_cap(result, proposal, {})
    statuses = {r.rule: r.status for r in result.results}
    assert statuses["pmcc_position_cap"] == "pass"


@pytest.mark.asyncio
async def test_pmcc_long_not_macro_blocked(db_repos):
    """A PMCC long (BUY_TO_OPEN, 90-180 DTE) must NOT be macro-blackout-gated —
    its lifespan always spans a monthly macro event, which would block it
    perpetually. Macro blackout is a short-premium gate; a long isn't short
    premium. Regression for the strategy-killing bug."""
    from risk.limits import RiskCheckResult, RiskGate
    cfg = _risk_config()
    cfg["risk"] = {"macro_blackout": {"enabled": True, "event_types": ["FOMC", "CPI", "NFP"]}}
    gate = RiskGate(PaperBroker(cash=100_000), db_repos, cfg, _universe())
    proposal = _pmcc_proposal(OrderType.BUY_TO_OPEN, 130.0, 30.0)
    result = RiskCheckResult(proposal=proposal)
    await gate._rule_macro_blackout(result, proposal, date(2025, 6, 1))
    statuses = {r.rule: r.status for r in result.results}
    # Skipped (long-option open), NOT failed.
    assert statuses["macro_blackout"] == "skip"


@pytest.mark.asyncio
async def test_pmcc_short_below_breakeven_blocked_by_rule(db_repos):
    from risk.limits import RiskCheckResult, RiskGate
    # Seed a PMCC_LONG_OPEN with long strike 140 + debit 16 → breakeven 156.
    cycle_id = await db_repos.cycles.insert(
        WheelCycle(account_id="test", symbol="AAPL", strategy_id="pmcc",
                   started_at=_utc(), n_orders=1)
    )
    await db_repos.orders.insert(
        Order(account_id="test", symbol="AAPL", strategy_id="pmcc", cycle_id=cycle_id,
              order_type=OrderType.BUY_TO_OPEN, contract_symbol="AAPL_long",
              strike=140.0, expiration=date(2025, 6, 1) + timedelta(days=120),
              option_type=OptionType.CALL, quantity=1, fill_price=16.0,
              status=OrderStatus.FILLED, placed_at=_utc(), client_order_id="l")
    )
    await db_repos.positions.insert(
        Position(account_id="test", symbol="AAPL", strategy_id="pmcc",
                 state=PositionState.PMCC_LONG_OPEN, shares=0,
                 current_cycle_id=cycle_id, state_changed_at=_utc())
    )
    gate = RiskGate(PaperBroker(cash=100_000), db_repos, _risk_config(), _universe())
    # Short strike 150 is BELOW breakeven 156 → fail.
    bad_short = _pmcc_proposal(OrderType.SELL_TO_OPEN, 150.0, 2.0)
    result = RiskCheckResult(proposal=bad_short)
    await gate._rule_pmcc_short_above_breakeven(result, bad_short)
    statuses = {r.rule: r.status for r in result.results}
    assert statuses["pmcc_short_above_breakeven"] == "fail"
    # Short strike 160 is ABOVE breakeven 156 → pass.
    good_short = _pmcc_proposal(OrderType.SELL_TO_OPEN, 160.0, 1.2)
    result2 = RiskCheckResult(proposal=good_short)
    await gate._rule_pmcc_short_above_breakeven(result2, good_short)
    statuses2 = {r.rule: r.status for r in result2.results}
    assert statuses2["pmcc_short_above_breakeven"] == "pass"
