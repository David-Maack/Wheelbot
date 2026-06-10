"""TICKET-016 — Calendar spread selector, close, risk, reconciler."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from core.models import (
    CycleOutcome,
    OptionContract,
    OptionType,
    Order,
    OrderStatus,
    OrderType,
    Position,
    PositionState,
    Quote,
    UniverseEntry,
    WheelCycle,
)
from core.strategies import StrategyDefinition
from execution.reconciler import ReconcileSummary, Reconciler
from platforms.paper_broker import PaperBroker
from strategies.calendar import propose_for_symbol, select_calendar
from strategies.calendar_close import propose_close_for_symbol


def _config() -> dict:
    return {
        "account": {"id": "test", "broker": "paper"},
        "wheel": {"open_interest_min": 0, "volume_min": 0, "bid_ask_spread_max_pct": 100.0},
    }


def _universe() -> dict:
    return {
        "tickers": [UniverseEntry(symbol="SPY", name="SPY", tier=1, overrides={})],
        "banned": [], "banned_rules": [],
    }


def _strategy(**overrides: Any) -> StrategyDefinition:
    params: dict[str, Any] = {
        "short_dte_min": 7, "short_dte_max": 21,
        "long_dte_min": 45, "long_dte_max": 75,
        "max_debit_per_spread_usd": 250, "profit_close_pct": 25,
        "time_close_short_dte": 2, "ivr_max": 35,
    }
    params.update(overrides)
    return StrategyDefinition(
        id="calendar", display_name="Calendar", type="calendar",
        enabled=True, max_concurrent=3, params=params,
    )


def _utc() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _occ(symbol: str, exp: date, strike: float) -> str:
    return f"{symbol}{exp:%y%m%d}C{int(strike*1000):08d}"


def _call(symbol: str, strike: float, exp: date, delta: float, mid: float) -> OptionContract:
    return OptionContract(
        underlying=symbol, occ_symbol=_occ(symbol, exp, strike), strike=strike,
        expiration=exp, option_type=OptionType.CALL, delta=delta,
        bid=mid - 0.05, ask=mid + 0.05, open_interest=5000, volume=2000,
    )


def _seed_calendar_chain(broker: PaperBroker, today: date, *, back_mid_at_atm=3.0, front_mid_at_atm=1.5):
    """Front (14 DTE) + back (60 DTE) calls, same strikes, ATM ~150. Back is
    pricier than front (normal term structure) → positive debit."""
    front_exp = today + timedelta(days=14)
    back_exp = today + timedelta(days=60)
    contracts = []
    for strike, delta in [(145.0, 0.70), (150.0, 0.50), (155.0, 0.30)]:
        # premium scales loosely with delta; back > front at each strike
        contracts.append(_call("SPY", strike, front_exp, delta, front_mid_at_atm * (delta / 0.5)))
        contracts.append(_call("SPY", strike, back_exp, delta, back_mid_at_atm * (delta / 0.5)))
    broker.seed_chain("SPY", contracts)
    return front_exp, back_exp


# -- select_calendar -------------------------------------------------------


@pytest.mark.asyncio
async def test_select_atm_calendar_two_expirations(db_repos):
    broker = PaperBroker()
    today = date(2025, 6, 1)
    front_exp, back_exp = _seed_calendar_chain(broker, today)
    picked = await select_calendar(broker, "SPY", _strategy().params, today=today)
    assert picked is not None
    short_front, long_back, net_debit = picked
    # ATM strike (delta closest to 0.50) is 150.
    assert short_front.strike == 150.0
    assert long_back.strike == 150.0
    # Different expirations: front 14 DTE, back 60 DTE.
    assert short_front.expiration == front_exp
    assert long_back.expiration == back_exp
    # Net debit positive (back pricier than front), within the $250 cap.
    assert net_debit > 0
    assert net_debit * 100 <= 250


@pytest.mark.asyncio
async def test_debit_above_cap_rejected(db_repos):
    broker = PaperBroker()
    today = date(2025, 6, 1)
    # Back hugely pricier → debit far above the $250 cap.
    _seed_calendar_chain(broker, today, back_mid_at_atm=8.0, front_mid_at_atm=1.0)
    picked = await select_calendar(broker, "SPY", _strategy(max_debit_per_spread_usd=250).params, today=today)
    assert picked is None


class _IVRStub:
    def __init__(self, rank): self._rank = rank
    async def iv_rank(self, symbol): return self._rank


@pytest.mark.asyncio
async def test_ivr_above_max_blocks_entry(db_repos):
    """INVERTED gate: calendars want LOW IV. IVR above ivr_max → skip."""
    broker = PaperBroker()
    today = date(2025, 6, 1)
    _seed_calendar_chain(broker, today)
    picked = await select_calendar(
        broker, "SPY", _strategy(ivr_max=35).params, today=today, ivr=_IVRStub(rank=60.0),
    )
    assert picked is None
    # Low IV (below max) → allowed.
    picked2 = await select_calendar(
        broker, "SPY", _strategy(ivr_max=35).params, today=today, ivr=_IVRStub(rank=20.0),
    )
    assert picked2 is not None


@pytest.mark.asyncio
async def test_propose_for_symbol_builds_calendar_mleg(db_repos):
    broker = PaperBroker()
    today = date(2025, 6, 1)
    _seed_calendar_chain(broker, today)
    proposal = await propose_for_symbol(
        broker, db_repos, "SPY", _config(), _universe(), today=today, strategy=_strategy(),
    )
    assert proposal is not None
    assert len(proposal.legs) == 2
    assert proposal.direction == "calendar"
    assert proposal.news_check_profile == "neutral_pin"
    assert proposal.width_dollars == 0.0
    # Net DEBIT → negative net_credit_per_spread.
    assert proposal.net_credit_per_spread < 0
    # Front sells, back buys.
    actions = sorted(str(leg.action) for leg in proposal.legs)
    assert actions == sorted([OrderType.SELL_TO_OPEN.value, OrderType.BUY_TO_OPEN.value])
    # Same strike, different expirations.
    strikes = {leg.strike for leg in proposal.legs}
    exps = {leg.expiration for leg in proposal.legs}
    assert len(strikes) == 1
    assert len(exps) == 2


# -- capital-at-risk math (width_dollars=0 → capital = debit) ---------------


@pytest.mark.asyncio
async def test_calendar_cycle_capital_at_risk_equals_debit(db_repos):
    """A calendar MLEG_OPEN with width_dollars=0 and a NEGATIVE fill_price
    (debit) records initial_capital_at_risk = debit * 100, via the existing
    _open_cycle_for_spread formula (width − net_credit) = (0 − (−debit))."""
    today = date(2025, 6, 1)
    exp_f = today + timedelta(days=14)
    exp_b = today + timedelta(days=60)
    legs = [
        {"contract_symbol": _occ("SPY", exp_f, 150.0), "underlying": "SPY",
         "option_type": "CALL", "strike": 150.0, "expiration": exp_f.isoformat(),
         "action": "SELL_TO_OPEN", "ratio_qty": 1},
        {"contract_symbol": _occ("SPY", exp_b, 150.0), "underlying": "SPY",
         "option_type": "CALL", "strike": 150.0, "expiration": exp_b.isoformat(),
         "action": "BUY_TO_OPEN", "ratio_qty": 1},
    ]
    order_id = await db_repos.orders.insert(
        Order(account_id="test", symbol="SPY", strategy_id="calendar",
              order_type=OrderType.MULTI_LEG_OPEN, contract_symbol=legs[0]["contract_symbol"],
              strike=150.0, expiration=exp_f, option_type=OptionType.CALL,
              quantity=1, limit_price=-1.50, fill_price=-1.50,  # debit 1.50
              status=OrderStatus.FILLED, placed_at=_utc(), client_order_id="cal-open",
              raw_request={"underlying": "SPY", "legs": legs, "quantity": 1,
                           "limit_price": -1.50, "width_dollars": 0.0})
    )
    rec = Reconciler(PaperBroker(), db_repos, _config())
    full = await db_repos.orders.get(order_id)
    cycle_id = await rec._open_cycle_for_spread(
        full, fill_price=-1.50, summary=ReconcileSummary(),
    )
    cyc = await db_repos.cycles.get(cycle_id)
    # debit 1.50 → (0 − (−1.50)) * 100 * 1 = 150.0
    assert cyc.initial_capital_at_risk == pytest.approx(150.0)


# -- close triggers --------------------------------------------------------


async def _seed_open_calendar(db_repos, today, debit, *, front_dte=14):
    exp_f = today + timedelta(days=front_dte)
    exp_b = today + timedelta(days=60)
    cycle_id = await db_repos.cycles.insert(
        WheelCycle(account_id="test", symbol="SPY", strategy_id="calendar",
                   started_at=_utc(), n_orders=1)
    )
    legs = [
        {"contract_symbol": _occ("SPY", exp_f, 150.0), "underlying": "SPY",
         "option_type": "CALL", "strike": 150.0, "expiration": exp_f.isoformat(),
         "action": "SELL_TO_OPEN", "ratio_qty": 1},
        {"contract_symbol": _occ("SPY", exp_b, 150.0), "underlying": "SPY",
         "option_type": "CALL", "strike": 150.0, "expiration": exp_b.isoformat(),
         "action": "BUY_TO_OPEN", "ratio_qty": 1},
    ]
    await db_repos.orders.insert(
        Order(account_id="test", symbol="SPY", strategy_id="calendar", cycle_id=cycle_id,
              order_type=OrderType.MULTI_LEG_OPEN, contract_symbol=legs[0]["contract_symbol"],
              strike=150.0, expiration=exp_f, option_type=OptionType.CALL,
              quantity=1, fill_price=-debit, status=OrderStatus.FILLED,
              placed_at=_utc(), client_order_id="cal-open",
              raw_request={"underlying": "SPY", "legs": legs, "quantity": 1,
                           "width_dollars": 0.0})
    )
    await db_repos.positions.insert(
        Position(account_id="test", symbol="SPY", strategy_id="calendar",
                 state=PositionState.SPREAD_OPEN, shares=0,
                 current_cycle_id=cycle_id, state_changed_at=_utc())
    )
    return exp_f, exp_b


@pytest.mark.asyncio
async def test_profit_close_at_25_pct(db_repos):
    """Close when close-value ≥ 1.25 × debit. Debit 1.50 → target 1.875.
    Seed front mid 0.50, back mid 2.50 → close value 2.00 ≥ 1.875 → triggers."""
    broker = PaperBroker()
    today = date(2025, 6, 1)
    exp_f, exp_b = await _seed_open_calendar(db_repos, today, debit=1.50)
    broker.seed_quote(Quote(symbol=_occ("SPY", exp_f, 150.0), bid=0.48, ask=0.52))  # front 0.50
    broker.seed_quote(Quote(symbol=_occ("SPY", exp_b, 150.0), bid=2.48, ask=2.52))  # back 2.50
    proposal = await propose_close_for_symbol(
        broker, db_repos, "SPY", _config(), today=date(2025, 6, 10), strategy=_strategy(),
    )
    assert proposal is not None
    assert proposal.order_type == OrderType.MULTI_LEG_CLOSE
    assert "calendar_profit" in proposal.rationale
    # close value = back 2.50 − front 0.50 = 2.00
    assert proposal.net_credit_per_spread == pytest.approx(2.00)


@pytest.mark.asyncio
async def test_time_close_at_front_dte_2(db_repos):
    """Front DTE ≤ 2 → force-close regardless of profit."""
    broker = PaperBroker()
    today = date(2025, 6, 1)
    # Front expires in 2 days from the as-of date.
    exp_f, exp_b = await _seed_open_calendar(db_repos, today, debit=1.50, front_dte=2)
    broker.seed_quote(Quote(symbol=_occ("SPY", exp_f, 150.0), bid=0.10, ask=0.14))
    broker.seed_quote(Quote(symbol=_occ("SPY", exp_b, 150.0), bid=1.40, ask=1.44))  # no profit
    proposal = await propose_close_for_symbol(
        broker, db_repos, "SPY", _config(), today=today, strategy=_strategy(),
    )
    assert proposal is not None
    assert "calendar_time" in proposal.rationale


# -- reconciler front-expiry guard -----------------------------------------


@pytest.mark.asyncio
async def test_reconciler_flags_manual_if_front_expires_while_open(db_repos):
    """A calendar still SPREAD_OPEN with the front short leg gone from the
    broker (the 2-DTE close didn't fire) → flag MANUAL_INTERVENTION."""
    await db_repos.positions.insert(
        Position(account_id="test", symbol="SPY", strategy_id="calendar",
                 state=PositionState.SPREAD_OPEN, shares=0, state_changed_at=_utc())
    )
    rec = Reconciler(PaperBroker(), db_repos, _config())
    summary = ReconcileSummary()
    local = await db_repos.positions.get_by_symbol("test", "SPY", strategy_id="calendar")
    # No broker rows → front short leg gone.
    await rec._diff_one("SPY", local, [], summary)
    assert summary.manual_interventions == 1


@pytest.mark.asyncio
async def test_reconciler_healthy_calendar_no_flag(db_repos):
    """Front short leg still alive at broker → no flag."""
    await db_repos.positions.insert(
        Position(account_id="test", symbol="SPY", strategy_id="calendar",
                 state=PositionState.SPREAD_OPEN, shares=0, state_changed_at=_utc())
    )
    rec = Reconciler(PaperBroker(), db_repos, _config())
    summary = ReconcileSummary()
    local = await db_repos.positions.get_by_symbol("test", "SPY", strategy_id="calendar")
    front_short = Position(account_id="test", symbol="SPY",
                           state=PositionState.CC_OPEN, shares=0, state_changed_at=_utc())
    await rec._diff_one("SPY", local, [front_short], summary)
    assert summary.manual_interventions == 0
