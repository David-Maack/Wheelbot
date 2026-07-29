"""0DTE paper-test strategy — entry window, flatten, caps, dual-ledger math.

Uses PaperBroker + the db_repos fixture like test_iron_condor / test_spreads.
Times are injected via the `now` kwarg (aware America/New_York datetimes) so
tests are independent of the wall clock; the trend reference is injected via
`session_ref` so no live VWAP fetch happens.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

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
from strategies.spreads import (
    DIRECTION_BEAR_CALL,
    DIRECTION_BULL_PUT,
    DIRECTION_IRON_CONDOR,
)
from strategies.zero_dte import (
    entry_penalized_credit,
    exit_penalized_debit,
    in_entry_window,
    past_flatten,
    pick_direction,
    propose_all_closes,
    propose_close_for_symbol,
    propose_for_symbol,
)

ET = ZoneInfo("America/New_York")

# Monday. All chain seeds expire SAME DAY (DTE 0).
TODAY = date(2026, 7, 27)


def _et(hh: int, mm: int) -> datetime:
    return datetime(TODAY.year, TODAY.month, TODAY.day, hh, mm, tzinfo=ET)


def _config() -> dict:
    return {"account": {"id": "test", "broker": "paper"}}


def _universe() -> dict:
    return {
        "tickers": [UniverseEntry(symbol="SPY", name="SPY", tier=1, overrides={})],
        "banned": [],
        "banned_rules": [],
    }


def _strategy(**overrides: Any) -> StrategyDefinition:
    params: dict[str, Any] = {
        "symbol": "SPY",
        "structure": "narrow_vertical",
        "spread_width_dollars": 2.0,
        "short_delta_min": 0.15,
        "short_delta_max": 0.25,
        # The synthetic chain's premium model (|delta| * 3) yields ~9% credit
        # on a $2 width — below the production 15% floor. Tests that exercise
        # the floor itself override this back up.
        "min_credit_pct_of_width": 5.0,
        "profit_close_pct": 50,
        "entry_start_et": "10:00",
        "entry_end_et": "10:15",
        "flatten_et": "15:00",
        "max_positions_per_day": 1,
        "max_capital_per_spread_usd": 200.0,
        "stop_slippage_adder": 0.05,
        "zero_dte_daily_loss_usd": 250.0,
    }
    params.update(overrides)
    return StrategyDefinition(
        id="zero_dte",
        display_name="0DTE SPY (paper test)",
        type="zero_dte",
        enabled=True,
        max_concurrent=1,
        params=params,
    )


def _occ(symbol: str, exp: date, kind: str, strike: float) -> str:
    k = "P" if kind.upper() == "PUT" else "C"
    return f"{symbol}{exp.strftime('%y%m%d')}{k}{int(round(strike * 1000)):08d}"


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


def _seed_same_day_chain(broker: PaperBroker, symbol: str, spot: float) -> None:
    """SPY-like chain expiring TODAY. Same delta/premium model as the iron
    condor tests: put_delta = -0.5 - 0.03*offset, call_delta = 0.5 - 0.03*offset,
    mid = max(0.10, |delta| * 3), bid/ask = mid -/+ 0.02."""
    contracts: list[OptionContract] = []
    for offset in range(-25, 26):
        strike = spot + offset
        put_delta = max(-0.99, min(-0.01, -0.50 - 0.03 * offset))
        call_delta = max(0.01, min(0.99, 0.50 - 0.03 * offset))
        put_mid = max(0.10, abs(put_delta) * 3.0)
        call_mid = max(0.10, abs(call_delta) * 3.0)
        contracts.append(_contract(
            symbol, kind=OptionType.PUT, strike=strike, exp=TODAY,
            delta=put_delta, bid=max(0.05, put_mid - 0.02), ask=put_mid + 0.02,
        ))
        contracts.append(_contract(
            symbol, kind=OptionType.CALL, strike=strike, exp=TODAY,
            delta=call_delta, bid=max(0.05, call_mid - 0.02), ask=call_mid + 0.02,
        ))
    broker.seed_chain(symbol, contracts)


def _entry_broker(spot: float = 500.0) -> PaperBroker:
    from core.models import Quote

    broker = PaperBroker(cash=50_000)
    _seed_same_day_chain(broker, "SPY", spot)
    broker.seed_quote(Quote(symbol="SPY", bid=spot - 0.01, ask=spot + 0.01))
    return broker


async def _propose(db_repos, broker, *, now, strategy=None, session_ref=499.0):
    return await propose_for_symbol(
        broker, db_repos, "SPY", _config(), _universe(),
        today=TODAY, strategy=strategy or _strategy(),
        now=now, session_ref=session_ref,
    )


# -- trend rule --------------------------------------------------------------


def test_pick_direction_above_ref_is_bull_put():
    assert pick_direction(500.0, 499.0) == DIRECTION_BULL_PUT
    # At the reference exactly, stay with the bullish default (>=).
    assert pick_direction(500.0, 500.0) == DIRECTION_BULL_PUT


def test_pick_direction_below_ref_is_bear_call():
    assert pick_direction(498.0, 499.0) == DIRECTION_BEAR_CALL


# -- entry window gating -----------------------------------------------------


def test_entry_window_helpers():
    params = _strategy().params
    assert not in_entry_window(params, _et(9, 55))
    assert in_entry_window(params, _et(10, 0))
    assert in_entry_window(params, _et(10, 15))
    assert not in_entry_window(params, _et(10, 20))
    assert not past_flatten(params, _et(14, 59))
    assert past_flatten(params, _et(15, 0))


@pytest.mark.asyncio
async def test_entry_before_window_skips(db_repos):
    proposal = await _propose(db_repos, _entry_broker(), now=_et(9, 55))
    assert proposal is None


@pytest.mark.asyncio
async def test_entry_inside_window_proposes_bull_put(db_repos):
    """SPY above session_ref -> bull put spread, 2 legs, DTE 0, wing-is-the-stop."""
    proposal = await _propose(db_repos, _entry_broker(500.0), now=_et(10, 5),
                              session_ref=499.0)
    assert proposal is not None
    assert proposal.strategy_id == "zero_dte"
    assert proposal.direction == DIRECTION_BULL_PUT
    assert len(proposal.legs) == 2
    for leg in proposal.legs:
        assert leg.expiration == TODAY          # same-day expiry
        assert leg.option_type == OptionType.PUT
    short = next(x for x in proposal.legs if x.action == OrderType.SELL_TO_OPEN)
    long = next(x for x in proposal.legs if x.action == OrderType.BUY_TO_OPEN)
    assert short.strike - long.strike == pytest.approx(2.0)
    assert proposal.net_credit_per_spread > 0


@pytest.mark.asyncio
async def test_entry_after_window_skips(db_repos):
    proposal = await _propose(db_repos, _entry_broker(), now=_et(10, 20))
    assert proposal is None


@pytest.mark.asyncio
async def test_entry_weekend_skips(db_repos):
    saturday = datetime(2026, 7, 25, 10, 5, tzinfo=ET)
    proposal = await propose_for_symbol(
        _entry_broker(), db_repos, "SPY", _config(), _universe(),
        today=date(2026, 7, 25), strategy=_strategy(),
        now=saturday, session_ref=499.0,
    )
    assert proposal is None


# -- trend rule end-to-end ---------------------------------------------------


@pytest.mark.asyncio
async def test_below_vwap_proposes_bear_call(db_repos):
    proposal = await _propose(db_repos, _entry_broker(500.0), now=_et(10, 5),
                              session_ref=501.0)  # spot below ref -> bearish
    assert proposal is not None
    assert proposal.direction == DIRECTION_BEAR_CALL
    for leg in proposal.legs:
        assert leg.option_type == OptionType.CALL
    short = next(x for x in proposal.legs if x.action == OrderType.SELL_TO_OPEN)
    long = next(x for x in proposal.legs if x.action == OrderType.BUY_TO_OPEN)
    assert long.strike - short.strike == pytest.approx(2.0)


# -- credit floor ------------------------------------------------------------


@pytest.mark.asyncio
async def test_min_credit_pct_floor_skips(db_repos):
    """The synthetic chain pays ~9% of a $2 width — production's 15% floor
    rejects it."""
    strat = _strategy(min_credit_pct_of_width=15.0)
    proposal = await _propose(db_repos, _entry_broker(), now=_et(10, 5), strategy=strat)
    assert proposal is None


# -- max positions per day ---------------------------------------------------


@pytest.mark.asyncio
async def test_max_positions_per_day_blocks_second_entry(db_repos):
    await db_repos.cycles.insert(WheelCycle(
        account_id="test", symbol="SPY", strategy_id="zero_dte",
        started_at=datetime(TODAY.year, TODAY.month, TODAY.day, 14, 5),
    ))
    proposal = await _propose(db_repos, _entry_broker(), now=_et(10, 5))
    assert proposal is None


@pytest.mark.asyncio
async def test_yesterdays_cycle_does_not_block_today(db_repos):
    await db_repos.cycles.insert(WheelCycle(
        account_id="test", symbol="SPY", strategy_id="zero_dte",
        started_at=datetime(2026, 7, 24, 14, 5),
    ))
    proposal = await _propose(db_repos, _entry_broker(), now=_et(10, 5))
    assert proposal is not None


# -- daily loss cap ----------------------------------------------------------


@pytest.mark.asyncio
async def test_daily_loss_cap_skips_entries(db_repos):
    """Today's PENALIZED realized pnl <= -zero_dte_daily_loss_usd -> no entry."""
    await db_repos.zero_dte_ledger.insert_entry({
        "symbol": "SPY", "structure": "narrow_vertical", "quantity": 1,
        "entry_ts": f"{TODAY.isoformat()}T14:05:00",
        "entry_credit_raw": 0.30, "entry_spread_width_quotes": 0.08,
        "entry_credit_penalized": 0.26,
        "exit_ts": f"{TODAY.isoformat()}T15:00:00",
        "exit_debit_raw": 2.0, "exit_debit_penalized": 3.26,
        "pnl_raw": -170.0, "pnl_penalized": -300.0,
        "exit_reason": "flatten",
    })
    proposal = await _propose(db_repos, _entry_broker(), now=_et(10, 5))
    assert proposal is None


@pytest.mark.asyncio
async def test_loss_under_cap_still_enters(db_repos):
    await db_repos.zero_dte_ledger.insert_entry({
        "symbol": "SPY", "structure": "narrow_vertical", "quantity": 1,
        "entry_ts": f"{TODAY.isoformat()}T14:05:00",
        "exit_ts": f"{TODAY.isoformat()}T15:00:00",
        "pnl_raw": -80.0, "pnl_penalized": -100.0,
        "exit_reason": "flatten",
        "entry_credit_raw": 0.3, "entry_spread_width_quotes": 0.08,
        "entry_credit_penalized": 0.26,
        "exit_debit_raw": 1.1, "exit_debit_penalized": 1.26,
    })
    proposal = await _propose(db_repos, _entry_broker(), now=_et(10, 5))
    assert proposal is not None


# -- penalized ledger math ---------------------------------------------------


def test_entry_credit_penalization():
    # raw credit 0.40, summed leg spreads 0.08 -> concede half the spread.
    assert entry_penalized_credit(0.40, 0.08) == pytest.approx(0.36)


def test_exit_debit_penalization_normal_close():
    assert exit_penalized_debit(
        0.20, 0.08, flatten=False, slippage_adder=0.05,
    ) == pytest.approx(0.24)


def test_exit_debit_penalization_flatten_adds_adder():
    assert exit_penalized_debit(
        1.00, 0.08, flatten=True, slippage_adder=0.05,
    ) == pytest.approx(1.09)


@pytest.mark.asyncio
async def test_entry_writes_penalized_ledger_row(db_repos):
    proposal = await _propose(db_repos, _entry_broker(), now=_et(10, 5))
    assert proposal is not None
    row = await db_repos.zero_dte_ledger.latest_for_symbol("SPY")
    assert row is not None
    assert row["structure"] == "narrow_vertical"
    assert row["quantity"] == proposal.quantity
    assert row["entry_credit_raw"] == pytest.approx(proposal.net_credit_per_spread)
    # Two legs at 0.04-wide quotes each -> summed spread 0.08, penalty 0.04.
    assert row["entry_spread_width_quotes"] == pytest.approx(0.08)
    assert row["entry_credit_penalized"] == pytest.approx(
        proposal.net_credit_per_spread - 0.04
    )
    assert row["exit_ts"] is None


# -- close pass: flatten + profit take ---------------------------------------


SHORT_OCC = _occ("SPY", TODAY, "PUT", 490.0)
LONG_OCC = _occ("SPY", TODAY, "PUT", 488.0)


async def _open_spread_position(db_repos, *, fill_price: float = 0.40) -> int:
    """SPREAD_OPEN zero_dte position + FILLED MULTI_LEG_OPEN order, matching
    the reconciler's persisted shape (raw_request carries the legs)."""
    cycle_id = await db_repos.cycles.insert(WheelCycle(
        account_id="test", symbol="SPY", strategy_id="zero_dte",
        started_at=datetime(TODAY.year, TODAY.month, TODAY.day, 14, 5),
    ))
    await db_repos.positions.insert(Position(
        account_id="test", symbol="SPY", strategy_id="zero_dte",
        state=PositionState.SPREAD_OPEN, shares=0, current_cycle_id=cycle_id,
        state_changed_at=datetime(TODAY.year, TODAY.month, TODAY.day, 14, 6),
    ))
    legs = [
        {
            "contract_symbol": SHORT_OCC, "underlying": "SPY",
            "option_type": "PUT", "strike": 490.0,
            "expiration": TODAY.isoformat(),
            "action": "SELL_TO_OPEN", "ratio_qty": 1,
        },
        {
            "contract_symbol": LONG_OCC, "underlying": "SPY",
            "option_type": "PUT", "strike": 488.0,
            "expiration": TODAY.isoformat(),
            "action": "BUY_TO_OPEN", "ratio_qty": 1,
        },
    ]
    await db_repos.orders.insert(Order(
        account_id="test", symbol="SPY", strategy_id="zero_dte",
        cycle_id=cycle_id, order_type=OrderType.MULTI_LEG_OPEN,
        quantity=1, fill_price=fill_price, status=OrderStatus.FILLED,
        placed_at=datetime(TODAY.year, TODAY.month, TODAY.day, 14, 5),
        filled_at=datetime(TODAY.year, TODAY.month, TODAY.day, 14, 6),
        raw_request={"legs": legs},
        client_order_id="zdte-test-open-1",
        broker_order_id="paper-zdte-1",
    ))
    return cycle_id


def _close_broker(*, short_mid: float, long_mid: float) -> PaperBroker:
    from core.models import Quote

    broker = PaperBroker(cash=50_000)
    broker.seed_quote(Quote(symbol=SHORT_OCC, bid=short_mid - 0.02, ask=short_mid + 0.02))
    broker.seed_quote(Quote(symbol=LONG_OCC, bid=long_mid - 0.02, ask=long_mid + 0.02))
    return broker


@pytest.mark.asyncio
async def test_flatten_fires_at_1500_regardless_of_pnl(db_repos):
    """Deep-losing spread (debit 1.00 vs 0.40 credit) at 15:00 ET -> close
    proposed anyway. The wing was the stop all day; 15:00 is the hard out."""
    await _open_spread_position(db_repos)
    broker = _close_broker(short_mid=1.50, long_mid=0.50)  # debit-to-close 1.00
    proposal = await propose_close_for_symbol(
        broker, db_repos, "SPY", _config(), strategy=_strategy(), now=_et(15, 0),
    )
    assert proposal is not None
    assert proposal.order_type == OrderType.MULTI_LEG_CLOSE
    assert "HARD FLATTEN" in proposal.rationale
    actions = sorted(str(x.action) for x in proposal.legs)
    assert actions == ["BUY_TO_CLOSE", "SELL_TO_CLOSE"]


@pytest.mark.asyncio
async def test_no_close_before_flatten_without_profit(db_repos):
    """Same losing spread at 14:30 -> no proposal: no stop orders by design
    (the wing is the stop), and the profit target isn't hit."""
    await _open_spread_position(db_repos)
    broker = _close_broker(short_mid=1.50, long_mid=0.50)
    proposal = await propose_close_for_symbol(
        broker, db_repos, "SPY", _config(), strategy=_strategy(), now=_et(14, 30),
    )
    assert proposal is None


@pytest.mark.asyncio
async def test_profit_take_before_flatten(db_repos):
    """Debit 0.08 <= 50% target (0.20 on a 0.40 credit) -> profit close."""
    await _open_spread_position(db_repos)
    broker = _close_broker(short_mid=0.10, long_mid=0.02)
    proposal = await propose_close_for_symbol(
        broker, db_repos, "SPY", _config(), strategy=_strategy(), now=_et(12, 0),
    )
    assert proposal is not None
    assert "profit_close" in proposal.rationale


@pytest.mark.asyncio
async def test_flatten_exit_updates_ledger_with_adder(db_repos):
    """Flatten exit: penalized debit = raw + half summed spreads + adder;
    pnl columns derive from the stored entry row."""
    await _open_spread_position(db_repos)
    await db_repos.zero_dte_ledger.insert_entry({
        "symbol": "SPY", "structure": "narrow_vertical", "quantity": 1,
        "entry_ts": f"{TODAY.isoformat()}T14:05:00",
        "entry_credit_raw": 0.40, "entry_spread_width_quotes": 0.08,
        "entry_credit_penalized": 0.36,
    })
    broker = _close_broker(short_mid=1.50, long_mid=0.50)
    proposal = await propose_close_for_symbol(
        broker, db_repos, "SPY", _config(), strategy=_strategy(), now=_et(15, 0),
    )
    assert proposal is not None
    row = await db_repos.zero_dte_ledger.latest_for_symbol("SPY")
    assert row["exit_reason"] == "flatten"
    assert row["exit_debit_raw"] == pytest.approx(1.00)
    # 1.00 + (0.04 + 0.04)/2 + 0.05 adder = 1.09
    assert row["exit_debit_penalized"] == pytest.approx(1.09)
    assert row["pnl_raw"] == pytest.approx((0.40 - 1.00) * 100)
    assert row["pnl_penalized"] == pytest.approx((0.36 - 1.09) * 100)


@pytest.mark.asyncio
async def test_profit_exit_ledger_has_no_adder(db_repos):
    await _open_spread_position(db_repos)
    await db_repos.zero_dte_ledger.insert_entry({
        "symbol": "SPY", "structure": "narrow_vertical", "quantity": 1,
        "entry_ts": f"{TODAY.isoformat()}T14:05:00",
        "entry_credit_raw": 0.40, "entry_spread_width_quotes": 0.08,
        "entry_credit_penalized": 0.36,
    })
    broker = _close_broker(short_mid=0.10, long_mid=0.02)
    proposal = await propose_close_for_symbol(
        broker, db_repos, "SPY", _config(), strategy=_strategy(), now=_et(12, 0),
    )
    assert proposal is not None
    row = await db_repos.zero_dte_ledger.latest_for_symbol("SPY")
    assert row["exit_reason"] == "profit_target"
    # debit 0.08 + half of summed spreads (0.08/2) = 0.12, NO adder.
    assert row["exit_debit_penalized"] == pytest.approx(0.12)


@pytest.mark.asyncio
async def test_propose_all_closes_walks_open_positions(db_repos):
    await _open_spread_position(db_repos)
    broker = _close_broker(short_mid=1.50, long_mid=0.50)
    proposals = await propose_all_closes(
        broker, db_repos, _config(), strategy=_strategy(), now=_et(15, 5),
    )
    assert len(proposals) == 1


# -- iron_condor structure ---------------------------------------------------


@pytest.mark.asyncio
async def test_condor_structure_builds_four_legs(db_repos):
    strat = _strategy(
        structure="iron_condor",
        max_capital_per_spread_usd=400.0,  # $3 wings -> max_loss > $200
    )
    proposal = await _propose(db_repos, _entry_broker(500.0), now=_et(10, 5),
                              strategy=strat)
    assert proposal is not None
    assert proposal.direction == DIRECTION_IRON_CONDOR
    assert len(proposal.legs) == 4
    for leg in proposal.legs:
        assert leg.expiration == TODAY
    row = await db_repos.zero_dte_ledger.latest_for_symbol("SPY")
    assert row is not None and row["structure"] == "iron_condor"
    # Four legs at 0.04-wide quotes -> summed spread 0.16, penalty 0.08.
    assert row["entry_spread_width_quotes"] == pytest.approx(0.16)
    assert row["entry_credit_penalized"] == pytest.approx(
        row["entry_credit_raw"] - 0.08
    )


# -- sizing ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skips_when_max_loss_exceeds_capital_cap(db_repos):
    """$2-wide spread max_loss ~$180 vs a $100 cap -> qty 0 -> skip (never
    floor to 1 — audit #13 rule)."""
    strat = _strategy(max_capital_per_spread_usd=100.0)
    proposal = await _propose(db_repos, _entry_broker(), now=_et(10, 5),
                              strategy=strat)
    assert proposal is None
