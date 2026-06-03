"""TICKET-006 — mid-cycle earnings recheck.

Covers the partial-split kill-switch behaviour (flag_manual unconditional,
close skips on kill switch), provider-unavailable distinguishability,
rate-limit, spread close-action fallback, and the dashboard badge render.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from core.models import (
    Order,
    OrderStatus,
    OrderType,
    OptionType,
    Position,
    PositionState,
    Quote,
    WheelCycle,
)
from data.earnings import EarningsLookup
from platforms.paper_broker import PaperBroker
from risk.earnings_recheck import (
    ACTION_CLOSE_PROPOSED,
    ACTION_CLOSE_SKIPPED_KILL_SWITCH,
    ACTION_CLOSE_SPREAD_UNSUPPORTED,
    ACTION_FLAG_MANUAL,
    ACTION_PROVIDER_UNAVAILABLE,
    EarningsRecheckResult,
    check_open_positions_for_new_earnings,
    is_in_earnings_window,
)


# -- helpers ----------------------------------------------------------------


def _today() -> date:
    return date(2026, 6, 1)


def _config(**overrides: Any) -> dict[str, Any]:
    base = {
        "account": {"id": "test"},
        "risk": {
            "earnings_recheck": {
                "enabled": True,
                "check_interval_ticks": 1,   # tests want each call to do work
                "action": "flag_manual",
                "days_before": 5,
                "days_after": 2,
            },
        },
    }
    base["risk"]["earnings_recheck"].update(overrides)
    return base


def _stub_lookup(symbol_to_date: dict[str, date | None]):
    def _f(symbol: str) -> EarningsLookup:
        d = symbol_to_date.get(symbol)
        return EarningsLookup(symbol=symbol, next_date=d, source="finnhub" if d else "none")
    return _f


async def _seed_csp(db_repos, *, symbol: str, contract: str, expiration: date) -> Position:
    now = datetime.now(UTC).replace(tzinfo=None)
    cycle_id = await db_repos.cycles.insert(
        WheelCycle(
            account_id="test", symbol=symbol, strategy_id="monthly_wheel",
            started_at=now,
        )
    )
    await db_repos.orders.insert(
        Order(
            account_id="test", symbol=symbol, strategy_id="monthly_wheel",
            cycle_id=cycle_id,
            order_type=OrderType.SELL_TO_OPEN, contract_symbol=contract,
            strike=10.0, expiration=expiration,
            option_type=OptionType.PUT, quantity=1, fill_price=0.50,
            status=OrderStatus.FILLED, placed_at=now, filled_at=now,
        )
    )
    pos = Position(
        account_id="test", symbol=symbol, strategy_id="monthly_wheel",
        state=PositionState.CSP_OPEN, shares=0,
        current_cycle_id=cycle_id, state_changed_at=now,
    )
    pos_id = await db_repos.positions.insert(pos)
    return pos.model_copy(update={"id": pos_id})


async def _seed_spread(db_repos, *, symbol: str, short_expiration: date) -> Position:
    """SPREAD_OPEN position whose cycle's MULTI_LEG_OPEN raw_request carries
    legs with the expected expiration on the short leg."""
    now = datetime.now(UTC).replace(tzinfo=None)
    cycle_id = await db_repos.cycles.insert(
        WheelCycle(
            account_id="test", symbol=symbol, strategy_id="put_spread",
            started_at=now,
        )
    )
    legs = [
        {"contract_symbol": "X1", "underlying": symbol, "option_type": "PUT",
         "strike": 100.0, "expiration": short_expiration.isoformat(),
         "action": "SELL_TO_OPEN", "ratio_qty": 1},
        {"contract_symbol": "X2", "underlying": symbol, "option_type": "PUT",
         "strike": 95.0,  "expiration": short_expiration.isoformat(),
         "action": "BUY_TO_OPEN",  "ratio_qty": 1},
    ]
    await db_repos.orders.insert(
        Order(
            account_id="test", symbol=symbol, strategy_id="put_spread",
            cycle_id=cycle_id,
            order_type=OrderType.MULTI_LEG_OPEN, contract_symbol=None,
            quantity=1, fill_price=0.50,
            status=OrderStatus.FILLED, placed_at=now, filled_at=now,
            raw_request={"legs": legs},
        )
    )
    pos = Position(
        account_id="test", symbol=symbol, strategy_id="put_spread",
        state=PositionState.SPREAD_OPEN, shares=0,
        current_cycle_id=cycle_id, state_changed_at=now,
    )
    pos_id = await db_repos.positions.insert(pos)
    return pos.model_copy(update={"id": pos_id})


# -- predicate ---------------------------------------------------------------


def test_is_in_earnings_window_basic():
    """The pure predicate — same math the dashboard and the recheck use.
    Window around EXPIRATION: [expiration − days_before, expiration + days_after]."""
    short_exp = date(2026, 7, 5)
    # Same day → inside.
    assert is_in_earnings_window(date(2026, 7, 5), short_exp, days_before=5, days_after=2) is True
    # 3 days BEFORE expiry → inside (within days_before=5).
    assert is_in_earnings_window(date(2026, 7, 2), short_exp, days_before=5, days_after=2) is True
    # Exactly 5 days BEFORE expiry → boundary, inside.
    assert is_in_earnings_window(date(2026, 6, 30), short_exp, days_before=5, days_after=2) is True
    # 6 days BEFORE expiry → outside (one past days_before).
    assert is_in_earnings_window(date(2026, 6, 29), short_exp, days_before=5, days_after=2) is False
    # 2 days AFTER expiry → inside (boundary of days_after).
    assert is_in_earnings_window(date(2026, 7, 7), short_exp, days_before=5, days_after=2) is True
    # 3 days AFTER expiry → outside.
    assert is_in_earnings_window(date(2026, 7, 8), short_exp, days_before=5, days_after=2) is False
    # 10 days BEFORE expiry → outside.
    assert is_in_earnings_window(date(2026, 6, 25), short_exp, days_before=5, days_after=2) is False


# -- per-ticket required cases ----------------------------------------------


@pytest.mark.asyncio
async def test_earnings_outside_dte_no_action(db_repos):
    """Earnings 60 days out vs a 35-DTE position → no result."""
    pos = await _seed_csp(
        db_repos, symbol="F", contract="F260706P00010000",
        expiration=_today() + timedelta(days=35),
    )
    results = await check_open_positions_for_new_earnings(
        repos=db_repos, router=None, config=_config(),
        today=_today(), recheck_state={"ticks_since_check": 99},
        next_earnings_fn=_stub_lookup({"F": _today() + timedelta(days=60)}),
    )
    assert results == []
    reloaded = await db_repos.positions.get(pos.id)
    assert reloaded.state == PositionState.CSP_OPEN


@pytest.mark.asyncio
async def test_earnings_inside_dte_flags_manual(db_repos):
    """Earnings 10 days out on a 14-DTE CSP → flag_manual. Position flips to
    MANUAL_INTERVENTION + state_log row written."""
    pos = await _seed_csp(
        db_repos, symbol="F", contract="F260706P00010000",
        expiration=_today() + timedelta(days=14),
    )
    results = await check_open_positions_for_new_earnings(
        repos=db_repos, router=None, config=_config(action="flag_manual"),
        today=_today(), recheck_state={"ticks_since_check": 99},
        next_earnings_fn=_stub_lookup({"F": _today() + timedelta(days=10)}),
    )
    assert len(results) == 1 and results[0].action_taken == ACTION_FLAG_MANUAL
    reloaded = await db_repos.positions.get(pos.id)
    assert reloaded.state == PositionState.MANUAL_INTERVENTION
    log_rows = await db_repos.state_log.list_for_position(pos.id)
    assert any(r.to_state == PositionState.MANUAL_INTERVENTION for r in log_rows)


@pytest.mark.asyncio
async def test_earnings_inside_dte_close_action_builds_btc(db_repos):
    """action=close + CSP → router.place is called with a BUY_TO_CLOSE
    Proposal whose trigger_reason is earnings_recheck_close."""
    pos = await _seed_csp(
        db_repos, symbol="F", contract="F260706P00010000",
        expiration=_today() + timedelta(days=14),
    )

    class _CapturingRouter:
        def __init__(self):
            self.placed: list[Any] = []
        async def place(self, proposal, **kw):
            self.placed.append(proposal)
            from types import SimpleNamespace
            return SimpleNamespace(placed=SimpleNamespace(broker_order_id="paper-x"))

    router = _CapturingRouter()
    results = await check_open_positions_for_new_earnings(
        repos=db_repos, router=router, config=_config(action="close"),
        today=_today(), recheck_state={"ticks_since_check": 99},
        next_earnings_fn=_stub_lookup({"F": _today() + timedelta(days=10)}),
    )
    assert len(results) == 1 and results[0].action_taken == ACTION_CLOSE_PROPOSED
    assert len(router.placed) == 1
    p = router.placed[0]
    assert p.order_type == OrderType.BUY_TO_CLOSE
    assert p.trigger_reason == "earnings_recheck_close"
    assert p.strategy_id == "monthly_wheel"


# -- partial-split kill-switch behaviour (Q1 from review) -------------------


@pytest.mark.asyncio
async def test_flag_manual_runs_when_kill_switch_tripped(db_repos):
    """LOCKED-IN behaviour: action=flag_manual must run regardless of kill
    switch — it's a state annotation the operator needs to see in order to
    act. Future refactors that block all defensive paths under kill switch
    would silently break this."""
    pos = await _seed_csp(
        db_repos, symbol="F", contract="F260706P00010000",
        expiration=_today() + timedelta(days=14),
    )
    results = await check_open_positions_for_new_earnings(
        repos=db_repos, router=None, config=_config(action="flag_manual"),
        today=_today(), kill_switch_tripped=True,
        recheck_state={"ticks_since_check": 99},
        next_earnings_fn=_stub_lookup({"F": _today() + timedelta(days=10)}),
    )
    assert len(results) == 1 and results[0].action_taken == ACTION_FLAG_MANUAL
    reloaded = await db_repos.positions.get(pos.id)
    assert reloaded.state == PositionState.MANUAL_INTERVENTION


@pytest.mark.asyncio
async def test_close_action_skips_when_kill_switch_tripped(db_repos):
    """The OTHER side of the partial split: action=close + kill_switch_tripped
    must NOT mutate state or route an order. Result records the skip so
    future-you can audit whether a defensive bypass would have helped."""
    pos = await _seed_csp(
        db_repos, symbol="F", contract="F260706P00010000",
        expiration=_today() + timedelta(days=14),
    )

    class _RecordingRouter:
        def __init__(self):
            self.placed: list[Any] = []
        async def place(self, proposal, **kw):  # should never be called
            self.placed.append(proposal)

    router = _RecordingRouter()
    results = await check_open_positions_for_new_earnings(
        repos=db_repos, router=router, config=_config(action="close"),
        today=_today(), kill_switch_tripped=True,
        recheck_state={"ticks_since_check": 99},
        next_earnings_fn=_stub_lookup({"F": _today() + timedelta(days=10)}),
    )
    assert len(results) == 1
    assert results[0].action_taken == ACTION_CLOSE_SKIPPED_KILL_SWITCH
    assert router.placed == []
    reloaded = await db_repos.positions.get(pos.id)
    assert reloaded.state == PositionState.CSP_OPEN  # untouched


# -- spread fallback --------------------------------------------------------


@pytest.mark.asyncio
async def test_spread_close_action_falls_back_to_flag_manual(db_repos):
    """SPREAD_OPEN + action=close: no generic spread-close helper exists
    yet (lands with TICKET-014). Fall back to flag_manual and record the
    distinct ACTION_CLOSE_SPREAD_UNSUPPORTED result."""
    pos = await _seed_spread(
        db_repos, symbol="MSFT", short_expiration=_today() + timedelta(days=10),
    )
    results = await check_open_positions_for_new_earnings(
        repos=db_repos, router=None, config=_config(action="close"),
        today=_today(), recheck_state={"ticks_since_check": 99},
        next_earnings_fn=_stub_lookup({"MSFT": _today() + timedelta(days=6)}),
    )
    assert len(results) == 1
    assert results[0].action_taken == ACTION_CLOSE_SPREAD_UNSUPPORTED
    reloaded = await db_repos.positions.get(pos.id)
    assert reloaded.state == PositionState.MANUAL_INTERVENTION


# -- provider unavailability (Issue #1 from review) -------------------------


@pytest.mark.asyncio
async def test_earnings_provider_unavailable_no_action(db_repos):
    """Provider returns None (Finnhub rate-limited and yfinance returned
    nothing). Logged distinctly via ACTION_PROVIDER_UNAVAILABLE so 'we
    didn't check' is observable, not silently equal to 'no event'."""
    pos = await _seed_csp(
        db_repos, symbol="F", contract="F260706P00010000",
        expiration=_today() + timedelta(days=14),
    )
    results = await check_open_positions_for_new_earnings(
        repos=db_repos, router=None, config=_config(),
        today=_today(), recheck_state={"ticks_since_check": 99},
        next_earnings_fn=_stub_lookup({"F": None}),
    )
    assert len(results) == 1
    assert results[0].action_taken == ACTION_PROVIDER_UNAVAILABLE
    assert results[0].earnings_date is None
    reloaded = await db_repos.positions.get(pos.id)
    assert reloaded.state == PositionState.CSP_OPEN  # untouched


# -- rate limit + disabled --------------------------------------------------


@pytest.mark.asyncio
async def test_check_interval_ticks_rate_limit(db_repos):
    """The per-position work only runs when ticks_since_check >= interval.
    Intermediate calls return [] and just bump the counter."""
    await _seed_csp(
        db_repos, symbol="F", contract="F260706P00010000",
        expiration=_today() + timedelta(days=14),
    )
    state = {"ticks_since_check": 0}
    cfg = _config(check_interval_ticks=4, action="flag_manual")
    earnings_fn = _stub_lookup({"F": _today() + timedelta(days=10)})

    # Calls 1, 2, 3 — under interval, no work.
    for _ in range(3):
        r = await check_open_positions_for_new_earnings(
            repos=db_repos, router=None, config=cfg,
            today=_today(), recheck_state=state,
            next_earnings_fn=earnings_fn,
        )
        assert r == []
    # Call 4 — hits the interval, work runs.
    r = await check_open_positions_for_new_earnings(
        repos=db_repos, router=None, config=cfg,
        today=_today(), recheck_state=state,
        next_earnings_fn=earnings_fn,
    )
    assert len(r) == 1 and r[0].action_taken == ACTION_FLAG_MANUAL
    # Counter reset → next call no work again.
    assert state["ticks_since_check"] == 0
    r = await check_open_positions_for_new_earnings(
        repos=db_repos, router=None, config=cfg,
        today=_today(), recheck_state=state,
        next_earnings_fn=earnings_fn,
    )
    assert r == []  # still under interval after reset


@pytest.mark.asyncio
async def test_disabled_in_config_no_action(db_repos):
    """enabled=false short-circuits before any DB or provider work."""
    await _seed_csp(
        db_repos, symbol="F", contract="F260706P00010000",
        expiration=_today() + timedelta(days=14),
    )
    cfg = _config(enabled=False)
    results = await check_open_positions_for_new_earnings(
        repos=db_repos, router=None, config=cfg,
        today=_today(), recheck_state={"ticks_since_check": 99},
        next_earnings_fn=_stub_lookup({"F": _today() + timedelta(days=10)}),
    )
    assert results == []


@pytest.mark.asyncio
async def test_flag_manual_is_idempotent(db_repos):
    """If a position is already MANUAL_INTERVENTION, a second flag is a
    no-op (no Discord re-spam, no duplicate state_log row)."""
    pos = await _seed_csp(
        db_repos, symbol="F", contract="F260706P00010000",
        expiration=_today() + timedelta(days=14),
    )
    cfg = _config(action="flag_manual")
    earnings_fn = _stub_lookup({"F": _today() + timedelta(days=10)})
    state = {"ticks_since_check": 99}
    # First call — flips state.
    await check_open_positions_for_new_earnings(
        repos=db_repos, router=None, config=cfg, today=_today(),
        recheck_state=state, next_earnings_fn=earnings_fn,
    )
    # Bump counter past interval again.
    state["ticks_since_check"] = 99
    log_rows_before = await db_repos.state_log.list_for_position(pos.id)
    # Second call — position already MANUAL_INTERVENTION → no new state_log row.
    await check_open_positions_for_new_earnings(
        repos=db_repos, router=None, config=cfg, today=_today(),
        recheck_state=state, next_earnings_fn=earnings_fn,
    )
    log_rows_after = await db_repos.state_log.list_for_position(pos.id)
    assert len(log_rows_after) == len(log_rows_before)
