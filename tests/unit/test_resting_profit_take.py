"""Resting profit-takes (2026-10-27): router id family and cancel-before-urgent,
the close pass's resting mode, the orphan sweep, the quote tape, and the
per-strategy PULLBACK opt-out."""

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
    Quote,
    UniverseEntry,
)
from execution.quote_log import log_order_quotes
from execution.resting import RESTING_CLIENT_PREFIX, profit_target, resting_buy_limit
from execution.router import OrderRouter
from platforms.paper_broker import PaperBroker
from strategies.wheel import Proposal
from strategies.wheel_close import propose_all_closes
from tests.unit.test_wheel_close import _seed_open_csp, _strategy

TODAY = date(2025, 6, 1)
OCC = "F250706P00010000"


def _config(**extra: Any) -> dict:
    cfg = {
        "account": {"id": "test", "broker": "paper"},
        "wheel": {"bid_ask_spread_max_pct": 10.0},
        "regime": {"enabled": False},
        "execution": {
            "retry_max_attempts": 1,
            "retry_initial_backoff_seconds": 0,
            "retry_max_backoff_seconds": 0,
        },
    }
    cfg.update(extra)
    return cfg


def _universe() -> dict:
    return {"tickers": [UniverseEntry(symbol="F", name="Ford", tier=1, overrides={})], "banned": []}


def _contract(bid: float = 0.30, ask: float = 0.34) -> OptionContract:
    return OptionContract(
        underlying="F", occ_symbol=OCC, strike=10.0, expiration=TODAY + timedelta(days=35),
        option_type=OptionType.PUT, bid=bid, ask=ask,
    )


def _btc(*, resting: bool, limit: float | None = None, trigger: str = "profit") -> Proposal:
    return Proposal(
        symbol="F", contract=_contract(), order_type=OrderType.BUY_TO_CLOSE, quantity=1,
        rationale="test", strategy_id="monthly_wheel", trigger_reason=trigger,
        resting=resting, limit_price=limit,
    )


async def _noop_sleep(_s: float) -> None:
    return None


@pytest.fixture(autouse=True)
def _stub_earnings(monkeypatch):
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)


# -- limit arithmetic ---------------------------------------------------------


def test_resting_buy_limit_rounds_down_to_a_tick():
    assert resting_buy_limit(0.205) == 0.20
    assert resting_buy_limit(0.119) == 0.11
    assert resting_buy_limit(0.20) == 0.20
    assert resting_buy_limit(3.07) == 3.05
    assert resting_buy_limit(0.004) == 0.01


def test_profit_target_is_premium_less_the_take():
    assert profit_target(0.40, 50) == pytest.approx(0.20)


# -- router ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resting_order_rests_at_target_under_its_own_id(db_repos):
    router = OrderRouter(PaperBroker(cash=20_000), db_repos, _config(), _universe())
    res = await router.place(_btc(resting=True, limit=0.205), sleep=_noop_sleep, today=TODAY)
    assert res.placed is not None
    assert res.placed.client_order_id.startswith(RESTING_CLIENT_PREFIX)
    assert res.placed.limit_price == 0.20          # target, not the 0.34 ask
    assert res.placed.raw_request["wb_quote"] == {
        "bid": 0.30, "ask": 0.34, "mid": 0.32, "resting": True,
    }
    assert await db_repos.positions.get_by_symbol("test", "F", strategy_id="monthly_wheel") is None


@pytest.mark.asyncio
async def test_live_resting_order_is_never_stale_replaced(db_repos):
    router = OrderRouter(PaperBroker(cash=20_000), db_repos, _config(), _universe())
    first = await router.place(_btc(resting=True, limit=0.20), sleep=_noop_sleep, today=TODAY)
    row = await db_repos.orders.get_by_client_id(first.placed.client_order_id)
    old = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=3)
    await db_repos.orders.update(row.id, placed_at=old.isoformat())
    again = await router.place(_btc(resting=True, limit=0.20), sleep=_noop_sleep, today=TODAY)
    assert again.placed is None and again.skipped_duplicate_pending
    assert len(await db_repos.orders.list_recent("test")) == 1


@pytest.mark.asyncio
async def test_urgent_close_cancels_the_resting_order_first(db_repos):
    broker = PaperBroker(cash=20_000)
    router = OrderRouter(broker, db_repos, _config(), _universe())
    rest = await router.place(_btc(resting=True, limit=0.20), sleep=_noop_sleep, today=TODAY)
    stop = await router.place(
        _btc(resting=False, trigger="stop_loss"), sleep=_noop_sleep, today=TODAY,
    )
    assert stop.placed is not None
    assert not stop.placed.client_order_id.startswith(RESTING_CLIENT_PREFIX)
    assert stop.placed.limit_price == 0.34                  # urgent: pays the ask
    local_rest = await db_repos.orders.get_by_client_id(rest.placed.client_order_id)
    assert local_rest.status == OrderStatus.CANCELLED
    broker_view = {
        o.broker_order_id: o for o in await broker.get_orders_since(datetime(2000, 1, 1))
    }
    assert broker_view[rest.placed.broker_order_id].status == OrderStatus.CANCELLED


@pytest.mark.asyncio
async def test_urgent_close_waits_when_the_resting_order_already_filled(db_repos):
    broker = PaperBroker(cash=20_000)
    router = OrderRouter(broker, db_repos, _config(), _universe())
    rest = await router.place(_btc(resting=True, limit=0.20), sleep=_noop_sleep, today=TODAY)
    await broker.fill_order(rest.placed.broker_order_id)   # filled; reconciler hasn't seen it
    stop = await router.place(
        _btc(resting=False, trigger="stop_loss"), sleep=_noop_sleep, today=TODAY,
    )
    assert stop.placed is None and stop.skipped_duplicate_pending
    assert len(await db_repos.orders.list_recent("test")) == 1


@pytest.mark.asyncio
async def test_wide_spread_never_blocks_an_urgent_close(db_repos):
    router = OrderRouter(PaperBroker(cash=20_000), db_repos, _config(), _universe())
    wide = Proposal(
        symbol="F", contract=_contract(bid=0.10, ask=0.50), order_type=OrderType.BUY_TO_CLOSE,
        quantity=1, rationale="stop", strategy_id="monthly_wheel", trigger_reason="stop_loss",
    )
    res = await router.place(wide, sleep=_noop_sleep, today=TODAY)  # 133% spread > 10% cap
    assert res.placed is not None and res.placed.limit_price == 0.50


@pytest.mark.asyncio
async def test_orphan_sweep_cancels_resting_only_when_position_is_not_open(db_repos):
    broker = PaperBroker(cash=20_000)
    router = OrderRouter(broker, db_repos, _config(), _universe())
    rest = await router.place(_btc(resting=True, limit=0.20), sleep=_noop_sleep, today=TODAY)
    now = datetime.now(UTC).replace(tzinfo=None)
    pos_id = await db_repos.positions.insert(Position(
        account_id="test", symbol="F", strategy_id="monthly_wheel",
        state=PositionState.CSP_OPEN, shares=0, state_changed_at=now,
    ))
    assert await router.cancel_orphan_resting() == 0
    await db_repos.positions.update_state(pos_id, PositionState.SHARES_HELD, "assigned", when=now)
    assert await router.cancel_orphan_resting() == 1
    broker_view = {
        o.broker_order_id: o for o in await broker.get_orders_since(datetime(2000, 1, 1))
    }
    assert broker_view[rest.placed.broker_order_id].status == OrderStatus.CANCELLED


# -- close pass ---------------------------------------------------------------------


@pytest.fixture
def market_open(monkeypatch):
    monkeypatch.setattr("strategies.wheel_close.is_market_hours", lambda: True)


async def _live_order(db_repos, client_id: str) -> None:
    await db_repos.orders.insert(Order(
        account_id="test", symbol="F", strategy_id="monthly_wheel",
        order_type=OrderType.BUY_TO_CLOSE, contract_symbol=OCC, quantity=1,
        limit_price=0.20, status=OrderStatus.PENDING,
        placed_at=datetime.now(UTC).replace(tzinfo=None),
        client_order_id=client_id, broker_order_id=f"b-{client_id}",
    ))


async def _closes(db_repos, broker, **params: Any):
    return await propose_all_closes(
        broker, db_repos, {"account": {"id": "test"}},
        today=TODAY,
        strategy=_strategy(resting_profit_take=True, csp_stop_loss_mult=2.0, **params),
    )


@pytest.mark.asyncio
async def test_resting_mode_proposes_the_target_order_when_none_is_working(db_repos, market_open):
    await _seed_open_csp(db_repos, fill_price=0.40)
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol=OCC, bid=0.34, ask=0.36))     # no trigger
    out = await _closes(db_repos, broker)
    assert len(out) == 1
    assert out[0].resting and out[0].limit_price == pytest.approx(0.20)
    assert out[0].trigger_reason == "profit"
    assert (out[0].contract.bid, out[0].contract.ask) == (0.34, 0.36)


@pytest.mark.asyncio
async def test_resting_mode_leaves_a_working_order_alone(db_repos, market_open):
    await _seed_open_csp(db_repos, fill_price=0.40)
    await _live_order(db_repos, f"{RESTING_CLIENT_PREFIX}abc")
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol=OCC, bid=0.14, ask=0.16))     # mid below target
    assert await _closes(db_repos, broker) == []                 # no crossing profit close


@pytest.mark.asyncio
async def test_stop_still_fires_while_a_resting_order_works(db_repos, market_open):
    await _seed_open_csp(db_repos, fill_price=0.40)
    await _live_order(db_repos, f"{RESTING_CLIENT_PREFIX}abc")
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol=OCC, bid=0.84, ask=0.88))     # >= 2x premium
    out = await _closes(db_repos, broker)
    assert len(out) == 1
    assert not out[0].resting and out[0].trigger_reason == "stop_loss"
    # the real quote, so the router pays the ask
    assert (out[0].contract.bid, out[0].contract.ask) == (0.84, 0.88)


@pytest.mark.asyncio
async def test_other_inflight_orders_still_skip_the_position(db_repos, market_open):
    await _seed_open_csp(db_repos, fill_price=0.40)
    await _live_order(db_repos, "wb-urgentclose")
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol=OCC, bid=0.84, ask=0.88))
    assert await _closes(db_repos, broker) == []


@pytest.mark.asyncio
async def test_resting_mode_places_nothing_outside_market_hours(db_repos, monkeypatch):
    monkeypatch.setattr("strategies.wheel_close.is_market_hours", lambda: False)
    await _seed_open_csp(db_repos, fill_price=0.40)
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol=OCC, bid=0.34, ask=0.36))
    assert await _closes(db_repos, broker) == []


# -- quote tape --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_quote_tape_records_working_orders_of_listed_strategies(db_repos, monkeypatch):
    monkeypatch.setattr("execution.quote_log.is_market_hours", lambda now=None: True)
    await _live_order(db_repos, f"{RESTING_CLIENT_PREFIX}abc")
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol=OCC, bid=0.18, ask=0.22))
    cfg = {"account": {"id": "test"},
           "execution": {"quote_log": {"enabled": True, "strategies": ["monthly_wheel"]}}}
    assert await log_order_quotes(broker, db_repos, cfg) == 1
    rows = await db_repos.order_quote_log.list_since(datetime(2000, 1, 1))
    got = [(r["phase"], r["bid"], r["ask"], r["limit_price"]) for r in rows]
    assert got == [("working", 0.18, 0.22, 0.20)]
    cfg["execution"]["quote_log"]["strategies"] = ["zero_dte"]
    assert await log_order_quotes(broker, db_repos, cfg) == 0


# -- PULLBACK opt-out ----------------------------------------------------------------


async def _regime(db_repos, regime: str, csps_allowed: bool) -> None:
    c = await db_repos.db.connect()
    await c.execute(
        "INSERT INTO regime_snapshots (snapshot_date, regime, csps_allowed) VALUES (?, ?, ?)",
        (TODAY.isoformat(), regime, int(csps_allowed)),
    )
    await c.commit()


def _csp() -> Proposal:
    return Proposal(
        symbol="F", contract=_contract(bid=0.31, ask=0.33), order_type=OrderType.SELL_TO_OPEN,
        quantity=1, rationale="csp", strategy_id="monthly_wheel",
    )


def _gate_config(**params: Any) -> dict:
    return _config(
        regime={"enabled": True},
        strategies=[
            {"id": "monthly_wheel", "type": "wheel", "max_concurrent": 2, "params": params},
        ],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("regime,opt_out,allowed", [
    ("PULLBACK", True, True),
    ("PULLBACK", False, False),
    ("BEAR_TREND", True, False),
    ("HIGH_VOL", True, False),
])
async def test_pullback_opt_out_only_lifts_pullback(db_repos, regime, opt_out, allowed):
    from risk.limits import RiskCheckFailed, RiskGate

    await _regime(db_repos, regime, csps_allowed=False)
    cfg = _gate_config(pullback_allows_csps=opt_out)
    gate = RiskGate(PaperBroker(cash=20_000), db_repos, cfg, _universe())
    if allowed:
        await gate.evaluate(_csp(), today=TODAY)
    else:
        with pytest.raises(RiskCheckFailed) as exc:
            await gate.evaluate(_csp(), today=TODAY)
        assert exc.value.rule == "regime"
