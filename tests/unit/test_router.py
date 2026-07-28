"""execution/router tests — risk gate, idempotency, retry, dry-run."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from core.broker import BrokerUnavailable, OrderRejected
from core.models import (
    OptionContract,
    OptionType,
    OrderStatus,
    OrderType,
    Position,
    PositionState,
    UniverseEntry,
)
from execution.router import OrderRouter, _client_order_id
from platforms.paper_broker import PaperBroker
from strategies.wheel import Proposal


def _config(dry_run: bool = False, **wheel_overrides: Any) -> dict:
    base = {
        "account": {"id": "test", "broker": "paper"},
        "wheel": {
            "buying_power_floor_pct": 20,
            "max_position_pct_of_account": 30,
            "max_concurrent_positions": 4,
            "open_interest_min": 100,
            "volume_min": 50,
            "bid_ask_spread_max_pct": 10.0,
        },
        "regime": {"enabled": False},
        "execution": {
            "dry_run": dry_run,
            "retry_max_attempts": 3,
            "retry_initial_backoff_seconds": 0,
            "retry_max_backoff_seconds": 0,
        },
    }
    base["wheel"].update(wheel_overrides)
    return base


def _universe() -> dict:
    return {
        "tickers": [UniverseEntry(symbol="F", name="Ford", tier=1, overrides={})],
        "banned": [],
        "banned_rules": [],
    }


def _put_contract() -> OptionContract:
    today = date(2025, 6, 1)
    return OptionContract(
        underlying="F",
        occ_symbol="F250706P00009500",
        strike=9.5,
        expiration=today + timedelta(days=35),
        option_type=OptionType.PUT,
        bid=0.39,
        ask=0.41,
        delta=-0.25,
        open_interest=1000,
        volume=200,
    )


def _proposal() -> Proposal:
    return Proposal(
        symbol="F",
        contract=_put_contract(),
        order_type=OrderType.SELL_TO_OPEN,
        quantity=1,
        rationale="csp test",
    )


async def _noop_sleep(seconds: float) -> None:
    return None


@pytest.fixture(autouse=True)
def _stub_earnings(monkeypatch):
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)


@pytest.mark.asyncio
async def test_happy_path_places_order_and_writes_db(db_repos):
    broker = PaperBroker(cash=20_000)
    router = OrderRouter(broker, db_repos, _config(), _universe())
    result = await router.place(_proposal(), sleep=_noop_sleep, today=date(2025, 6, 1))
    assert result.placed is not None
    assert result.placed.broker_order_id is not None
    pos = await db_repos.positions.get_by_symbol("test", "F")
    assert pos is not None
    assert pos.state == PositionState.CSP_PENDING
    persisted = await db_repos.orders.get_by_client_id(result.placed.client_order_id)
    assert persisted is not None
    assert persisted.status == OrderStatus.PENDING


@pytest.mark.asyncio
async def test_risk_failure_short_circuits_and_no_writes(db_repos):
    broker = PaperBroker(cash=500)  # too small → BP floor fail
    router = OrderRouter(broker, db_repos, _config(), _universe())
    result = await router.place(_proposal(), sleep=_noop_sleep, today=date(2025, 6, 1))
    assert result.risk_failed is True
    assert result.placed is None
    assert (await db_repos.positions.get_by_symbol("test", "F")) is None
    cid = _client_order_id(_proposal(), date(2025, 6, 1))
    assert (await db_repos.orders.get_by_client_id(cid)) is None


@pytest.mark.asyncio
async def test_idempotent_resubmit_with_same_client_id(db_repos):
    """Double-submit safety: second call within the stale-pending window
    detects the same client_order_id is already PENDING locally and skips
    the submission entirely (no broker-side dedup needed)."""
    broker = PaperBroker(cash=20_000)
    router = OrderRouter(broker, db_repos, _config(), _universe())
    first = await router.place(_proposal(), sleep=_noop_sleep, today=date(2025, 6, 1))
    second = await router.place(_proposal(), sleep=_noop_sleep, today=date(2025, 6, 1))
    assert first.placed is not None
    assert second.placed is None
    assert second.skipped_duplicate_pending is True
    # Only one orders row — first call created it; second skipped before insert.
    rows = await db_repos.orders.list_recent("test")
    assert len(rows) == 1
    assert rows[0].client_order_id == first.placed.client_order_id


@pytest.mark.asyncio
async def test_retry_on_broker_unavailable_then_succeeds(db_repos):
    broker = PaperBroker(cash=20_000)
    real_place = broker.place_order
    calls = {"n": 0}

    async def flaky(order):
        calls["n"] += 1
        if calls["n"] < 3:
            raise BrokerUnavailable("transient")
        return await real_place(order)

    broker.place_order = flaky  # type: ignore[assignment]

    router = OrderRouter(broker, db_repos, _config(), _universe())
    result = await router.place(_proposal(), sleep=_noop_sleep, today=date(2025, 6, 1))
    assert result.placed is not None
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_order_rejected_does_not_retry(db_repos):
    broker = PaperBroker(cash=20_000)
    calls = {"n": 0}

    async def reject(order):
        calls["n"] += 1
        raise OrderRejected("bad")

    broker.place_order = reject  # type: ignore[assignment]
    router = OrderRouter(broker, db_repos, _config(), _universe())
    with pytest.raises(OrderRejected):
        await router.place(_proposal(), sleep=_noop_sleep, today=date(2025, 6, 1))
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_dry_run_skips_broker_and_db(db_repos):
    broker = PaperBroker(cash=20_000)
    calls = {"n": 0}
    real_place = broker.place_order

    async def counted(order):
        calls["n"] += 1
        return await real_place(order)

    broker.place_order = counted  # type: ignore[assignment]
    router = OrderRouter(broker, db_repos, _config(dry_run=True), _universe())
    result = await router.place(_proposal(), sleep=_noop_sleep, today=date(2025, 6, 1))
    assert result.dry_run is True
    assert result.placed is None
    assert calls["n"] == 0
    assert (await db_repos.positions.get_by_symbol("test", "F")) is None
    assert len(await db_repos.orders.list_recent("test")) == 0


def test_option_limit_price_pennies_below_three_dollars():
    from execution.router import _option_limit_price
    # 0.13/0.14 → mid 0.135 → 0.14 (nearest cent, banker's rounding goes to 0.14)
    assert _option_limit_price(0.13, 0.14) in (0.13, 0.14)
    # 0.50/0.52 → mid 0.51 → 0.51
    assert _option_limit_price(0.50, 0.52) == pytest.approx(0.51)
    # Sub-penny mid floors to 0.01 (we never send 0.00 limit on SELL)
    assert _option_limit_price(0.00, 0.01) == pytest.approx(0.01)


def test_option_limit_price_nickels_at_or_above_three_dollars():
    from execution.router import _option_limit_price
    # 3.00/3.10 → mid 3.05 → 3.05 (already on nickel)
    assert _option_limit_price(3.00, 3.10) == pytest.approx(3.05)
    # 4.13/4.18 → mid 4.155 → 4.15 (nearest nickel)
    assert _option_limit_price(4.13, 4.18) == pytest.approx(4.15)
    # 7.97/8.05 → mid 8.01 → 8.00
    assert _option_limit_price(7.97, 8.05) == pytest.approx(8.00)


def test_option_limit_price_crosses_to_fill():
    from execution.router import _option_limit_price
    # A mid limit on a directional buy never fills — cross to the ask (buy) /
    # bid (exit-sell); patient sells stay at mid.
    assert _option_limit_price(9.10, 9.20) == pytest.approx(9.15)             # mid (default)
    assert _option_limit_price(9.10, 9.20, cross="ask") == pytest.approx(9.20)  # pay the offer
    assert _option_limit_price(9.10, 9.20, cross="bid") == pytest.approx(9.10)  # hit the bid
    assert _option_limit_price(0.50, 0.54, cross="ask") == pytest.approx(0.54)  # penny tick


def test_option_limit_price_returns_none_on_missing_quote():
    from execution.router import _option_limit_price
    assert _option_limit_price(None, 0.50) is None
    assert _option_limit_price(0.50, None) is None
    assert _option_limit_price(None, None) is None


@pytest.mark.asyncio
async def test_client_order_id_is_deterministic_per_day():
    p1 = _proposal()
    p2 = _proposal()
    assert _client_order_id(p1, date(2025, 6, 1)) == _client_order_id(p2, date(2025, 6, 1))
    assert _client_order_id(p1, date(2025, 6, 1)) != _client_order_id(p1, date(2025, 6, 2))


# -- Stale PENDING handling (Sprint 14) -------------------------------------


@pytest.mark.asyncio
async def test_stale_pending_skipped_when_fresh(db_repos, monkeypatch):
    """A PENDING order with the same client_order_id placed minutes ago is
    NOT replaced — broker may still fill the original limit."""
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)
    from core.models import Order
    broker = PaperBroker(cash=20_000)
    cfg = _config()
    cfg["execution"]["stale_pending_minutes"] = 15.0
    router = OrderRouter(broker, db_repos, cfg, _universe())
    proposal = _proposal()
    from execution.router import _client_order_id
    cid = _client_order_id(proposal, date(2025, 6, 1))
    # Seed an existing PENDING order with the same client_order_id, placed 2 min ago.
    now = datetime.now(UTC).replace(tzinfo=None)
    await db_repos.orders.insert(
        Order(
            account_id="test",
            symbol="F",
            order_type=OrderType.SELL_TO_OPEN,
            contract_symbol="F250706P00009500",
            strike=9.5, expiration=date(2025, 7, 6), option_type=OptionType.PUT,
            quantity=1, limit_price=0.40,
            status=OrderStatus.PENDING,
            placed_at=now - timedelta(minutes=2),
            client_order_id=cid,
            broker_order_id="broker-existing-1",
        )
    )
    result = await router.place(proposal, sleep=_noop_sleep, today=date(2025, 6, 1))
    assert result.placed is None
    assert result.skipped_duplicate_pending is True


@pytest.mark.asyncio
async def test_stale_pending_cancelled_and_replaced(db_repos, monkeypatch):
    """A PENDING order older than the threshold is cancelled at the broker
    and the new proposal goes through with a fresh client_order_id."""
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)
    from core.models import Order
    broker = PaperBroker(cash=20_000)
    # Pre-place the "stale" order at the broker so cancel_order finds it.
    stale_placed = await broker.place_order(
        Order(
            account_id="test",
            symbol="F",
            order_type=OrderType.SELL_TO_OPEN,
            contract_symbol="F250706P00009500",
            strike=9.5, expiration=date(2025, 7, 6), option_type=OptionType.PUT,
            quantity=1, limit_price=0.40,
            status=OrderStatus.PENDING,
            placed_at=datetime.now(UTC).replace(tzinfo=None),
            client_order_id="some-existing-broker-id",
        )
    )
    cfg = _config()
    cfg["execution"]["stale_pending_minutes"] = 5.0
    router = OrderRouter(broker, db_repos, cfg, _universe())
    proposal = _proposal()
    from execution.router import _client_order_id
    cid = _client_order_id(proposal, date(2025, 6, 1))
    # Persist local record with the stale-aged placed_at, referencing the broker order.
    now = datetime.now(UTC).replace(tzinfo=None)
    await db_repos.orders.insert(
        Order(
            account_id="test",
            symbol="F",
            order_type=OrderType.SELL_TO_OPEN,
            contract_symbol="F250706P00009500",
            strike=9.5, expiration=date(2025, 7, 6), option_type=OptionType.PUT,
            quantity=1, limit_price=0.40,
            status=OrderStatus.PENDING,
            placed_at=now - timedelta(minutes=30),  # well past 5-min threshold
            client_order_id=cid,
            broker_order_id=stale_placed.broker_order_id,
        )
    )

    result = await router.place(proposal, sleep=_noop_sleep, today=date(2025, 6, 1))

    # New order DID place — replacement client_order_id, not the stale one.
    assert result.placed is not None
    assert result.placed.client_order_id != cid
    assert result.placed.client_order_id.startswith(f"{cid}-r")
    # Broker shows the stale order as CANCELLED.
    stale_after = broker._orders[stale_placed.broker_order_id]
    assert stale_after.status == OrderStatus.CANCELLED


@pytest.mark.asyncio
async def test_no_existing_pending_proceeds_normally(db_repos, monkeypatch):
    """Regression: when nothing exists with the client_order_id, place flows
    through unchanged."""
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)
    broker = PaperBroker(cash=20_000)
    router = OrderRouter(broker, db_repos, _config(), _universe())
    result = await router.place(_proposal(), sleep=_noop_sleep, today=date(2025, 6, 1))
    assert result.placed is not None
    assert result.skipped_duplicate_pending is False


# -- Entry-window gate (Sprint 14) ------------------------------------------


@pytest.mark.asyncio
async def test_open_skipped_outside_entry_window(db_repos, monkeypatch):
    """A CSP open attempted near/after close is skipped, not placed."""
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)
    # Force the gate CLOSED (override the conftest autouse default).
    monkeypatch.setattr("execution.router.within_entry_window", lambda **k: False)
    broker = PaperBroker(cash=20_000)
    router = OrderRouter(broker, db_repos, _config(), _universe())
    result = await router.place(_proposal(), sleep=_noop_sleep, today=date(2025, 6, 1))
    assert result.placed is None
    assert result.skipped_outside_entry_window is True
    # Nothing persisted.
    rows = await db_repos.orders.list_recent("test")
    assert len(rows) == 0


@pytest.mark.asyncio
async def test_close_not_gated_by_entry_window(db_repos, monkeypatch):
    """A BUY_TO_CLOSE must still go through even after the close cutoff —
    exiting late is fine, only entries are gated."""
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)
    monkeypatch.setattr("execution.router.within_entry_window", lambda **k: False)
    broker = PaperBroker(cash=20_000)
    router = OrderRouter(broker, db_repos, _config(), _universe())
    close_proposal = Proposal(
        symbol="F",
        contract=_put_contract(),
        order_type=OrderType.BUY_TO_CLOSE,
        quantity=1,
        rationale="close",
    )
    result = await router.place(close_proposal, sleep=_noop_sleep, today=date(2025, 6, 1))
    assert result.placed is not None
    assert result.skipped_outside_entry_window is False


# -- stale-open sweep (2026-07-08 fix) -----------------------------------------


class _CancelRecorder:
    """Broker stub for the sweep — records cancel calls, optionally raises."""

    def __init__(self, fail_ids: set[str] | None = None):
        self.cancelled: list[str] = []
        self._fail_ids = fail_ids or set()

    async def cancel_order(self, broker_order_id: str) -> None:
        if broker_order_id in self._fail_ids:
            raise BrokerUnavailable("cancel refused")
        self.cancelled.append(broker_order_id)


async def _insert_order(
    db_repos,
    *,
    client_id: str,
    broker_id: str | None,
    order_type: OrderType,
    age_minutes: float,
    status: OrderStatus = OrderStatus.PENDING,
    symbol: str = "AAPL",
):
    from core.models import Order
    placed = datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=age_minutes)
    await db_repos.orders.insert(Order(
        account_id="test", symbol=symbol, strategy_id="put_spread",
        order_type=order_type, quantity=1, status=status,
        placed_at=placed, client_order_id=client_id, broker_order_id=broker_id,
    ))


@pytest.mark.asyncio
async def test_sweep_cancels_stale_pending_open(db_repos):
    broker = _CancelRecorder()
    router = OrderRouter(broker, db_repos, _config(), _universe())
    await _insert_order(db_repos, client_id="c1", broker_id="b1",
                        order_type=OrderType.MULTI_LEG_OPEN, age_minutes=30)
    n = await router.cancel_stale_pending_opens()
    assert n == 1
    assert broker.cancelled == ["b1"]
    # No local write — the reconciler observes the broker cancel and drives
    # the status + position transitions (single source of truth).
    order = await db_repos.orders.get_by_client_id("c1")
    assert order.status == OrderStatus.PENDING


@pytest.mark.asyncio
async def test_sweep_leaves_fresh_and_nonopen_and_partial(db_repos):
    broker = _CancelRecorder()
    router = OrderRouter(broker, db_repos, _config(), _universe())
    # Fresh open — inside the stale window.
    await _insert_order(db_repos, client_id="fresh", broker_id="b-fresh",
                        order_type=OrderType.MULTI_LEG_OPEN, age_minutes=5)
    # Stale CLOSE — exempt (closes re-propose every tick; replace path covers them).
    await _insert_order(db_repos, client_id="close", broker_id="b-close",
                        order_type=OrderType.MULTI_LEG_CLOSE, age_minutes=60)
    # Stale PARTIAL open — human territory, never auto-cancel.
    await _insert_order(db_repos, client_id="part", broker_id="b-part",
                        order_type=OrderType.MULTI_LEG_OPEN, age_minutes=60,
                        status=OrderStatus.PARTIAL)
    # No broker id yet — nothing to cancel.
    await _insert_order(db_repos, client_id="nobrk", broker_id=None,
                        order_type=OrderType.MULTI_LEG_OPEN, age_minutes=60)
    n = await router.cancel_stale_pending_opens()
    assert n == 0
    assert broker.cancelled == []


@pytest.mark.asyncio
async def test_sweep_covers_single_leg_opens_too(db_repos):
    """A wheel CSP entry that never fills dead-ends the same way (CSP_PENDING
    suppresses re-proposals) — the sweep covers all _OPEN_ORDER_TYPES."""
    broker = _CancelRecorder()
    router = OrderRouter(broker, db_repos, _config(), _universe())
    await _insert_order(db_repos, client_id="csp", broker_id="b-csp",
                        order_type=OrderType.SELL_TO_OPEN, age_minutes=30, symbol="F")
    n = await router.cancel_stale_pending_opens()
    assert n == 1
    assert broker.cancelled == ["b-csp"]


@pytest.mark.asyncio
async def test_sweep_survives_broker_cancel_failure(db_repos):
    broker = _CancelRecorder(fail_ids={"b-bad"})
    router = OrderRouter(broker, db_repos, _config(), _universe())
    await _insert_order(db_repos, client_id="bad", broker_id="b-bad",
                        order_type=OrderType.MULTI_LEG_OPEN, age_minutes=30)
    await _insert_order(db_repos, client_id="good", broker_id="b-good",
                        order_type=OrderType.MULTI_LEG_OPEN, age_minutes=30, symbol="MSFT")
    n = await router.cancel_stale_pending_opens()
    assert n == 1
    assert broker.cancelled == ["b-good"]


# -- fresh client id after a terminal row (2026-07-23 review fix) --------------


@pytest.mark.asyncio
async def test_same_day_reentry_gets_fresh_client_id(db_repos):
    """A terminal (FILLED/CANCELLED) row under the day-scoped id means that id
    is spent at the broker — reusing it 422-rejected every same-day re-entry.
    The router now mints a suffixed replacement id."""
    broker = PaperBroker(cash=40_000)
    router = OrderRouter(broker, db_repos, _config(), _universe())
    r1 = await router.place(_proposal(), sleep=_noop_sleep, today=date(2025, 6, 1))
    assert r1.placed is not None
    first_id = r1.placed.client_order_id

    # Mark the first order terminal + reset the position so a re-entry is legal.
    first_row = await db_repos.orders.get_by_client_id(first_id)
    await db_repos.orders.update(first_row.id, status="CANCELLED")
    pos = await db_repos.positions.get_by_symbol("test", "F")
    await db_repos.positions.update_state(pos.id, PositionState.IDLE, "test reset")

    r2 = await router.place(_proposal(), sleep=_noop_sleep, today=date(2025, 6, 1))
    assert r2.placed is not None
    assert r2.placed.client_order_id != first_id
    assert r2.placed.client_order_id.startswith(first_id)  # "-r<ts>" suffix
