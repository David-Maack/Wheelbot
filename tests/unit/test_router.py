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
    broker = PaperBroker(cash=20_000)
    router = OrderRouter(broker, db_repos, _config(), _universe())
    first = await router.place(_proposal(), sleep=_noop_sleep, today=date(2025, 6, 1))
    second = await router.place(_proposal(), sleep=_noop_sleep, today=date(2025, 6, 1))
    assert first.placed.client_order_id == second.placed.client_order_id
    assert first.placed.broker_order_id == second.placed.broker_order_id
    # Only one orders row.
    rows = await db_repos.orders.list_recent("test")
    assert len(rows) == 1


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


@pytest.mark.asyncio
async def test_client_order_id_is_deterministic_per_day():
    p1 = _proposal()
    p2 = _proposal()
    assert _client_order_id(p1, date(2025, 6, 1)) == _client_order_id(p2, date(2025, 6, 1))
    assert _client_order_id(p1, date(2025, 6, 1)) != _client_order_id(p1, date(2025, 6, 2))
