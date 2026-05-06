"""dashboard/app — auth, view rendering, manual-stop POST, healthz."""

from __future__ import annotations

import base64
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from core.models import (
    DailyState,
    Order,
    OrderStatus,
    OrderType,
    Position,
    PositionState,
    WheelCycle,
)
from dashboard.app import DashboardDeps, build_app
from platforms.paper_broker import PaperBroker


def _auth_header(user: str, pw: str) -> dict[str, str]:
    token = base64.b64encode(f"{user}:{pw}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


@pytest_asyncio.fixture
async def app_client(db_repos, tmp_path):
    broker = PaperBroker(cash=20_000)
    deps = DashboardDeps(
        repos=db_repos,
        broker=broker,
        config={
            "account": {"id": "test", "broker": "paper"},
            "dashboard": {"basic_auth_user": "wheelbot"},
            "risk": {"stop_file_path": str(tmp_path / "STOP")},
        },
        auth_user="wheelbot",
        auth_password="hunter2",
    )
    app = build_app(deps)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, deps, broker


@pytest.mark.asyncio
async def test_unauthenticated_request_gets_401(app_client):
    client, _deps, _broker = app_client
    resp = await client.get("/")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_bad_password_rejected(app_client):
    client, _deps, _broker = app_client
    resp = await client.get("/", headers=_auth_header("wheelbot", "wrong"))
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_positions_view_renders(app_client):
    client, deps, _broker = app_client
    now = datetime.now(UTC).replace(tzinfo=None)
    await deps.repos.positions.insert(
        Position(
            account_id="test",
            symbol="F",
            state=PositionState.SHARES_HELD,
            shares=100,
            cost_basis=9.0,
            state_changed_at=now - timedelta(days=2),
        )
    )
    resp = await client.get("/", headers=_auth_header("wheelbot", "hunter2"))
    assert resp.status_code == 200
    assert "F" in resp.text
    assert "SHARES_HELD" in resp.text


@pytest.mark.asyncio
async def test_positions_partial_returns_polling_div(app_client):
    client, _deps, _broker = app_client
    resp = await client.get(
        "/positions/_table", headers=_auth_header("wheelbot", "hunter2")
    )
    assert resp.status_code == 200
    assert "hx-get=" in resp.text


@pytest.mark.asyncio
async def test_cycles_view_filters_by_symbol(app_client):
    client, deps, _broker = app_client
    now = datetime.now(UTC).replace(tzinfo=None)
    cid = await deps.repos.cycles.insert(
        WheelCycle(account_id="test", symbol="F", started_at=now - timedelta(days=10))
    )
    await deps.repos.cycles.update(
        cid,
        ended_at=now.isoformat(),
        final_pnl=50.0,
        cycle_outcome="CSP_EXPIRED",
        days_held=10,
    )
    resp = await client.get(
        "/cycles?symbol=F", headers=_auth_header("wheelbot", "hunter2")
    )
    assert resp.status_code == 200
    assert "CSP_EXPIRED" in resp.text


@pytest.mark.asyncio
async def test_orders_view_renders_recent_orders(app_client):
    client, deps, _broker = app_client
    await deps.repos.orders.insert(
        Order(
            account_id="test",
            symbol="F",
            order_type=OrderType.SELL_TO_OPEN,
            contract_symbol="F250706P00009500",
            quantity=1,
            limit_price=0.50,
            status=OrderStatus.PENDING,
            placed_at=datetime.now(UTC).replace(tzinfo=None),
            client_order_id="wb-test-1",
        )
    )
    resp = await client.get("/orders", headers=_auth_header("wheelbot", "hunter2"))
    assert resp.status_code == 200
    assert "F250706P00009500" in resp.text


@pytest.mark.asyncio
async def test_candidates_view_handles_empty_state(app_client):
    client, _deps, _broker = app_client
    resp = await client.get("/candidates", headers=_auth_header("wheelbot", "hunter2"))
    assert resp.status_code == 200
    assert "no screener output" in resp.text.lower() or "Sprint 7" in resp.text


@pytest.mark.asyncio
async def test_risk_view_renders(app_client):
    client, _deps, _broker = app_client
    resp = await client.get("/risk", headers=_auth_header("wheelbot", "hunter2"))
    assert resp.status_code == 200
    assert "Kill switch" in resp.text


@pytest.mark.asyncio
async def test_manual_stop_engage_creates_file(app_client):
    client, deps, _broker = app_client
    stop = Path(deps.config["risk"]["stop_file_path"])
    assert not stop.exists()
    resp = await client.post(
        "/risk/manual_stop",
        data={"action": "engage"},
        headers=_auth_header("wheelbot", "hunter2"),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert stop.exists()


@pytest.mark.asyncio
async def test_manual_stop_release_removes_file(app_client):
    client, deps, _broker = app_client
    stop = Path(deps.config["risk"]["stop_file_path"])
    stop.write_text("engaged")
    resp = await client.post(
        "/risk/manual_stop",
        data={"action": "release"},
        headers=_auth_header("wheelbot", "hunter2"),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert not stop.exists()


@pytest.mark.asyncio
async def test_healthz_does_not_require_auth(app_client):
    client, _deps, _broker = app_client
    resp = await client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["broker_connected"] is True


@pytest.mark.asyncio
async def test_missing_password_returns_503(db_repos):
    """If WHEELBOT_DASHBOARD_PASSWORD is unset, requests fail safely (not 200)."""
    deps = DashboardDeps(
        repos=db_repos,
        broker=PaperBroker(cash=10_000),
        config={"account": {"id": "test"}, "dashboard": {"basic_auth_user": "wheelbot"}},
        auth_user="wheelbot",
        auth_password=None,
    )
    app = build_app(deps)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/", headers=_auth_header("wheelbot", "anything"))
        assert resp.status_code == 503
