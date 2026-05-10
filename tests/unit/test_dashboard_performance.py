"""dashboard /performance — stats + cumulative P&L + outcome counts."""

from __future__ import annotations

import base64
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from core.models import WheelCycle
from dashboard.app import DashboardDeps, _performance_data, build_app
from platforms.paper_broker import PaperBroker


def _auth() -> dict[str, str]:
    return {"Authorization": "Basic " + base64.b64encode(b"wheelbot:hunter2").decode()}


def _utc() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


@pytest_asyncio.fixture
async def app_client(db_repos, tmp_path):
    deps = DashboardDeps(
        repos=db_repos,
        broker=PaperBroker(cash=20_000),
        config={
            "account": {"id": "primary"},
            "dashboard": {"basic_auth_user": "wheelbot"},
            "risk": {"stop_file_path": str(tmp_path / "STOP")},
            "strategies": [
                {"id": "monthly_wheel", "display_name": "Monthly Wheel"},
                {"id": "weekly_wheel", "display_name": "Weekly Wheel"},
                {"id": "put_spread", "display_name": "Bull Put Spread"},
            ],
        },
        auth_user="wheelbot",
        auth_password="hunter2",
    )
    app = build_app(deps)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client, deps


async def _seed_cycle(
    repos, *, symbol, pnl, outcome, days_held, ended_offset_days,
    strategy_id: str = "monthly_wheel",
):
    cid = await repos.cycles.insert(
        WheelCycle(
            account_id="primary",
            symbol=symbol,
            strategy_id=strategy_id,
            started_at=_utc() - timedelta(days=days_held + ended_offset_days),
        )
    )
    await repos.cycles.update(
        cid,
        ended_at=(_utc() - timedelta(days=ended_offset_days)).isoformat(),
        final_pnl=pnl,
        cycle_outcome=outcome,
        days_held=days_held,
    )
    return cid


@pytest.mark.asyncio
async def test_empty_state_renders(app_client):
    client, _deps = app_client
    resp = await client.get("/performance", headers=_auth())
    assert resp.status_code == 200
    assert "No closed cycles yet" in resp.text


@pytest.mark.asyncio
async def test_stats_aggregate_correctly(app_client):
    client, deps = app_client
    await _seed_cycle(deps.repos, symbol="F",   pnl=50.0,  outcome="CSP_EXPIRED",       days_held=30, ended_offset_days=20)
    await _seed_cycle(deps.repos, symbol="BAC", pnl=-25.0, outcome="MANUAL_CLOSE",      days_held=20, ended_offset_days=10)
    await _seed_cycle(deps.repos, symbol="F",   pnl=80.0,  outcome="CC_CALLED_AWAY",    days_held=40, ended_offset_days=2)

    data = await _performance_data(deps)
    s = data["stats"]
    assert s["total_realized_pnl_usd"] == pytest.approx(105.0)
    assert s["wins"] == 2
    assert s["losses"] == 1
    assert s["n_closed"] == 3
    assert s["win_rate_pct"] == pytest.approx(66.66666, rel=1e-2)
    assert s["avg_days_held"] == pytest.approx(30.0)


@pytest.mark.asyncio
async def test_cumulative_is_ordered_and_runs(app_client):
    client, deps = app_client
    await _seed_cycle(deps.repos, symbol="F",   pnl=50.0,  outcome="CSP_EXPIRED",    days_held=30, ended_offset_days=20)
    await _seed_cycle(deps.repos, symbol="BAC", pnl=-25.0, outcome="MANUAL_CLOSE",   days_held=20, ended_offset_days=10)
    await _seed_cycle(deps.repos, symbol="F",   pnl=80.0,  outcome="CC_CALLED_AWAY", days_held=40, ended_offset_days=2)

    data = await _performance_data(deps)
    cum = data["cumulative"]
    assert len(cum) == 3
    # Sorted by ended_at ascending; running totals follow.
    assert cum[0]["running"] == pytest.approx(50.0)
    assert cum[1]["running"] == pytest.approx(25.0)
    assert cum[2]["running"] == pytest.approx(105.0)


@pytest.mark.asyncio
async def test_outcome_counts(app_client):
    client, deps = app_client
    await _seed_cycle(deps.repos, symbol="F",   pnl=50.0, outcome="CSP_EXPIRED",    days_held=30, ended_offset_days=20)
    await _seed_cycle(deps.repos, symbol="BAC", pnl=30.0, outcome="CSP_EXPIRED",    days_held=25, ended_offset_days=15)
    await _seed_cycle(deps.repos, symbol="F",   pnl=80.0, outcome="CC_CALLED_AWAY", days_held=40, ended_offset_days=2)

    data = await _performance_data(deps)
    assert data["outcomes"] == {"CSP_EXPIRED": 2, "CC_CALLED_AWAY": 1}


@pytest.mark.asyncio
async def test_open_cycles_counted_separately(app_client):
    client, deps = app_client
    # An open cycle (no ended_at).
    await deps.repos.cycles.insert(
        WheelCycle(account_id="primary", symbol="F", started_at=_utc() - timedelta(days=5))
    )
    await _seed_cycle(deps.repos, symbol="BAC", pnl=20.0, outcome="CSP_EXPIRED", days_held=20, ended_offset_days=2)

    data = await _performance_data(deps)
    assert data["stats"]["open_cycles"] == 1
    assert data["stats"]["n_closed"] == 1


@pytest.mark.asyncio
async def test_view_renders_with_data(app_client):
    client, deps = app_client
    await _seed_cycle(deps.repos, symbol="F", pnl=50.0, outcome="CSP_EXPIRED", days_held=30, ended_offset_days=2)
    resp = await client.get("/performance", headers=_auth())
    assert resp.status_code == 200
    assert "Cumulative realized" in resp.text
    assert "cumChart" in resp.text  # canvas id present


# -- Per-strategy breakdown -----------------------------------------------


@pytest.mark.asyncio
async def test_per_strategy_splits_correctly_and_sums_to_global(app_client):
    client, deps = app_client
    await _seed_cycle(
        deps.repos, symbol="F", pnl=50.0, outcome="CSP_EXPIRED",
        days_held=30, ended_offset_days=20, strategy_id="monthly_wheel",
    )
    await _seed_cycle(
        deps.repos, symbol="HOOD", pnl=15.0, outcome="CSP_EXPIRED",
        days_held=10, ended_offset_days=5, strategy_id="weekly_wheel",
    )
    await _seed_cycle(
        deps.repos, symbol="HOOD", pnl=-5.0, outcome="MANUAL_CLOSE",
        days_held=8, ended_offset_days=2, strategy_id="weekly_wheel",
    )
    await _seed_cycle(
        deps.repos, symbol="AAPL", pnl=30.0, outcome="SPREAD_CLOSED_PROFIT",
        days_held=20, ended_offset_days=1, strategy_id="put_spread",
    )

    data = await _performance_data(deps)

    by_id = {s["id"]: s for s in data["per_strategy"]}
    assert by_id["monthly_wheel"]["total_pnl"] == pytest.approx(50.0)
    assert by_id["monthly_wheel"]["wins"] == 1
    assert by_id["monthly_wheel"]["n_closed"] == 1

    assert by_id["weekly_wheel"]["total_pnl"] == pytest.approx(10.0)
    assert by_id["weekly_wheel"]["wins"] == 1
    assert by_id["weekly_wheel"]["losses"] == 1
    assert by_id["weekly_wheel"]["n_closed"] == 2
    assert by_id["weekly_wheel"]["win_rate_pct"] == pytest.approx(50.0)

    assert by_id["put_spread"]["total_pnl"] == pytest.approx(30.0)
    assert by_id["put_spread"]["n_closed"] == 1

    # Per-strategy sums equal global.
    sum_pnl = sum(s["total_pnl"] for s in data["per_strategy"])
    assert sum_pnl == pytest.approx(data["stats"]["total_realized_pnl_usd"])
    sum_n = sum(s["n_closed"] for s in data["per_strategy"])
    assert sum_n == data["stats"]["n_closed"]


@pytest.mark.asyncio
async def test_per_strategy_preserves_config_order(app_client):
    """Cards render in config order: monthly → weekly → put_spread."""
    client, deps = app_client
    # Insert in scrambled order; ordering should still match config.
    await _seed_cycle(
        deps.repos, symbol="AAPL", pnl=30.0, outcome="SPREAD_CLOSED_PROFIT",
        days_held=20, ended_offset_days=1, strategy_id="put_spread",
    )
    await _seed_cycle(
        deps.repos, symbol="HOOD", pnl=15.0, outcome="CSP_EXPIRED",
        days_held=10, ended_offset_days=5, strategy_id="weekly_wheel",
    )
    await _seed_cycle(
        deps.repos, symbol="F", pnl=50.0, outcome="CSP_EXPIRED",
        days_held=30, ended_offset_days=20, strategy_id="monthly_wheel",
    )

    data = await _performance_data(deps)
    ids = [s["id"] for s in data["per_strategy"]]
    assert ids == ["monthly_wheel", "weekly_wheel", "put_spread"]


@pytest.mark.asyncio
async def test_per_strategy_open_count_attributed_to_right_card(app_client):
    client, deps = app_client
    # An open monthly_wheel cycle.
    await deps.repos.cycles.insert(
        WheelCycle(
            account_id="primary",
            symbol="F",
            strategy_id="monthly_wheel",
            started_at=_utc() - timedelta(days=5),
        )
    )
    # An open put_spread cycle.
    await deps.repos.cycles.insert(
        WheelCycle(
            account_id="primary",
            symbol="AAPL",
            strategy_id="put_spread",
            started_at=_utc() - timedelta(days=2),
        )
    )

    data = await _performance_data(deps)
    by_id = {s["id"]: s for s in data["per_strategy"]}
    assert by_id["monthly_wheel"]["open_cycles"] == 1
    assert by_id["weekly_wheel"]["open_cycles"] == 0
    assert by_id["put_spread"]["open_cycles"] == 1
    assert data["stats"]["open_cycles"] == 2


@pytest.mark.asyncio
async def test_per_strategy_cumulative_runs_per_strategy(app_client):
    """Each strategy's cumulative series is its own running total, not global."""
    client, deps = app_client
    await _seed_cycle(
        deps.repos, symbol="F", pnl=50.0, outcome="CSP_EXPIRED",
        days_held=30, ended_offset_days=20, strategy_id="monthly_wheel",
    )
    await _seed_cycle(
        deps.repos, symbol="HOOD", pnl=15.0, outcome="CSP_EXPIRED",
        days_held=10, ended_offset_days=15, strategy_id="weekly_wheel",
    )
    await _seed_cycle(
        deps.repos, symbol="F", pnl=20.0, outcome="CSP_EXPIRED",
        days_held=25, ended_offset_days=5, strategy_id="monthly_wheel",
    )

    data = await _performance_data(deps)
    by_id = {s["id"]: s for s in data["per_strategy"]}
    monthly_running = [p["running"] for p in by_id["monthly_wheel"]["cumulative"]]
    assert monthly_running == [pytest.approx(50.0), pytest.approx(70.0)]
    weekly_running = [p["running"] for p in by_id["weekly_wheel"]["cumulative"]]
    assert weekly_running == [pytest.approx(15.0)]


@pytest.mark.asyncio
async def test_legacy_cycles_with_null_strategy_default_to_monthly_wheel(app_client):
    """Legacy rows without strategy_id are rolled into monthly_wheel."""
    client, deps = app_client
    cid = await deps.repos.cycles.insert(
        WheelCycle(
            account_id="primary",
            symbol="F",
            # strategy_id intentionally omitted (None)
            started_at=_utc() - timedelta(days=30),
        )
    )
    await deps.repos.cycles.update(
        cid,
        ended_at=_utc().isoformat(),
        final_pnl=12.0,
        cycle_outcome="CSP_EXPIRED",
        days_held=25,
    )
    data = await _performance_data(deps)
    by_id = {s["id"]: s for s in data["per_strategy"]}
    assert by_id["monthly_wheel"]["total_pnl"] == pytest.approx(12.0)


@pytest.mark.asyncio
async def test_view_renders_per_strategy_section(app_client):
    client, deps = app_client
    await _seed_cycle(
        deps.repos, symbol="F", pnl=50.0, outcome="CSP_EXPIRED",
        days_held=30, ended_offset_days=2, strategy_id="monthly_wheel",
    )
    resp = await client.get("/performance", headers=_auth())
    assert resp.status_code == 200
    assert "By strategy" in resp.text
    assert "Monthly Wheel" in resp.text
    assert "Weekly Wheel" in resp.text  # empty card still rendered
    assert "Bull Put Spread" in resp.text
