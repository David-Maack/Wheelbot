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
    OptionType,
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
async def test_positions_renders_pmcc_both_legs(app_client):
    """TICKET-015: a PMCC_BOTH_OPEN position shows both legs (long + short,
    different expirations) and the breakeven."""
    client, deps, _broker = app_client
    now = datetime.now(UTC).replace(tzinfo=None)
    cycle_id = await deps.repos.cycles.insert(
        WheelCycle(account_id="test", symbol="AAPL", strategy_id="pmcc",
                   started_at=now, n_orders=2)
    )
    await deps.repos.orders.insert(
        Order(account_id="test", symbol="AAPL", strategy_id="pmcc", cycle_id=cycle_id,
              order_type=OrderType.BUY_TO_OPEN, contract_symbol="AAPL260116C00140000",
              strike=140.0, expiration=date(2026, 1, 16), option_type=OptionType.CALL,
              quantity=1, fill_price=16.0, status=OrderStatus.FILLED,
              placed_at=now, client_order_id="pl")
    )
    await deps.repos.orders.insert(
        Order(account_id="test", symbol="AAPL", strategy_id="pmcc", cycle_id=cycle_id,
              order_type=OrderType.SELL_TO_OPEN, contract_symbol="AAPL250620C00160000",
              strike=160.0, expiration=date(2025, 6, 20), option_type=OptionType.CALL,
              quantity=1, fill_price=1.20, status=OrderStatus.FILLED,
              placed_at=now, client_order_id="ps")
    )
    await deps.repos.positions.insert(
        Position(account_id="test", symbol="AAPL", strategy_id="pmcc",
                 state=PositionState.PMCC_BOTH_OPEN, shares=0,
                 current_cycle_id=cycle_id, state_changed_at=now)
    )
    resp = await client.get("/", headers=_auth_header("wheelbot", "hunter2"))
    assert resp.status_code == 200
    body = resp.text
    assert "PMCC_BOTH_OPEN" in body
    assert "L 140.0c" in body          # long leg
    assert "S 160.0c" in body          # short leg
    assert "be 156.00" in body         # breakeven = 140 + 16


@pytest.mark.asyncio
async def test_positions_renders_pmcc_long_only(app_client):
    """A PMCC_LONG_OPEN position (no short yet) shows the long + a 'no short'
    indicator."""
    client, deps, _broker = app_client
    now = datetime.now(UTC).replace(tzinfo=None)
    cycle_id = await deps.repos.cycles.insert(
        WheelCycle(account_id="test", symbol="AAPL", strategy_id="pmcc",
                   started_at=now, n_orders=1)
    )
    await deps.repos.orders.insert(
        Order(account_id="test", symbol="AAPL", strategy_id="pmcc", cycle_id=cycle_id,
              order_type=OrderType.BUY_TO_OPEN, contract_symbol="AAPL260116C00140000",
              strike=140.0, expiration=date(2026, 1, 16), option_type=OptionType.CALL,
              quantity=1, fill_price=16.0, status=OrderStatus.FILLED,
              placed_at=now, client_order_id="pl")
    )
    await deps.repos.positions.insert(
        Position(account_id="test", symbol="AAPL", strategy_id="pmcc",
                 state=PositionState.PMCC_LONG_OPEN, shares=0,
                 current_cycle_id=cycle_id, state_changed_at=now)
    )
    resp = await client.get("/", headers=_auth_header("wheelbot", "hunter2"))
    assert resp.status_code == 200
    body = resp.text
    assert "PMCC_LONG_OPEN" in body
    assert "L 140.0c" in body
    assert "no short" in body


@pytest.mark.asyncio
async def test_positions_partial_returns_polling_div(app_client):
    client, _deps, _broker = app_client
    resp = await client.get(
        "/positions/_table", headers=_auth_header("wheelbot", "hunter2")
    )
    assert resp.status_code == 200
    assert "hx-get=" in resp.text


# -- Multi-leg position rows: DTE + unrealized from MULTI_LEG_OPEN ----------


async def _seed_spread_position(
    deps,
    broker,
    *,
    symbol: str,
    strategy_id: str,
    short_strike: float,
    long_strike: float,
    option_type: str,  # "PUT" for bull_put, "CALL" for bear_call
    fill_price: float,
    expiration: date,
    short_mid: float,
    long_mid: float,
    state: PositionState = PositionState.SPREAD_OPEN,
):
    """Create a position with a FILLED MULTI_LEG_OPEN parent order.

    Mirrors what the reconciler does after a multi-leg fill: cycle row,
    position (SPREAD_OPEN by default), parent order with `raw_request["legs"]`.
    Pass state=SPREAD_PENDING to simulate a close order in flight on an open
    spread (the cycle stays open until the close fills).
    """
    from core.models import OrderType as OT
    now = datetime.now(UTC).replace(tzinfo=None)
    cycle_id = await deps.repos.cycles.insert(
        WheelCycle(
            account_id="test",
            symbol=symbol,
            strategy_id=strategy_id,
            started_at=now,
            initial_csp_premium=fill_price,
            initial_capital_at_risk=350.0,
        )
    )
    short_occ = f"{symbol}{expiration.strftime('%y%m%d')}{option_type[0]}{int(short_strike * 1000):08d}"
    long_occ = f"{symbol}{expiration.strftime('%y%m%d')}{option_type[0]}{int(long_strike * 1000):08d}"
    legs_raw = [
        {
            "contract_symbol": short_occ,
            "underlying": symbol,
            "option_type": option_type,
            "strike": short_strike,
            "expiration": expiration.isoformat(),
            "action": "SELL_TO_OPEN",
            "ratio_qty": 1,
        },
        {
            "contract_symbol": long_occ,
            "underlying": symbol,
            "option_type": option_type,
            "strike": long_strike,
            "expiration": expiration.isoformat(),
            "action": "BUY_TO_OPEN",
            "ratio_qty": 1,
        },
    ]
    await deps.repos.orders.insert(
        Order(
            account_id="test",
            symbol=symbol,
            strategy_id=strategy_id,
            cycle_id=cycle_id,
            order_type=OT.MULTI_LEG_OPEN,
            quantity=1,
            limit_price=fill_price,
            fill_price=fill_price,
            status=OrderStatus.FILLED,
            placed_at=now,
            filled_at=now,
            raw_request={"legs": legs_raw},
        )
    )
    await deps.repos.positions.insert(
        Position(
            account_id="test",
            symbol=symbol,
            strategy_id=strategy_id,
            state=state,
            shares=0,
            current_cycle_id=cycle_id,
            state_changed_at=now,
        )
    )
    # Seed quotes for both legs.
    from core.models import Quote
    broker.seed_quote(Quote(symbol=short_occ, bid=short_mid - 0.01, ask=short_mid + 0.01))
    broker.seed_quote(Quote(symbol=long_occ, bid=long_mid - 0.01, ask=long_mid + 0.01))


@pytest.mark.asyncio
async def test_positions_row_populates_dte_and_unrealized_for_put_spread(app_client):
    """SPREAD_OPEN put_spread must show DTE + unrealized on the dashboard.

    Regression: before this fix, _latest_open_option_for_cycle only queried
    SELL_TO_OPEN, so multi-leg positions returned None → dashboard showed —.
    """
    from dashboard.app import _positions_rows, _QuoteCache
    _client, deps, broker = app_client
    expiration = datetime.now(UTC).date() + timedelta(days=30)
    await _seed_spread_position(
        deps, broker,
        symbol="F", strategy_id="put_spread",
        short_strike=10.0, long_strike=9.0, option_type="PUT",
        fill_price=0.30,
        expiration=expiration,
        short_mid=0.18, long_mid=0.06,  # debit-to-close = 0.18 - 0.06 = 0.12
    )

    cache = _QuoteCache(ttl_seconds=60)
    rows = await _positions_rows(deps, cache)
    spread_rows = [r for r in rows if r["strategy_id"] == "put_spread"]
    assert len(spread_rows) == 1
    row = spread_rows[0]
    assert row["dte"] == 30
    # original credit 0.30, current debit 0.12 → unrealized = (0.30 - 0.12) × 100 × 1 = 18
    assert row["unrealized"] == pytest.approx(18.0)
    # P&L % = (0.30 - 0.12) / 0.30 * 100 = 60% captured
    assert row["unrealized_pct"] == pytest.approx(60.0)


@pytest.mark.asyncio
async def test_spread_pending_never_shows_pnl_even_with_attached_cycle(app_client):
    """The bug David hit: a SPREAD_PENDING position whose current_cycle_id points
    at a (stale/earlier) cycle with a FILLED open leg was showing that cycle's
    DTE + unrealized P&L — so a pending OPEN looked like it already had a gain.
    Pending means nothing is settled; the dashboard must show "—" regardless of
    any attached cycle."""
    from dashboard.app import _positions_rows, _QuoteCache
    _client, deps, broker = app_client
    expiration = datetime.now(UTC).date() + timedelta(days=20)
    await _seed_spread_position(
        deps, broker,
        symbol="MSFT", strategy_id="put_spread",
        short_strike=400.0, long_strike=395.0, option_type="PUT",
        fill_price=0.80,
        expiration=expiration,
        short_mid=0.12, long_mid=0.04,  # would compute a big gain if mis-attributed
        state=PositionState.SPREAD_PENDING,  # order in flight, nothing settled
    )

    cache = _QuoteCache(ttl_seconds=60)
    rows = await _positions_rows(deps, cache)
    row = next(r for r in rows if r["symbol"] == "MSFT")
    assert row["state"] == "SPREAD_PENDING"   # honest — order still in flight
    assert row["dte"] is None
    assert row["unrealized"] is None
    assert row["unrealized_pct"] is None


@pytest.mark.asyncio
async def test_spread_open_still_shows_pnl(app_client):
    """Control: a genuinely OPEN spread still shows DTE + unrealized (the fix
    only suppresses P&L for *_PENDING states, not open ones)."""
    from dashboard.app import _positions_rows, _QuoteCache
    _client, deps, broker = app_client
    expiration = datetime.now(UTC).date() + timedelta(days=20)
    await _seed_spread_position(
        deps, broker,
        symbol="MSFT", strategy_id="put_spread",
        short_strike=400.0, long_strike=395.0, option_type="PUT",
        fill_price=0.80,
        expiration=expiration,
        short_mid=0.12, long_mid=0.04,  # debit-to-close 0.08
        state=PositionState.SPREAD_OPEN,
    )

    cache = _QuoteCache(ttl_seconds=60)
    rows = await _positions_rows(deps, cache)
    row = next(r for r in rows if r["symbol"] == "MSFT")
    assert row["state"] == "SPREAD_OPEN"
    assert row["dte"] == 20
    # (0.80 - 0.08) × 100 = 72
    assert row["unrealized"] == pytest.approx(72.0)


# -- TICKET-005: trigger_reason column ------------------------------------


async def _seed_csp_with_close(
    deps,
    *,
    symbol: str,
    contract: str,
    trigger_reason: str | None,
):
    """Seed CSP_OPEN with an open cycle + a FILLED BUY_TO_CLOSE carrying the
    given trigger_reason. Returns the position's id."""
    now = datetime.now(UTC).replace(tzinfo=None)
    cycle_id = await deps.repos.cycles.insert(
        WheelCycle(
            account_id="test", symbol=symbol, strategy_id="monthly_wheel",
            started_at=now,
        )
    )
    await deps.repos.orders.insert(
        Order(
            account_id="test", symbol=symbol, strategy_id="monthly_wheel",
            cycle_id=cycle_id,
            order_type=OrderType.SELL_TO_OPEN, contract_symbol=contract,
            strike=10.0, expiration=date(2026, 8, 15),
            option_type=__import__("core.models", fromlist=["OptionType"]).OptionType.PUT,
            quantity=1, fill_price=1.00, status=OrderStatus.FILLED,
            placed_at=now, filled_at=now,
        )
    )
    if trigger_reason is not None:
        await deps.repos.orders.insert(
            Order(
                account_id="test", symbol=symbol, strategy_id="monthly_wheel",
                cycle_id=cycle_id,
                order_type=OrderType.BUY_TO_CLOSE, contract_symbol=contract,
                strike=10.0, expiration=date(2026, 8, 15),
                option_type=__import__("core.models", fromlist=["OptionType"]).OptionType.PUT,
                quantity=1, fill_price=0.50, status=OrderStatus.FILLED,
                placed_at=now + timedelta(minutes=1),
                filled_at=now + timedelta(minutes=1),
                trigger_reason=trigger_reason,
            )
        )
    pos_id = await deps.repos.positions.insert(
        Position(
            account_id="test", symbol=symbol, strategy_id="monthly_wheel",
            state=PositionState.CSP_OPEN,
            shares=0, current_cycle_id=cycle_id,
            state_changed_at=now,
        )
    )
    return pos_id


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger", ["delta_stop_close", "delta_stop_close_fallback"])
async def test_positions_table_renders_close_trigger_column(app_client, trigger):
    """Both delta_stop_close AND the new delta_stop_close_fallback value must
    render through the template. Parametrised so anyone hardcoding the column
    to recognise only one value gets caught immediately."""
    client, deps, _broker = app_client
    await _seed_csp_with_close(
        deps, symbol="F", contract="F260815P00010000", trigger_reason=trigger,
    )

    # Hit the HTMX partial route — same template the full page renders.
    resp = await client.get("/positions/_table", headers=_auth_header("wheelbot", "hunter2"))
    assert resp.status_code == 200
    body = resp.text
    assert "Close Trigger" in body, "header missing"
    assert trigger in body, f"trigger value {trigger!r} not rendered"
    # The CSS hook for future colour-coding must be present.
    assert f"trigger-{trigger}" in body, "trigger-<value> CSS hook missing"


@pytest.mark.asyncio
async def test_positions_table_renders_earnings_warning_badge(app_client, monkeypatch):
    """TICKET-006: when next_earnings returns a date inside the position's
    remaining DTE ± window, the row renders a ⚠ badge with the earnings date
    in the title attribute. Same predicate + same days_before/days_after as
    the recheck loop — single source of truth."""
    from data.earnings import EarningsLookup
    client, deps, broker = app_client

    # Open a CSP_OPEN with expiration 14 days from today (the dashboard uses
    # UTC.today as 'today').
    today = datetime.now(UTC).date()
    expiration = today + timedelta(days=14)
    occ = f"F{expiration.strftime('%y%m%d')}P00010000"
    cycle_id = await deps.repos.cycles.insert(
        WheelCycle(account_id="test", symbol="F", strategy_id="monthly_wheel",
                   started_at=datetime.now(UTC).replace(tzinfo=None))
    )
    await deps.repos.orders.insert(
        Order(
            account_id="test", symbol="F", strategy_id="monthly_wheel",
            cycle_id=cycle_id,
            order_type=OrderType.SELL_TO_OPEN, contract_symbol=occ,
            strike=10.0, expiration=expiration,
            option_type=__import__("core.models", fromlist=["OptionType"]).OptionType.PUT,
            quantity=1, fill_price=0.50, status=OrderStatus.FILLED,
            placed_at=datetime.now(UTC).replace(tzinfo=None),
            filled_at=datetime.now(UTC).replace(tzinfo=None),
        )
    )
    await deps.repos.positions.insert(
        Position(
            account_id="test", symbol="F", strategy_id="monthly_wheel",
            state=PositionState.CSP_OPEN, shares=0,
            current_cycle_id=cycle_id,
            state_changed_at=datetime.now(UTC).replace(tzinfo=None),
        )
    )
    broker.seed_quote(__import__("core.models", fromlist=["Quote"]).Quote(symbol=occ, bid=0.40, ask=0.42))

    # Earnings 10 days from today → inside (default 5/2 window around expiry).
    earnings_date = today + timedelta(days=10)
    def _stub_lookup(symbol: str, **_):
        if symbol == "F":
            return EarningsLookup("F", earnings_date, "finnhub")
        return EarningsLookup(symbol, None, "none")
    monkeypatch.setattr("dashboard.app._next_earnings", _stub_lookup)
    # Ensure the dashboard config carries the earnings_recheck window block —
    # the dashboard reads days_before/days_after from there.
    deps.config.setdefault("risk", {})["earnings_recheck"] = {
        "enabled": True, "check_interval_ticks": 12, "action": "flag_manual",
        "days_before": 5, "days_after": 2,
    }

    resp = await client.get("/positions/_table", headers=_auth_header("wheelbot", "hunter2"))
    assert resp.status_code == 200
    body = resp.text
    # Badge present + title attribute carries the earnings date.
    assert "warn-earnings" in body, "earnings-warning CSS class missing"
    assert "⚠" in body, "warning glyph missing"
    assert earnings_date.isoformat() in body, "earnings date not in title attr"


@pytest.mark.asyncio
async def test_positions_table_no_badge_when_earnings_outside_window(app_client, monkeypatch):
    """Negative case — earnings 60 days out → no badge, body has no
    warn-earnings class. Catches a future regression where the predicate
    always returns True."""
    from data.earnings import EarningsLookup
    client, deps, broker = app_client
    today = datetime.now(UTC).date()
    expiration = today + timedelta(days=14)
    occ = f"F{expiration.strftime('%y%m%d')}P00010000"
    cycle_id = await deps.repos.cycles.insert(
        WheelCycle(account_id="test", symbol="F", strategy_id="monthly_wheel",
                   started_at=datetime.now(UTC).replace(tzinfo=None))
    )
    await deps.repos.orders.insert(
        Order(
            account_id="test", symbol="F", strategy_id="monthly_wheel",
            cycle_id=cycle_id, order_type=OrderType.SELL_TO_OPEN,
            contract_symbol=occ, strike=10.0, expiration=expiration,
            option_type=__import__("core.models", fromlist=["OptionType"]).OptionType.PUT,
            quantity=1, fill_price=0.50, status=OrderStatus.FILLED,
            placed_at=datetime.now(UTC).replace(tzinfo=None),
            filled_at=datetime.now(UTC).replace(tzinfo=None),
        )
    )
    await deps.repos.positions.insert(
        Position(
            account_id="test", symbol="F", strategy_id="monthly_wheel",
            state=PositionState.CSP_OPEN, shares=0,
            current_cycle_id=cycle_id,
            state_changed_at=datetime.now(UTC).replace(tzinfo=None),
        )
    )
    broker.seed_quote(__import__("core.models", fromlist=["Quote"]).Quote(symbol=occ, bid=0.40, ask=0.42))

    def _stub_lookup(symbol: str, **_):
        return EarningsLookup(symbol, today + timedelta(days=60), "finnhub")
    monkeypatch.setattr("dashboard.app._next_earnings", _stub_lookup)
    deps.config.setdefault("risk", {})["earnings_recheck"] = {
        "days_before": 5, "days_after": 2,
    }

    resp = await client.get("/positions/_table", headers=_auth_header("wheelbot", "hunter2"))
    assert resp.status_code == 200
    assert "warn-earnings" not in resp.text


@pytest.mark.asyncio
async def test_macro_page_renders_empty_when_no_events(app_client):
    """Empty-state — page renders 200, shows "no upcoming events" copy."""
    client, deps, _broker = app_client
    deps.config.setdefault("risk", {})["macro_blackout"] = {
        "enabled": True, "event_types": ["FOMC", "CPI", "NFP"],
        "blackout_days_before": 1, "blackout_days_after": 0,
        "stale_threshold_hours": 48,
    }
    resp = await client.get("/macro", headers=_auth_header("wheelbot", "hunter2"))
    assert resp.status_code == 200
    assert "Macro Event Blackout" in resp.text
    assert "No upcoming events" in resp.text


@pytest.mark.asyncio
async def test_macro_page_renders_upcoming_events(app_client):
    """Seed two events; assert both render with their canonical types."""
    from core.models import MacroEvent
    client, deps, _broker = app_client
    deps.config.setdefault("risk", {})["macro_blackout"] = {
        "enabled": True, "event_types": ["FOMC", "CPI", "NFP"],
        "blackout_days_before": 1, "blackout_days_after": 0,
        "stale_threshold_hours": 48,
    }
    now = datetime.now(UTC).replace(tzinfo=None)
    today = now.date()
    await deps.repos.macro_events.upsert_many([
        MacroEvent(
            event_date=today + timedelta(days=5), event_type="FOMC", impact="high",
            description="FOMC Statement", fetched_at=now, created_at=now,
        ),
        MacroEvent(
            event_date=today + timedelta(days=12), event_type="CPI", impact="high",
            description="CPI YoY", fetched_at=now, created_at=now,
        ),
    ])
    resp = await client.get("/macro", headers=_auth_header("wheelbot", "hunter2"))
    assert resp.status_code == 200
    assert "FOMC" in resp.text
    assert "CPI" in resp.text
    assert "BLACKOUT" in resp.text   # the FOMC day appears in day_rows
    # Nav link present from base.html
    assert 'href="/macro"' in resp.text


@pytest.mark.asyncio
async def test_macro_nav_link_present_on_other_pages(app_client):
    """The Macro nav link added to base.html shows on every page."""
    client, _deps, _broker = app_client
    resp = await client.get("/", headers=_auth_header("wheelbot", "hunter2"))
    assert resp.status_code == 200
    assert 'href="/macro">Macro</a>' in resp.text


@pytest.mark.asyncio
async def test_positions_table_renders_dash_when_no_close_yet(app_client):
    """A position with no BUY_TO_CLOSE yet renders '—', not 'None'."""
    client, deps, _broker = app_client
    await _seed_csp_with_close(
        deps, symbol="F", contract="F260815P00010000", trigger_reason=None,
    )

    resp = await client.get("/positions/_table", headers=_auth_header("wheelbot", "hunter2"))
    assert resp.status_code == 200
    assert "Close Trigger" in resp.text
    # Templates should NEVER spit out "None" into the HTML.
    assert ">None<" not in resp.text


@pytest.mark.asyncio
async def test_positions_row_populates_dte_and_unrealized_for_bear_call_spread(app_client):
    """SPREAD_OPEN bear_call_spread also gets DTE + unrealized.

    Same code path as put_spread because both are MULTI_LEG_OPEN — this test
    locks in that the direction doesn't break the unrealized math.
    """
    from dashboard.app import _positions_rows, _QuoteCache
    _client, deps, broker = app_client
    expiration = datetime.now(UTC).date() + timedelta(days=30)
    await _seed_spread_position(
        deps, broker,
        symbol="IWM", strategy_id="bear_call_spread",
        short_strike=284.0, long_strike=289.0, option_type="CALL",
        fill_price=1.26,
        expiration=expiration,
        short_mid=0.80, long_mid=0.25,  # debit-to-close = 0.80 - 0.25 = 0.55
    )

    cache = _QuoteCache(ttl_seconds=60)
    rows = await _positions_rows(deps, cache)
    spread_rows = [r for r in rows if r["strategy_id"] == "bear_call_spread"]
    assert len(spread_rows) == 1
    row = spread_rows[0]
    assert row["dte"] == 30
    # original credit 1.26, current debit 0.55 → unrealized = (1.26 - 0.55) × 100 × 1 = 71
    assert row["unrealized"] == pytest.approx(71.0)
    # P&L % = 0.71 / 1.26 * 100 ≈ 56.3% captured
    assert row["unrealized_pct"] == pytest.approx(56.349, rel=1e-2)


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
    # TICKET-008: Drawdown Status section with tri-state badge.
    assert "Drawdown status" in resp.text
    # At least one strategy renders → NORMAL badge appears (default state).
    assert "NORMAL" in resp.text
    # TICKET-009: Win-rate Status section.
    assert "Win-rate status" in resp.text
    # Insufficient-data badge for fresh strategies (no cycles in test DB).
    assert "N/A (0/10)" in resp.text


@pytest.mark.asyncio
async def test_runbook_view_renders(app_client):
    """TICKET-023: GET /runbook returns 200 + the markdown rendered to HTML.

    Locks the route against silent breakage (renderer missing, file path
    moved, mistune uninstalled). Asserts known strings from the markdown
    that survive rendering — section headings always do."""
    client, _deps, _broker = app_client
    resp = await client.get("/runbook", headers=_auth_header("wheelbot", "hunter2"))
    assert resp.status_code == 200
    body = resp.text
    # H1 from the doc — confirms file was found and rendered.
    assert "WheelBot Go-Live Runbook" in body
    # H2 from §3 — confirms downstream rendering didn't truncate.
    assert "Stop conditions" in body
    # Confirms it went through a markdown renderer (mistune emits <h2>)
    # rather than being dumped as raw text.
    assert "<h2>" in body


@pytest.mark.asyncio
async def test_runbook_view_requires_auth(app_client):
    """Same authorize gate as the rest of the dashboard."""
    client, _deps, _broker = app_client
    resp = await client.get("/runbook")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_parity_view_empty(app_client):
    """TICKET-022: /parity renders the empty state when no parity_log rows
    exist (which is the case until the cron has run at least once)."""
    client, _deps, _broker = app_client
    resp = await client.get("/parity", headers=_auth_header("wheelbot", "hunter2"))
    assert resp.status_code == 200
    body = resp.text
    # Empty-state copy.
    assert "No parity data" in body or "no parity data" in body
    # Header still renders.
    assert "Broker pricing parity" in body


@pytest.mark.asyncio
async def test_cycles_view_includes_iron_condor_outcome_options(app_client):
    """TICKET-014: /cycles outcome dropdown is now driven from the
    CycleOutcome enum, so the three new IRON_CONDOR_* outcomes appear
    automatically without a template edit."""
    client, _deps, _broker = app_client
    resp = await client.get("/cycles", headers=_auth_header("wheelbot", "hunter2"))
    assert resp.status_code == 200
    body = resp.text
    assert "IRON_CONDOR_EXPIRED_PROFIT" in body
    assert "IRON_CONDOR_CLOSED_PROFIT" in body
    assert "IRON_CONDOR_CLOSED_LOSS" in body


@pytest.mark.asyncio
async def test_parity_view_with_data(app_client):
    """When parity_log has rows, the summary table and trend chart render."""
    client, deps, _broker = app_client
    now = datetime.now(UTC).replace(tzinfo=None)
    await deps.repos.broker_parity_log.insert_many([
        {
            "fetched_at": (now - timedelta(hours=1)).isoformat(),
            "symbol": "F",
            "contract_symbol": "F250620P00010000",
            "alpaca_mid": 1.0, "tasty_mid": 1.01, "mid_diff_pct": 1.0,
            "alpaca_bid": 0.99, "alpaca_ask": 1.01,
            "tasty_bid": 1.00, "tasty_ask": 1.02,
            "asymmetric_liquidity": 0,
        },
    ])
    resp = await client.get("/parity", headers=_auth_header("wheelbot", "hunter2"))
    assert resp.status_code == 200
    body = resp.text
    assert "F" in body
    assert "1.00%" in body  # avg matches
    assert "PASS" in body


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
