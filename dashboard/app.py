"""WheelBot dashboard — FastAPI on :8889, server-side HTMX.

Per spec §11. Five views + healthz + manual-stop POST.

Auth: HTTP Basic. Username from `config.dashboard.basic_auth_user`, password
from `WHEELBOT_DASHBOARD_PASSWORD`. Bind 127.0.0.1; reach via Tailscale or
SSH tunnel — never internet-exposed.

Construction: callers pass a `DashboardDeps` so the app is testable against
a tmp-DB / PaperBroker without hitting real services. A factory
`build_app(deps)` returns the FastAPI instance. The module-level `app`
singleton lazily builds one from `load_config()` on first request.
"""

from __future__ import annotations

import asyncio
import os
import secrets
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from core.broker import Broker, BrokerUnavailable
from core.broker_factory import make_broker
from core.config import load_config
from core.models import Order, OrderStatus, OrderType, Position, PositionState, WheelCycle
from db.repo import Database, Repos

ROOT = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(ROOT / "templates"))
STATIC_DIR = ROOT / "static"
basic = HTTPBasic()


@dataclass
class DashboardDeps:
    repos: Repos
    broker: Broker
    config: dict[str, Any]
    auth_user: str
    auth_password: str | None
    quote_cache_seconds: float = 30.0


# --- Quote cache (per-app) ----------------------------------------------------

class _QuoteCache:
    def __init__(self, ttl_seconds: float):
        self._ttl = ttl_seconds
        self._data: dict[str, tuple[float, float | None]] = {}
        self._lock = asyncio.Lock()

    async def get(self, broker: Broker, symbol: str) -> float | None:
        async with self._lock:
            cached = self._data.get(symbol)
            now = time.monotonic()
            if cached and now - cached[0] < self._ttl:
                return cached[1]
        try:
            quote = await broker.get_quote(symbol)
            mid = quote.mid if quote.mid is not None else (quote.last or quote.bid or quote.ask)
        except BrokerUnavailable:
            mid = None
        except Exception:
            mid = None
        async with self._lock:
            self._data[symbol] = (time.monotonic(), mid)
        return mid


# --- App factory --------------------------------------------------------------

def build_app(deps: DashboardDeps) -> FastAPI:
    app = FastAPI(title="WheelBot")
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    cache = _QuoteCache(deps.quote_cache_seconds)

    def authorize(creds: HTTPBasicCredentials = Depends(basic)) -> str:
        expected_user = deps.auth_user
        expected_pw = deps.auth_password
        if not expected_pw:
            # Hard fail so we don't accidentally serve unauthenticated.
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="WHEELBOT_DASHBOARD_PASSWORD not set",
            )
        ok_user = secrets.compare_digest(creds.username, expected_user)
        ok_pw = secrets.compare_digest(creds.password, expected_pw)
        if not (ok_user and ok_pw):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="bad credentials",
                headers={"WWW-Authenticate": "Basic"},
            )
        return creds.username

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        try:
            await deps.broker.get_account()
            broker_ok = True
        except Exception:
            broker_ok = False
        account_id = deps.config.get("account", {}).get("id", "primary")
        anchor = await deps.repos.daily_state.get(
            account_id, datetime.now(UTC).date()
        )
        return {
            "status": "ok" if broker_ok else "degraded",
            "broker_connected": broker_ok,
            "kill_switch_armed": bool(anchor.kill_switch_armed) if anchor else False,
            "kill_switch_reason": anchor.kill_switch_reason if anchor else None,
        }

    @app.get("/", response_class=HTMLResponse)
    async def positions_view(request: Request, _user: str = Depends(authorize)) -> Any:
        rows = await _positions_rows(deps, cache)
        ks_state = await _kill_switch_state(deps)
        return TEMPLATES.TemplateResponse(
            request,
            "positions.html",
            {"rows": rows, "kill_switch": ks_state},
        )

    @app.get("/positions/_table", response_class=HTMLResponse)
    async def positions_partial(request: Request, _user: str = Depends(authorize)) -> Any:
        rows = await _positions_rows(deps, cache)
        return TEMPLATES.TemplateResponse(
            request, "_positions_table.html", {"rows": rows}
        )

    @app.get("/cycles", response_class=HTMLResponse)
    async def cycles_view(
        request: Request,
        symbol: str | None = None,
        outcome: str | None = None,
        strategy: str | None = None,
        _user: str = Depends(authorize),
    ) -> Any:
        account_id = deps.config.get("account", {}).get("id", "primary")
        opens = await deps.repos.cycles.list_open(account_id)
        closed = await deps.repos.cycles.list_closed(account_id, limit=200)
        cycles = list(opens) + list(closed)
        if symbol:
            cycles = [c for c in cycles if c.symbol.upper() == symbol.upper()]
        if outcome:
            cycles = [c for c in cycles if (c.cycle_outcome or "").upper() == outcome.upper()]
        if strategy:
            cycles = [c for c in cycles if (c.strategy_id or "") == strategy]
        cycles.sort(key=lambda c: (c.final_pnl is None, -(c.final_pnl or 0)))
        return TEMPLATES.TemplateResponse(
            request,
            "cycles.html",
            {
                "cycles": cycles,
                "filter_symbol": symbol or "",
                "filter_outcome": outcome or "",
                "filter_strategy": strategy or "",
                "strategies": _known_strategies(deps),
            },
        )

    @app.get("/candidates", response_class=HTMLResponse)
    async def candidates_view(request: Request, _user: str = Depends(authorize)) -> Any:
        c = await deps.repos.db.connect()
        async with c.execute(
            "SELECT * FROM candidates ORDER BY run_date DESC, rank ASC LIMIT 50"
        ) as cur:
            rows = await cur.fetchall()
        candidates = [dict(r) for r in rows]
        return TEMPLATES.TemplateResponse(
            request, "candidates.html", {"candidates": candidates}
        )

    @app.get("/orders", response_class=HTMLResponse)
    async def orders_view(
        request: Request,
        strategy: str | None = None,
        _user: str = Depends(authorize),
    ) -> Any:
        account_id = deps.config.get("account", {}).get("id", "primary")
        recent = await deps.repos.orders.list_recent(account_id, limit=100)
        if strategy:
            recent = [o for o in recent if (o.strategy_id or "") == strategy]
        return TEMPLATES.TemplateResponse(
            request,
            "orders.html",
            {
                "orders": recent,
                "filter_strategy": strategy or "",
                "strategies": _known_strategies(deps),
            },
        )

    @app.get("/decisions", response_class=HTMLResponse)
    async def decisions_view(
        request: Request,
        decision_type: str | None = None,
        decision: str | None = None,
        _user: str = Depends(authorize),
    ) -> Any:
        """Recent LLM decisions — screener, news_check, roll advisor."""
        c = await deps.repos.db.connect()
        sql = "SELECT * FROM llm_decisions"
        params: list[Any] = []
        wheres: list[str] = []
        if decision_type:
            wheres.append("decision_type = ?")
            params.append(decision_type)
        if decision:
            wheres.append("LOWER(decision) = LOWER(?)")
            params.append(decision)
        if wheres:
            sql += " WHERE " + " AND ".join(wheres)
        sql += " ORDER BY created_at DESC LIMIT 200"
        async with c.execute(sql, tuple(params)) as cur:
            rows = await cur.fetchall()
        decisions = [dict(r) for r in rows]

        async with c.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM llm_decisions "
            "WHERE date(created_at) = date('now')"
        ) as cur:
            spent_today = float((await cur.fetchone())[0])
        budget_cap = float(
            deps.config.get("intelligence", {}).get("daily_budget_usd", 1.0)
        )
        return TEMPLATES.TemplateResponse(
            request,
            "decisions.html",
            {
                "decisions": decisions,
                "filter_type": decision_type or "",
                "filter_decision": decision or "",
                "spent_today": spent_today,
                "budget_cap": budget_cap,
            },
        )

    @app.get("/performance", response_class=HTMLResponse)
    async def performance_view(request: Request, _user: str = Depends(authorize)) -> Any:
        data = await _performance_data(deps)
        return TEMPLATES.TemplateResponse(
            request,
            "performance.html",
            {"data": data},
        )

    @app.get("/risk", response_class=HTMLResponse)
    async def risk_view(request: Request, _user: str = Depends(authorize)) -> Any:
        account_id = deps.config.get("account", {}).get("id", "primary")
        try:
            account = await deps.broker.get_account()
        except Exception:
            account = None
        active = await deps.repos.positions.list_active(account_id)
        ks_state = await _kill_switch_state(deps)
        regime_row = await _latest_regime_row(deps)
        bp_used_pct = None
        if account and account.equity:
            bp_used_pct = (1 - account.buying_power / account.equity) * 100.0
        return TEMPLATES.TemplateResponse(
            request,
            "risk.html",
            {
                "account": account,
                "bp_used_pct": bp_used_pct,
                "active_positions": active,
                "kill_switch": ks_state,
                "regime": regime_row,
                "stop_file": deps.config.get("risk", {}).get("stop_file_path"),
            },
        )

    @app.post("/risk/manual_stop")
    async def manual_stop(
        action: str = Form(...),
        _user: str = Depends(authorize),
    ) -> Any:
        path = deps.config.get("risk", {}).get("stop_file_path")
        if not path:
            raise HTTPException(400, "risk.stop_file_path not configured")
        target = Path(path).expanduser()
        if action == "engage":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f"engaged via dashboard at {datetime.now(UTC).isoformat()}\n")
        elif action == "release":
            if target.exists():
                target.unlink()
        else:
            raise HTTPException(400, f"unknown action {action!r}")
        return RedirectResponse(url="/risk", status_code=status.HTTP_303_SEE_OTHER)

    return app


# --- View helpers -------------------------------------------------------------

async def _positions_rows(deps: DashboardDeps, cache: _QuoteCache) -> list[dict[str, Any]]:
    account_id = deps.config.get("account", {}).get("id", "primary")
    positions = await deps.repos.positions.list_all(account_id)
    out: list[dict[str, Any]] = []
    today = datetime.now(UTC).replace(tzinfo=None)
    for p in positions:
        if p.state == PositionState.IDLE:
            continue
        days_in_state = max((today - p.state_changed_at).days, 0)
        dte: int | None = None
        unrealized: float | None = None
        if p.current_cycle_id:
            cycle = await deps.repos.cycles.get(p.current_cycle_id)
            if cycle:
                # Find an open option order for this position to compute DTE / unrealized.
                latest_open_option = await _latest_open_option_for_cycle(deps, cycle.id)
                if latest_open_option and latest_open_option.expiration:
                    dte = (latest_open_option.expiration - today.date()).days
                if latest_open_option and latest_open_option.contract_symbol:
                    mid = await cache.get(deps.broker, latest_open_option.contract_symbol)
                    if mid is not None and latest_open_option.fill_price is not None:
                        # short option: we collected fill_price; current cost-to-close is mid.
                        # premium per share × 100 × qty.
                        unrealized = (
                            (latest_open_option.fill_price - mid)
                            * 100
                            * latest_open_option.quantity
                        )
        out.append(
            {
                "symbol": p.symbol,
                "strategy_id": p.strategy_id,
                "state": str(p.state),
                "days_in_state": days_in_state,
                "shares": p.shares,
                "cost_basis": p.cost_basis,
                "dte": dte,
                "unrealized": unrealized,
            }
        )
    return out


async def _latest_open_option_for_cycle(deps: DashboardDeps, cycle_id: int | None) -> Order | None:
    if cycle_id is None:
        return None
    c = await deps.repos.db.connect()
    async with c.execute(
        "SELECT * FROM orders WHERE cycle_id = ? "
        "AND order_type IN (?, ?) "
        "AND status IN (?, ?) "
        "ORDER BY placed_at DESC LIMIT 1",
        (
            cycle_id,
            OrderType.SELL_TO_OPEN.value,
            OrderType.SELL_TO_OPEN.value,
            OrderStatus.FILLED.value,
            OrderStatus.PARTIAL.value,
        ),
    ) as cur:
        row = await cur.fetchone()
    if row is None:
        return None
    from db.repo import _row_to_dict, JSON_FIELDS_BY_TABLE

    return Order(**_row_to_dict(row, JSON_FIELDS_BY_TABLE["orders"]))


async def _kill_switch_state(deps: DashboardDeps) -> dict[str, Any]:
    account_id = deps.config.get("account", {}).get("id", "primary")
    today = datetime.now(UTC).date()
    anchor = await deps.repos.daily_state.get(account_id, today)
    stop_path = deps.config.get("risk", {}).get("stop_file_path")
    stop_file_present = bool(stop_path and Path(stop_path).expanduser().exists())
    return {
        "armed": bool(anchor.kill_switch_armed) if anchor else False,
        "reason": anchor.kill_switch_reason if anchor else None,
        "stop_file_present": stop_file_present,
        "session_open_equity": anchor.session_open_equity if anchor else None,
    }


def _known_strategies(deps: DashboardDeps) -> list[dict[str, str]]:
    """Strategies block from config — list of {id, display_name} for the
    filter dropdown. Falls back to a single monthly_wheel entry for legacy
    config without a strategies section."""
    raw = deps.config.get("strategies") or []
    if not raw:
        return [{"id": "monthly_wheel", "display_name": "Monthly Wheel"}]
    out: list[dict[str, str]] = []
    for s in raw:
        if not isinstance(s, dict):
            continue
        sid = str(s.get("id", "")).strip()
        if not sid:
            continue
        out.append({"id": sid, "display_name": str(s.get("display_name", sid))})
    return out


async def _latest_regime_row(deps: DashboardDeps) -> dict[str, Any] | None:
    c = await deps.repos.db.connect()
    async with c.execute(
        "SELECT * FROM regime_snapshots ORDER BY snapshot_date DESC LIMIT 1"
    ) as cur:
        row = await cur.fetchone()
    return dict(row) if row else None


async def _performance_data(deps: DashboardDeps) -> dict[str, Any]:
    """Aggregate stats + time series for the /performance view.

    All numbers come from wheel_cycles. The cumulative P&L list is sorted by
    cycle close time so Chart.js can plot it as a single line.
    """
    account_id = deps.config.get("account", {}).get("id", "primary")
    c = await deps.repos.db.connect()

    async with c.execute(
        "SELECT id, symbol, started_at, ended_at, final_pnl, "
        "       cycle_outcome, days_held "
        "FROM wheel_cycles "
        "WHERE account_id = ? AND ended_at IS NOT NULL AND final_pnl IS NOT NULL "
        "ORDER BY ended_at",
        (account_id,),
    ) as cur:
        closed_rows = await cur.fetchall()
    closed = [dict(r) for r in closed_rows]

    async with c.execute(
        "SELECT COUNT(*) FROM wheel_cycles "
        "WHERE account_id = ? AND ended_at IS NULL",
        (account_id,),
    ) as cur:
        open_count = (await cur.fetchone())[0]

    cumulative: list[dict[str, Any]] = []
    running = 0.0
    for row in closed:
        running += float(row["final_pnl"] or 0)
        cumulative.append(
            {
                "ended_at": row["ended_at"],
                "symbol": row["symbol"],
                "pnl": float(row["final_pnl"] or 0),
                "running": running,
                "outcome": row["cycle_outcome"],
            }
        )

    wins = sum(1 for r in closed if (r["final_pnl"] or 0) > 0)
    losses = sum(1 for r in closed if (r["final_pnl"] or 0) < 0)
    total_pnl = sum(float(r["final_pnl"] or 0) for r in closed)
    win_rate = (wins / len(closed) * 100.0) if closed else None

    days_held = [int(r["days_held"]) for r in closed if r["days_held"] is not None]
    avg_days = (sum(days_held) / len(days_held)) if days_held else None

    outcomes: dict[str, int] = {}
    for r in closed:
        key = r["cycle_outcome"] or "UNKNOWN"
        outcomes[key] = outcomes.get(key, 0) + 1

    return {
        "stats": {
            "total_realized_pnl_usd": total_pnl,
            "wins": wins,
            "losses": losses,
            "n_closed": len(closed),
            "win_rate_pct": win_rate,
            "open_cycles": int(open_count),
            "avg_days_held": avg_days,
        },
        "cumulative": cumulative,
        "outcomes": outcomes,
    }


# --- Production factory -------------------------------------------------------
#
# Run with:
#     uvicorn dashboard.app:create_app --factory --host 127.0.0.1 --port 8889
#
# The factory pattern keeps importing this module side-effect-free (no broker
# construction at import time), so unit tests can `build_app(deps)` against a
# tmp-DB + PaperBroker without touching production env vars.

def create_app() -> FastAPI:
    config = load_config()
    db_path = Path(config.get("database", {}).get("path", "wheelbot.db")).expanduser()
    db = Database(db_path)
    repos = Repos(db)
    broker = make_broker(config)
    deps = DashboardDeps(
        repos=repos,
        broker=broker,
        config=config,
        auth_user=config.get("dashboard", {}).get("basic_auth_user", "wheelbot"),
        auth_password=os.environ.get("WHEELBOT_DASHBOARD_PASSWORD"),
    )
    return build_app(deps)
