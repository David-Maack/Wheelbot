"""FastMCP server: exposes WheelbotMcpService as MCP tools over authenticated HTTP.

Read tools are always available; control tools (marked [control]) additionally
require `mcp.controls_enabled` (checked inside the service). Every request must
carry `Authorization: Bearer <WHEELBOT_MCP_TOKEN>` — the token is the primary
guard, so the server refuses to start without one (see scripts/run_mcp.py).
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from mcp_server.service import WheelbotMcpService


def build_server(service: WheelbotMcpService, *, host: str, port: int) -> FastMCP:
    mcp = FastMCP("wheelbot", host=host, port=port)

    # ---- read tools ----
    @mcp.tool()
    async def get_positions() -> dict:
        """Open positions from the bot's DB: state, strategy, cost basis, cycle."""
        return await service.get_positions()

    @mcp.tool()
    async def get_account_risk() -> dict:
        """Live account equity/cash/buying-power, concurrent-cap usage, net position mark."""
        return await service.get_account_risk()

    @mcp.tool()
    async def get_strategy_status() -> dict:
        """Per-strategy: config enabled flag, runtime drawdown/pause state, open-position count."""
        return await service.get_strategy_status()

    @mcp.tool()
    async def get_performance(limit: int = 200) -> dict:
        """Realized P&L + win rate by strategy from closed cycles, plus the most recent closes."""
        return await service.get_performance(limit=limit)

    @mcp.tool()
    async def get_recent_decisions(decision_type: str | None = None, limit: int = 20) -> dict:
        """Recent screener/news_check LLM decisions with decision, confidence, and cost."""
        return await service.get_recent_decisions(decision_type=decision_type, limit=limit)

    @mcp.tool()
    async def get_regime_and_calendar(days: int = 14) -> dict:
        """Market regime flags + upcoming macro events and macro-calendar freshness."""
        return await service.get_regime_and_calendar(days=days)

    @mcp.tool()
    async def diagnose_symbol(symbol: str) -> dict:
        """Why a symbol may not be trading: owning strategies, regime gate, cap, earnings, recent orders."""
        return await service.diagnose_symbol(symbol)

    @mcp.tool()
    async def get_watchlists() -> dict:
        """Applied universe-refresh watchlists per strategy + latest pending proposal diff."""
        return await service.get_watchlists()

    # ---- guarded control tools (require mcp.controls_enabled) ----
    @mcp.tool()
    async def approve_watchlist(run_id: int, approve: bool = True,
                                reason: str = "operator review via MCP") -> dict:
        """[control] Apply (approve=true) or reject a PROPOSED universe-refresh watchlist run."""
        return await service.approve_watchlist(run_id, approve=approve, reason=reason)

    @mcp.tool()
    async def pause_strategy(strategy: str, reason: str = "operator pause via MCP") -> dict:
        """[control] Pause NEW entries for a strategy; existing positions stay managed."""
        return await service.pause_strategy(strategy, reason=reason)

    @mcp.tool()
    async def reenable_strategy(strategy: str, reason: str = "operator reenable via MCP") -> dict:
        """[control] Clear a strategy's runtime drawdown/pause state so it can open again."""
        return await service.reenable_strategy(strategy, reason=reason)

    @mcp.tool()
    async def engage_kill_switch(reason: str = "operator stop via MCP") -> dict:
        """[control] Engage the global kill switch — halts ALL new orders bot-wide."""
        return await service.engage_kill_switch(reason=reason)

    @mcp.tool()
    async def release_kill_switch() -> dict:
        """[control] Release the global kill switch."""
        return await service.release_kill_switch()

    @mcp.tool()
    async def refresh_macro_calendar() -> dict:
        """[control] Refresh the macro-event calendar now (idempotent)."""
        return await service.refresh_macro_calendar()

    @mcp.tool()
    async def flatten_position(symbol: str, execute: bool = False) -> dict:
        """[control] Close all option legs for one underlying. DRY-RUN unless execute=true."""
        return await service.flatten_position(symbol, execute=execute)

    return mcp


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Reject any request without `Authorization: Bearer <token>`."""

    def __init__(self, app: Any, token: str) -> None:
        super().__init__(app)
        self._expected = f"Bearer {token}"

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        if request.headers.get("authorization") != self._expected:
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)


def run_http(mcp: FastMCP, *, token: str) -> None:
    """Serve the MCP over streamable HTTP with bearer-token auth (blocks)."""
    import uvicorn

    app = mcp.streamable_http_app()
    app.add_middleware(BearerAuthMiddleware, token=token)
    uvicorn.run(app, host=mcp.settings.host, port=mcp.settings.port, log_level="info")
