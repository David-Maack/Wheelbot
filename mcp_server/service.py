"""Service layer for the WheelBot Ops MCP.

Plain async class over Repos + Broker + config, returning JSON-serializable
dicts. `server.py` registers each method as an MCP tool; tests target this
layer directly (no transport needed).

Read tools never mutate. Control tools are gated by `controls_enabled` and
write an `mcp_control` checkpoint (triggered_by=MCP) for the audit trail. They
reuse the bot's existing paths: strategy_runtime (pause/reenable), the
`risk.stop_file_path` stop-file (kill switch), and `refresh_events` (macro).
"""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from core.checkpoint import log_checkpoint
from core.config import load_universe
from core.strategies import load_strategies, universe_for_strategy
from data import earnings as earnings_mod
from data.macro_calendar import refresh_events


class ControlsDisabled(RuntimeError):
    """Raised when a control tool is called but mcp.controls_enabled is false."""


def _iso(v: Any) -> Any:
    """JSON-safe: dates/datetimes -> isoformat, enums -> .value, else passthrough."""
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    if v is not None and not isinstance(v, (str, int, float, bool, list, dict)) and hasattr(v, "value"):
        return v.value
    return v


def _underlying(occ: str) -> str:
    """Underlying ticker from an OCC option symbol (root before the 6-digit date)."""
    m = re.match(r"^([A-Z]+)\d{6}[CP]\d+$", occ.strip())
    return m.group(1) if m else occ.strip()


class WheelbotMcpService:
    def __init__(self, repos: Any, broker: Any, config: dict[str, Any], *, controls_enabled: bool):
        self._repos = repos
        self._broker = broker
        self._config = config
        self._controls_enabled = controls_enabled
        self._account_id = config.get("account", {}).get("id", "primary")

    # ----------------------------------------------------------------- helpers
    def _require_controls(self, action: str) -> None:
        if not self._controls_enabled:
            raise ControlsDisabled(
                f"control '{action}' refused: mcp.controls_enabled is false"
            )

    def _audit(self, action: str, **ctx: Any) -> None:
        log_checkpoint("mcp_control", status="ok", action=action, triggered_by="MCP", **ctx)

    async def _open_by_strategy(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for p in await self._repos.positions.list_active(self._account_id):
            sid = p.strategy_id or "?"
            out[sid] = out.get(sid, 0) + 1
        return out

    @staticmethod
    def _runtime_summary(rt: dict | None) -> dict | None:
        if not rt:
            return None
        return {
            "drawdown_state": rt.get("drawdown_state"),
            "pause_state": rt.get("pause_state"),
            "disabled_until": rt.get("disabled_until"),
            "reason": rt.get("disabled_reason") or rt.get("paused_reason"),
        }

    # -------------------------------------------------------------- read tools
    async def get_positions(self) -> dict:
        """Open positions from the bot's database: state, strategy, cost basis, cycle."""
        rows = [
            {
                "symbol": p.symbol,
                "strategy_id": p.strategy_id,
                "state": _iso(p.state),
                "shares": p.shares,
                "cost_basis": p.cost_basis,
                "current_cycle_id": p.current_cycle_id,
            }
            for p in await self._repos.positions.list_active(self._account_id)
        ]
        return {"count": len(rows), "positions": rows}

    async def get_account_risk(self) -> dict:
        """Live account equity/cash/buying-power + concurrent-cap usage and net position mark."""
        acct = await self._broker.get_account()
        used = len(await self._repos.positions.list_active(self._account_id))
        cap = int(self._config.get("account", {}).get("max_concurrent_total", 0))
        return {
            "equity": acct.equity,
            "cash": acct.cash,
            "buying_power": acct.buying_power,
            "options_buying_power": acct.options_buying_power,
            "net_position_value": round(acct.equity - acct.cash, 2),
            "pattern_day_trader": acct.pattern_day_trader,
            "concurrent_cap": {"limit": cap, "used": used, "available": max(cap - used, 0)},
            "open_by_strategy": await self._open_by_strategy(),
        }

    async def get_strategy_status(self) -> dict:
        """Per-strategy: config enabled flag, runtime drawdown/pause state, open count."""
        open_counts = await self._open_by_strategy()
        rows = []
        for s in load_strategies(self._config):
            rt = await self._repos.strategy_runtime.get(s.id)
            rows.append({
                "id": s.id,
                "type": s.type,
                "enabled": s.enabled,
                "runtime": self._runtime_summary(rt),
                "open_positions": open_counts.get(s.id, 0),
            })
        return {"strategies": rows}

    async def get_performance(self, limit: int = 200) -> dict:
        """Realized P&L and win rate by strategy from closed cycles, plus the most recent closes."""
        by_strat: dict[str, dict] = {}
        recent: list[dict] = []
        for c in await self._repos.cycles.list_closed(self._account_id, limit=limit):
            sid = c.strategy_id or "?"
            agg = by_strat.setdefault(sid, {"closed": 0, "wins": 0, "pnl": 0.0})
            agg["closed"] += 1
            agg["pnl"] = round(agg["pnl"] + (c.final_pnl or 0.0), 2)
            if c.final_pnl is not None and c.final_pnl > 0:
                agg["wins"] += 1
            if len(recent) < 15:
                recent.append({
                    "symbol": c.symbol,
                    "strategy_id": sid,
                    "outcome": _iso(c.cycle_outcome),
                    "final_pnl": c.final_pnl,
                    "ended_at": _iso(c.ended_at),
                    "days_held": c.days_held,
                })
        for agg in by_strat.values():
            agg["win_rate"] = round(agg["wins"] / agg["closed"], 3) if agg["closed"] else None
        return {
            "by_strategy": by_strat,
            "total_realized_pnl": round(sum(a["pnl"] for a in by_strat.values()), 2),
            "recent_closed": recent,
        }

    async def get_recent_decisions(self, decision_type: str | None = None, limit: int = 20) -> dict:
        """Recent LLM decisions (screener / news_check) with decision, confidence, and cost."""
        decisions = await self._repos.llm_decisions.list_recent(decision_type=decision_type, limit=limit)
        cost_today = await self._repos.llm_decisions.total_cost_today()
        rows = [{
            "type": _iso(d.decision_type),
            "model": d.model,
            "decision": d.decision,
            "confidence": d.confidence,
            "cost_usd": d.cost_usd,
            "created_at": _iso(d.created_at),
            "symbol": d.context.get("symbol") if isinstance(d.context, dict) else None,
        } for d in decisions]
        return {"cost_today_usd": round(cost_today or 0.0, 4), "decisions": rows}

    async def get_regime_and_calendar(self, days: int = 14) -> dict:
        """Latest market-regime flags + upcoming macro events and calendar freshness."""
        snap = await self._repos.regime.latest()
        today = datetime.now(UTC).date()
        events = await self._repos.macro_events.between(today, today + timedelta(days=days))
        last_fetched = await self._repos.macro_events.last_fetched_at()
        regime = None
        if snap is not None:
            regime = {
                "date": _iso(snap.snapshot_date),
                "regime": _iso(snap.regime),
                "csps_allowed": snap.csps_allowed,
                "bear_calls_allowed": snap.bear_calls_allowed,
                "spy_above_sma": snap.spy_above_sma,
                "vix_close": snap.vix_close,
            }
        age_h = None
        if last_fetched is not None:
            age_h = round((datetime.now(UTC).replace(tzinfo=None) - last_fetched).total_seconds() / 3600, 1)
        return {
            "regime": regime,
            "macro_calendar_age_hours": age_h,
            "upcoming_events": [{
                "date": _iso(e.event_date),
                "type": e.event_type,
                "impact": e.impact,
                "description": e.description,
            } for e in events],
        }

    async def diagnose_symbol(self, symbol: str) -> dict:
        """Why a symbol may not be trading: which strategies own it, their enabled/runtime
        state, the regime gate, cap usage, next earnings, its open position + recent orders."""
        symbol = symbol.upper()
        universe = load_universe()
        traded_by = []
        for s in load_strategies(self._config):
            su = universe_for_strategy(s, universe)
            tickers = [getattr(t, "symbol", t) for t in su.get("tickers", [])]
            if symbol in tickers:
                rt = await self._repos.strategy_runtime.get(s.id)
                traded_by.append({
                    "strategy_id": s.id,
                    "type": s.type,
                    "enabled": s.enabled,
                    "runtime": self._runtime_summary(rt),
                })
        snap = await self._repos.regime.latest()
        regime = None
        if snap is not None:
            regime = {
                "regime": _iso(snap.regime),
                "csps_allowed": snap.csps_allowed,
                "bear_calls_allowed": snap.bear_calls_allowed,
            }
        cap = int(self._config.get("account", {}).get("max_concurrent_total", 0))
        used = len(await self._repos.positions.list_active(self._account_id))
        try:
            er = earnings_mod.next_earnings(symbol)
            next_earnings = _iso(er.next_date) if er.next_date else None
        except Exception:
            next_earnings = None
        pos = await self._repos.positions.get_by_symbol(self._account_id, symbol)
        recent = [o for o in await self._repos.orders.list_recent(self._account_id, limit=50)
                  if o.symbol == symbol][:8]
        return {
            "symbol": symbol,
            "traded_by": traded_by,
            "regime": regime,
            "concurrent_cap": {"limit": cap, "used": used, "full": used >= cap > 0},
            "next_earnings": next_earnings,
            "open_position": ({"state": _iso(pos.state), "strategy_id": pos.strategy_id}
                              if pos is not None else None),
            "recent_orders": [{
                "order_type": _iso(o.order_type),
                "status": _iso(o.status),
                "limit_price": o.limit_price,
                "strategy_id": o.strategy_id,
            } for o in recent],
            "note": ("Fine-grained per-tick skips (low-credit, entry-window) live in the bot "
                     "logs; this reports persisted gating state."),
        }

    async def get_watchlists(self) -> dict:
        """Currently-applied watchlist membership + the latest pending proposal
        (as an adds/drops diff with scores and rationales)."""
        ur = self._config.get("universe_refresh", {}) or {}
        applied = None
        run = await self._repos.watchlists.latest_run(status="applied")
        if run is not None and run.id is not None:
            by_strategy: dict[str, list[str]] = {}
            for e in await self._repos.watchlists.entries_for_run(run.id):
                if e.action != "drop":
                    by_strategy.setdefault(e.strategy_id, []).append(e.symbol)
            applied = {
                "run_id": run.id,
                "run_date": _iso(run.run_date),
                "applied_at": _iso(run.applied_at),
                "applied_by": run.applied_by,
                "watchlists": by_strategy,
            }

        proposal = None
        prop = await self._repos.watchlists.latest_run(status="proposed")
        if prop is not None and prop.id is not None:
            changes: dict[str, dict[str, list[dict]]] = {}
            keeps = 0
            for e in await self._repos.watchlists.entries_for_run(prop.id):
                if e.action == "keep":
                    keeps += 1
                    continue
                bucket = changes.setdefault(e.strategy_id, {"adds": [], "drops": []})
                bucket["adds" if e.action == "add" else "drops"].append({
                    "symbol": e.symbol, "score": e.score, "rationale": e.rationale,
                })
            proposal = {
                "run_id": prop.id,
                "run_date": _iso(prop.run_date),
                "summary": prop.summary,
                "cost_usd": prop.cost_usd,
                "changes": changes,
                "unchanged_keeps": keeps,
                "hint": "approve_watchlist(run_id, approve=true) to apply",
            }

        return {
            "refresh_enabled": bool(ur.get("enabled", False)),
            "auto_apply": bool(ur.get("auto_apply", False)),
            "applied": applied,
            "latest_proposal": proposal,
            "note": ("no watchlist run applied — strategies use universe.yaml membership"
                     if applied is None else
                     "strategies use the applied run's membership; universe.yaml is the fallback"),
        }

    # ----------------------------------------------------------- control tools
    async def approve_watchlist(self, run_id: int, approve: bool = True,
                                reason: str = "operator review via MCP") -> dict:
        """Apply (or reject) a PROPOSED universe-refresh run. Applying makes its
        membership live on the bot's next tick and supersedes the previous run."""
        self._require_controls("approve_watchlist")
        run = await self._repos.watchlists.get_run(run_id)
        if run is None:
            return {"ok": False, "error": f"no watchlist run {run_id}"}
        if str(run.status) != "proposed":
            return {"ok": False, "error": f"run {run_id} is '{_iso(run.status)}', not 'proposed'"}
        if approve:
            await self._repos.watchlists.apply_run(run_id, applied_by="mcp")
        else:
            await self._repos.watchlists.set_status(run_id, "rejected")
        self._audit("approve_watchlist", run_id=run_id, approve=approve, reason=reason)
        return {"ok": True, "run_id": run_id,
                "status": "applied" if approve else "rejected", "reason": reason}

    async def pause_strategy(self, strategy: str, reason: str = "operator pause via MCP") -> dict:
        """Pause NEW entries for a strategy (existing positions keep being managed)."""
        self._require_controls("pause_strategy")
        await self._repos.strategy_runtime.mark_paused(strategy, reason=f"MCP: {reason}")
        self._audit("pause_strategy", strategy=strategy, reason=reason)
        return {"ok": True, "strategy": strategy, "paused": True, "reason": reason}

    async def reenable_strategy(self, strategy: str, reason: str = "operator reenable via MCP") -> dict:
        """Clear a strategy's runtime drawdown/pause state so it can open again."""
        self._require_controls("reenable_strategy")
        prior = await self._repos.strategy_runtime.get(strategy)
        await self._repos.strategy_runtime.enable(strategy)
        self._audit("reenable_strategy", strategy=strategy, reason=reason,
                    prior_drawdown=(prior or {}).get("drawdown_state"),
                    prior_pause=(prior or {}).get("pause_state"))
        return {"ok": True, "strategy": strategy, "cleared": bool(prior), "reason": reason}

    def _stop_file(self) -> Path:
        p = self._config.get("risk", {}).get("stop_file_path")
        if not p:
            raise RuntimeError("risk.stop_file_path is not configured")
        return Path(p).expanduser()

    async def engage_kill_switch(self, reason: str = "operator stop via MCP") -> dict:
        """Engage the global kill switch (halts all new orders bot-wide via the stop file)."""
        self._require_controls("engage_kill_switch")
        sf = self._stop_file()
        sf.parent.mkdir(parents=True, exist_ok=True)
        sf.write_text(f"MCP kill switch engaged: {reason}\n")
        self._audit("engage_kill_switch", reason=reason, stop_file=str(sf))
        return {"ok": True, "kill_switch": "ENGAGED", "stop_file": str(sf), "reason": reason}

    async def release_kill_switch(self) -> dict:
        """Release the global kill switch (remove the stop file)."""
        self._require_controls("release_kill_switch")
        sf = self._stop_file()
        existed = sf.exists()
        if existed:
            sf.unlink()
        self._audit("release_kill_switch", stop_file=str(sf), existed=existed)
        return {"ok": True, "kill_switch": "RELEASED", "stop_file": str(sf), "was_engaged": existed}

    async def refresh_macro_calendar(self) -> dict:
        """Refresh the macro-event calendar now (idempotent; clears the stale alert)."""
        self._require_controls("refresh_macro_calendar")
        rows, source = await refresh_events(self._repos, self._config)
        self._audit("refresh_macro_calendar", rows=rows, source=source)
        return {"ok": True, "distinct_rows": rows, "source": source}

    async def flatten_position(self, symbol: str, execute: bool = False) -> dict:
        """Close all option legs for one underlying. DRY-RUN by default — returns the plan;
        only acts when execute=true. Closes at the broker, then marks the DB row IDLE."""
        self._require_controls("flatten_position")
        symbol = symbol.upper()
        tc = getattr(self._broker, "_trading", None)
        if tc is None:
            return {"ok": False, "error": "flatten via MCP currently supports the Alpaca broker only"}
        raw = await asyncio.to_thread(tc.get_all_positions)
        legs = [p for p in raw if _underlying(p.symbol) == symbol]
        plan = [{"symbol": p.symbol, "qty": p.qty, "market_value": p.market_value,
                 "unrealized_pl": p.unrealized_pl} for p in legs]
        if not legs:
            return {"ok": True, "symbol": symbol, "executed": False, "plan": [],
                    "note": "no broker legs for this symbol"}
        if not execute:
            return {"ok": True, "symbol": symbol, "executed": False, "dry_run": True,
                    "would_close": plan, "hint": "call again with execute=true to close for real"}
        errors: list[str] = []
        for p in sorted(legs, key=lambda x: int(float(x.qty))):  # shorts (negative qty) first
            try:
                await asyncio.to_thread(tc.close_position, p.symbol)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{p.symbol}: {exc!r}")
        pos = await self._repos.positions.get_by_symbol(self._account_id, symbol)
        if pos is not None and pos.id is not None:
            await self._repos.positions.update(pos.id, state="IDLE", current_cycle_id=None)
        self._audit("flatten_position", symbol=symbol, legs=len(legs), errors=errors)
        return {"ok": not errors, "symbol": symbol, "executed": True, "closed": plan, "errors": errors}
