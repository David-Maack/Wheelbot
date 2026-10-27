"""Per-strategy drawdown circuit breaker (Sprint 13 sub-sprint 1).

For each strategy, sums realized P&L from cycles closed in the last 7 days.
Two states:

    NORMAL    — rolling P&L above `auto_disable_drawdown_usd`.
    DISABLED  — rolling P&L at or below it. New entries skipped (closes still
                run); 30-day auto-reset, or manual reenable.

The TICKET-008 WARNING tier (half size at `drawdown_warning_usd`) was removed
2026-10-27: both strategies trade one contract/spread, so halving was a no-op
that only added a state. The account-level kill switch (execution/
kill_switch.py) and this breaker are the two automatic circuits; the
scorecard (risk/scorecard.py) is the human's go/no-go rule.

Wired into scripts/run_bot.py: at the top of each strategy iteration the loop
reads `current_state(...)` (DISABLED → no opens); after the iteration
`check_and_apply(...)` updates the state from the latest rolling P&L.

    NORMAL   → DISABLED  fires `strategy_auto_disabled`
    DISABLED → NORMAL    only via the 30-day auto-reset OR manual reenable —
                         P&L recovery does NOT re-arm, to avoid thrashing.

Disable is "soft" — config.yaml is unchanged. Runtime state lives in the
`strategy_runtime_state` table.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from core.checkpoint import log_checkpoint
from core.notify import notify
from core.strategies import StrategyDefinition
from db.repo import Repos

AUTO_RESET_DAYS = 30
ROLLING_WINDOW_DAYS = 7


class DrawdownState(StrEnum):
    NORMAL = "NORMAL"
    DISABLED = "DISABLED"


async def compute_rolling_pnl(
    repos: Repos,
    strategy_id: str,
    *,
    account_id: str = "primary",
    days: int = ROLLING_WINDOW_DAYS,
    now: datetime | None = None,
) -> float:
    """Sum final_pnl across cycles closed in the last `days` days.

    Only counts cycles where ended_at is not null AND final_pnl is not null.
    Open cycles (unrealized) are excluded — would introduce tick-to-tick
    noise into the metric.
    """
    now = now or datetime.now(UTC).replace(tzinfo=None)
    cutoff = (now - timedelta(days=days)).isoformat()
    c = await repos.db.connect()
    async with c.execute(
        "SELECT COALESCE(SUM(final_pnl), 0) AS pnl FROM wheel_cycles "
        "WHERE account_id = ? AND strategy_id = ? "
        "AND ended_at IS NOT NULL AND final_pnl IS NOT NULL "
        "AND ended_at >= ?",
        (account_id, strategy_id, cutoff),
    ) as cur:
        row = await cur.fetchone()
    return float(row["pnl"]) if row and row["pnl"] is not None else 0.0


async def current_state(
    repos: Repos,
    strategy_id: str,
    *,
    now: datetime | None = None,
) -> DrawdownState:
    """Read the persisted drawdown state for a strategy. Auto-clears stale
    DISABLED rows (disabled_until in the past)."""
    now = now or datetime.now(UTC).replace(tzinfo=None)
    row = await repos.strategy_runtime.get(strategy_id)
    if row is None:
        return DrawdownState.NORMAL
    disabled_until_str = row.get("disabled_until")
    if disabled_until_str:
        until = datetime.fromisoformat(disabled_until_str)
        if until > now:
            return DrawdownState.DISABLED
        await repos.strategy_runtime.enable(strategy_id)
    return DrawdownState.NORMAL


async def check_and_apply(
    repos: Repos,
    strategy: StrategyDefinition,
    config: dict[str, Any],
    *,
    now: datetime | None = None,
) -> DrawdownState:
    """Evaluate the threshold and update state. Returns the NEW state.
    No-op (NORMAL) when the strategy has no / a zero
    `auto_disable_drawdown_usd`."""
    disable_thresh = strategy.params.get("auto_disable_drawdown_usd")
    if disable_thresh is None or float(disable_thresh) == 0:
        return DrawdownState.NORMAL
    disable_v = float(disable_thresh)

    now = now or datetime.now(UTC).replace(tzinfo=None)
    account_id = config.get("account", {}).get("id", "primary")
    pnl = await compute_rolling_pnl(repos, strategy.id, account_id=account_id, now=now)

    prior = await current_state(repos, strategy.id, now=now)
    if prior is DrawdownState.DISABLED:
        log_checkpoint(
            "auto_disable_check", status="ok", strategy=strategy.id,
            rolling_pnl=pnl, state=prior.value,
            note="DISABLED held — recovery only via 30d auto-reset or manual",
        )
        return DrawdownState.DISABLED

    if pnl > disable_v:
        log_checkpoint(
            "auto_disable_check", status="ok", strategy=strategy.id,
            rolling_pnl=pnl, threshold=disable_v, triggered=False,
        )
        return DrawdownState.NORMAL

    until = now + timedelta(days=AUTO_RESET_DAYS)
    reason = (
        f"rolling {ROLLING_WINDOW_DAYS}d realized P&L ${pnl:.2f} "
        f"≤ disable threshold ${disable_v:.2f}"
    )
    await repos.strategy_runtime.disable(strategy.id, until=until, reason=reason, now=now)
    await repos.strategy_runtime.set_drawdown_state(strategy.id, DrawdownState.DISABLED.value)
    log_checkpoint(
        "auto_disable_triggered", status="ok", strategy=strategy.id,
        rolling_pnl=pnl, threshold=disable_v, disabled_until=until.isoformat(),
    )
    await notify(
        "strategy_auto_disabled",
        f"Auto-disabled strategy: {strategy.id}",
        strategy=strategy.id,
        rolling_pnl=pnl,
        threshold=disable_v,
        disabled_until=until.isoformat(),
        reason=reason,
    )
    return DrawdownState.DISABLED


async def get_drawdown_overview(
    repos: Repos,
    strategies: list[StrategyDefinition],
    config: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, dict[str, Any]]:
    """Dashboard-ready per-strategy overview:
    `{strategy_id: {state, rolling_pnl, disable_threshold, disabled_until}}`."""
    now = now or datetime.now(UTC).replace(tzinfo=None)
    account_id = config.get("account", {}).get("id", "primary")
    out: dict[str, dict[str, Any]] = {}
    for s in strategies:
        state = await current_state(repos, s.id, now=now)
        pnl = await compute_rolling_pnl(repos, s.id, account_id=account_id, now=now)
        row = await repos.strategy_runtime.get(s.id) or {}
        out[s.id] = {
            "state": state.value,
            "rolling_pnl": pnl,
            "disable_threshold": s.params.get("auto_disable_drawdown_usd"),
            "disabled_until": row.get("disabled_until"),
        }
    return out
