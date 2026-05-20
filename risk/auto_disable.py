"""Sprint 13 sub-sprint 1 — auto-disable on drawdown.

Defensive circuit breaker. For each strategy, sums realized P&L from
cycles closed in the last 7 days. If the rolling sum is more negative
than `auto_disable_drawdown_usd` (configured per strategy), disables the
strategy at runtime for 30 days and fires a Discord notification.

Disable is "soft" — config.yaml is unchanged. Runtime state lives in the
`strategy_runtime_state` table. Auto-resets after 30 days as a safety
net so a one-bad-week event from months ago doesn't keep a strategy
permanently parked. Operator can manually re-enable any time via
`scripts/reenable_strategy.py` or the dashboard.

Wired into scripts/run_bot.py: at the top of each strategy iteration,
skip if currently disabled; after the iteration completes, evaluate
the threshold (so today's just-closed cycles count toward tomorrow's
check).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from core.checkpoint import log_checkpoint
from core.notify import notify
from core.strategies import StrategyDefinition
from db.repo import Repos

# Auto-reset window — disabled strategies come back to life after this
# many days even without manual intervention. Long enough to force a
# real review; short enough to avoid permanent state drift.
AUTO_RESET_DAYS = 30
# Rolling window for the drawdown metric.
ROLLING_WINDOW_DAYS = 7


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


async def check_and_apply(
    repos: Repos,
    strategy: StrategyDefinition,
    config: dict[str, Any],
    *,
    now: datetime | None = None,
) -> bool:
    """Evaluate the drawdown threshold and disable if exceeded.

    Returns True if the strategy was newly disabled this call (so callers
    can decide whether to short-circuit further work for it on this tick).
    No-op when:
      - the strategy has no `auto_disable_drawdown_usd` param
      - the threshold is 0 (means "disabled feature")
      - the strategy is already disabled (don't re-notify or extend the timer)
    """
    threshold_raw = strategy.params.get("auto_disable_drawdown_usd")
    if threshold_raw is None:
        return False
    threshold = float(threshold_raw)
    if threshold == 0:
        return False  # feature disabled for this strategy

    now = now or datetime.now(UTC).replace(tzinfo=None)
    already_disabled, _ = await repos.strategy_runtime.is_disabled(
        strategy.id, now=now
    )
    if already_disabled:
        return False

    account_id = config.get("account", {}).get("id", "primary")
    pnl = await compute_rolling_pnl(
        repos, strategy.id, account_id=account_id, now=now
    )
    # threshold is a negative number (e.g. -300). Disable when pnl <= threshold.
    if pnl > threshold:
        log_checkpoint(
            "auto_disable_check",
            status="ok",
            strategy=strategy.id,
            rolling_pnl=pnl,
            threshold=threshold,
            triggered=False,
        )
        return False

    until = now + timedelta(days=AUTO_RESET_DAYS)
    reason = (
        f"rolling {ROLLING_WINDOW_DAYS}d realized P&L ${pnl:.2f} "
        f"≤ threshold ${threshold:.2f}"
    )
    await repos.strategy_runtime.disable(
        strategy.id, until=until, reason=reason, now=now,
    )
    log_checkpoint(
        "auto_disable_triggered",
        status="ok",
        strategy=strategy.id,
        rolling_pnl=pnl,
        threshold=threshold,
        disabled_until=until.isoformat(),
    )
    await notify(
        event_type="strategy_auto_disabled",
        title=f"Auto-disabled strategy: {strategy.id}",
        strategy=strategy.id,
        rolling_pnl=pnl,
        threshold=threshold,
        disabled_until=until.isoformat(),
        reason=reason,
    )
    return True


async def is_currently_disabled(
    repos: Repos,
    strategy_id: str,
    *,
    now: datetime | None = None,
) -> tuple[bool, str | None]:
    """Convenience wrapper over the repo. Returns (disabled, reason)."""
    return await repos.strategy_runtime.is_disabled(strategy_id, now=now)
