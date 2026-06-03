"""Sprint 13 sub-sprint 1 — auto-disable on drawdown.
TICKET-008 extension — added intermediate WARNING tier.

Defensive circuit breaker. For each strategy, sums realized P&L from
cycles closed in the last 7 days. Three states:

    NORMAL    — rolling P&L above `drawdown_warning_usd`. Sizes unchanged.
    WARNING   — rolling P&L in `(drawdown_warning_usd, drawdown_disable_usd]`.
                New spread entries placed at HALF size. Wheel entries stay
                at qty=1 (single contracts can't halve — logged but not
                enforced). Re-evaluated every check; recovers silently when
                P&L crosses back above warning.
    DISABLED  — rolling P&L at or below `drawdown_disable_usd`. Strategy
                skipped entirely. 30-day auto-reset window (existing).

Wired into scripts/run_bot.py: at the top of each strategy iteration, the
loop reads `current_state(...)`. DISABLED → skip; WARNING → set
size_multiplier=0.5 on proposers; NORMAL → size_multiplier=1.0. After the
iteration, `check_and_apply(...)` updates the state from the latest rolling
P&L (so today's just-closed cycles count toward tomorrow's evaluation).

Transitions and Discord:
    NORMAL    → WARNING   fires `drawdown.warning` (rate-limited 4h via
                          alert_rate_limits — per-strategy key)
    NORMAL    → DISABLED  fires `strategy_auto_disabled` (existing)
    WARNING   → DISABLED  fires `strategy_auto_disabled` only
    WARNING   → NORMAL    silent (no notify, just clear)
    DISABLED  → *         only via the 30-day auto-reset OR manual reenable
                          — P&L recovery does NOT downgrade DISABLED to
                          avoid thrashing.

Disable is "soft" — config.yaml is unchanged. Runtime state lives in the
`strategy_runtime_state` table (drawdown_state column added by migration 010).
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
# 4h cooldown via alert_rate_limits so a P&L hovering around the threshold
# doesn't re-spam Discord every check.
WARNING_ALERT_COOLDOWN_HOURS = 4.0


class DrawdownState(StrEnum):
    """Per-strategy drawdown gauge — TICKET-008."""

    NORMAL = "NORMAL"
    WARNING = "WARNING"
    DISABLED = "DISABLED"

    @property
    def size_multiplier(self) -> float:
        """Quantity multiplier the proposers apply. Spreads halve; wheels
        can't (qty=1 baseline) — see strategies/wheel.py for the unhalved
        log."""
        return 0.5 if self is DrawdownState.WARNING else 1.0


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
    """Read the persisted drawdown state for a strategy.

    Auto-clears stale DISABLED rows (disabled_until in the past) — same
    behaviour the legacy `is_currently_disabled` had. WARNING-only rows have
    `disabled_until=NULL` and persist until P&L recovers (or operator clears).
    """
    now = now or datetime.now(UTC).replace(tzinfo=None)
    row = await repos.strategy_runtime.get(strategy_id)
    if row is None:
        return DrawdownState.NORMAL
    disabled_until_str = row.get("disabled_until")
    if disabled_until_str:
        until = datetime.fromisoformat(disabled_until_str)
        if until > now:
            return DrawdownState.DISABLED
        # Stale DISABLED → auto-clear the whole row (existing behaviour).
        await repos.strategy_runtime.enable(strategy_id)
        return DrawdownState.NORMAL
    raw_state = (row.get("drawdown_state") or "NORMAL").upper()
    if raw_state == DrawdownState.WARNING.value:
        return DrawdownState.WARNING
    return DrawdownState.NORMAL


async def is_currently_disabled(
    repos: Repos,
    strategy_id: str,
    *,
    now: datetime | None = None,
) -> tuple[bool, str | None]:
    """Backward-compat shim. Returns (True, reason) iff state is DISABLED."""
    state = await current_state(repos, strategy_id, now=now)
    if state is DrawdownState.DISABLED:
        row = await repos.strategy_runtime.get(strategy_id)
        return True, (row or {}).get("disabled_reason")
    return False, None


async def check_and_apply(
    repos: Repos,
    strategy: StrategyDefinition,
    config: dict[str, Any],
    *,
    now: datetime | None = None,
) -> DrawdownState:
    """Evaluate the drawdown thresholds and update state. Returns the
    NEW state (after any transition).

    No-op (returns NORMAL) when the strategy has no `auto_disable_drawdown_usd`
    AND no `drawdown_warning_usd` — feature off entirely. A 0 value disables
    that specific tier only.
    """
    disable_thresh = strategy.params.get("auto_disable_drawdown_usd")
    warning_thresh = strategy.params.get("drawdown_warning_usd")
    if disable_thresh is None and warning_thresh is None:
        return DrawdownState.NORMAL

    now = now or datetime.now(UTC).replace(tzinfo=None)
    account_id = config.get("account", {}).get("id", "primary")
    pnl = await compute_rolling_pnl(
        repos, strategy.id, account_id=account_id, now=now,
    )
    disable_v = float(disable_thresh) if disable_thresh is not None else None
    warning_v = float(warning_thresh) if warning_thresh is not None else None

    if disable_v is not None and disable_v != 0 and pnl <= disable_v:
        target = DrawdownState.DISABLED
    elif warning_v is not None and warning_v != 0 and pnl <= warning_v:
        target = DrawdownState.WARNING
    else:
        target = DrawdownState.NORMAL

    prior = await current_state(repos, strategy.id, now=now)

    # DISABLED → anything: only via auto-reset (handled in current_state) or
    # manual reenable. Don't let a P&L blip downgrade DISABLED to WARNING /
    # NORMAL — protects against thrashing in/out of disable.
    if prior is DrawdownState.DISABLED:
        log_checkpoint(
            "auto_disable_check",
            status="ok",
            strategy=strategy.id,
            rolling_pnl=pnl,
            state=prior.value,
            note="DISABLED held — recovery only via 30d auto-reset or manual",
        )
        return DrawdownState.DISABLED

    if target is DrawdownState.DISABLED:
        until = now + timedelta(days=AUTO_RESET_DAYS)
        reason = (
            f"rolling {ROLLING_WINDOW_DAYS}d realized P&L ${pnl:.2f} "
            f"≤ disable threshold ${disable_v:.2f}"
        )
        await repos.strategy_runtime.disable(
            strategy.id, until=until, reason=reason, now=now,
        )
        await repos.strategy_runtime.set_drawdown_state(
            strategy.id, DrawdownState.DISABLED.value,
        )
        log_checkpoint(
            "auto_disable_triggered",
            status="ok",
            strategy=strategy.id,
            rolling_pnl=pnl,
            threshold=disable_v,
            disabled_until=until.isoformat(),
            prior_state=prior.value,
        )
        await notify(
            "strategy_auto_disabled",
            f"Auto-disabled strategy: {strategy.id}",
            strategy=strategy.id,
            rolling_pnl=pnl,
            threshold=disable_v,
            disabled_until=until.isoformat(),
            reason=reason,
            prior_state=prior.value,
        )
        return DrawdownState.DISABLED

    if target is DrawdownState.WARNING:
        if prior is DrawdownState.WARNING:
            log_checkpoint(
                "drawdown_warning_check",
                status="ok",
                strategy=strategy.id,
                rolling_pnl=pnl,
                threshold=warning_v,
                state="WARNING_held",
            )
            return DrawdownState.WARNING
        # NORMAL → WARNING: persist + rate-limited Discord
        reason = (
            f"rolling {ROLLING_WINDOW_DAYS}d realized P&L ${pnl:.2f} "
            f"≤ warning threshold ${warning_v:.2f}"
        )
        await repos.strategy_runtime.mark_warning(strategy.id, reason=reason, now=now)
        log_checkpoint(
            "drawdown_warning_triggered",
            status="ok",
            strategy=strategy.id,
            rolling_pnl=pnl,
            threshold=warning_v,
        )
        if await repos.alert_rate_limits.try_fire(
            f"drawdown_warning_{strategy.id}",
            cooldown_hours=WARNING_ALERT_COOLDOWN_HOURS,
        ):
            await notify(
                "drawdown.warning",
                f"{strategy.id} entered drawdown WARNING",
                strategy=strategy.id,
                rolling_pnl=pnl,
                threshold=warning_v,
                size_multiplier=DrawdownState.WARNING.size_multiplier,
                reason=reason,
            )
        return DrawdownState.WARNING

    # target is NORMAL
    if prior is DrawdownState.WARNING:
        await repos.strategy_runtime.enable(strategy.id)  # clears row
        log_checkpoint(
            "drawdown_warning_cleared",
            status="ok",
            strategy=strategy.id,
            rolling_pnl=pnl,
        )
        return DrawdownState.NORMAL
    log_checkpoint(
        "auto_disable_check",
        status="ok",
        strategy=strategy.id,
        rolling_pnl=pnl,
        threshold=disable_v if disable_v is not None else 0.0,
        triggered=False,
    )
    return DrawdownState.NORMAL


async def get_drawdown_overview(
    repos: Repos,
    strategies: list[StrategyDefinition],
    config: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, dict[str, Any]]:
    """Dashboard-ready per-strategy overview.

    Returns `{strategy_id: {state, rolling_pnl, warning_threshold,
    disable_threshold, disabled_until}}`. Used by `/` (tri-state badge) and
    `/risk` (drawdown status section).
    """
    now = now or datetime.now(UTC).replace(tzinfo=None)
    account_id = config.get("account", {}).get("id", "primary")
    out: dict[str, dict[str, Any]] = {}
    for s in strategies:
        state = await current_state(repos, s.id, now=now)
        pnl = await compute_rolling_pnl(
            repos, s.id, account_id=account_id, now=now,
        )
        row = await repos.strategy_runtime.get(s.id) or {}
        out[s.id] = {
            "state": state.value,
            "rolling_pnl": pnl,
            "warning_threshold": s.params.get("drawdown_warning_usd"),
            "disable_threshold": s.params.get("auto_disable_drawdown_usd"),
            "disabled_until": row.get("disabled_until"),
            "size_multiplier": state.size_multiplier,
        }
    return out
