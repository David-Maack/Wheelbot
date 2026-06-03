"""TICKET-009 — per-strategy win-rate floor.

Companion to `risk/auto_disable.py`. Where the drawdown breaker catches
dollar-magnitude bleed, this catches *frequency-of-loss* bleed:

    win_rate = wins / N   over the last N closed cycles

When `win_rate < min_win_rate_pct`, the strategy is PAUSED. New entries
are skipped on every tick. Recovery requires manual operator action — the
underlying signal is a *pattern* of losing trades, not a transient
drawdown, so auto-clearing on win-rate recovery (or on calendar timeout)
would silently re-arm a broken setup. The pause persists indefinitely
until `scripts/reenable_strategy.py` clears it.

`paused_reenable_eligible_at = paused_at + pause_duration_days` is stored
on the row as advisory metadata only — the dashboard shows "eligible for
review in N days" so the operator knows when it's reasonable to revisit,
but the bot never reads this field. Operators can reenable sooner if
they have new information; the wait period is guidance, not a lock.

State machine (independent of drawdown):

    NORMAL                 → PAUSED_LOW_WIN_RATE   fires `strategy.paused_low_win_rate`
                                                   (24h-rate-limited via alert_rate_limits)
    PAUSED_LOW_WIN_RATE    → PAUSED_LOW_WIN_RATE   silent (held)
    PAUSED_LOW_WIN_RATE    → NORMAL                only via scripts/reenable_strategy.py
                                                   (operator action, not P&L recovery)

Interaction with drawdown (TICKET-008):

    drawdown DISABLED + pause LOW_WIN_RATE → both columns set, both
      independent. Bot's `_propose_and_route` checks drawdown first
      (cheaper read); pause is checked second. Either trips → skip.
      Manual `enable()` clears both at once via row DELETE.

    drawdown WARNING + pause LOW_WIN_RATE → pause wins the skip decision
      (more restrictive). Drawdown WARNING's size halving is moot when
      pause is set, but the warning's display in `/risk` is unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from core.checkpoint import log_checkpoint
from core.notify import notify
from core.strategies import StrategyDefinition
from db.repo import Repos

# 24h Discord cooldown per strategy. Drawdown WARNING uses 4h; win-rate trips
# are slower-moving so a longer cooldown is appropriate (avoids alert
# fatigue if a strategy is reenabled and re-trips on the same day).
PAUSE_ALERT_COOLDOWN_HOURS = 24.0

# Default eligible-reenable window. Stored on the row as advisory metadata.
DEFAULT_PAUSE_DURATION_DAYS = 14


class WinRateState(StrEnum):
    """Per-strategy pause gauge. Independent of `DrawdownState`.

    `.value` matches the DB `pause_state` column literals (NULL = NORMAL,
    'LOW_WIN_RATE' = PAUSED_LOW_WIN_RATE). The enum name carries the
    state-class context for readers; the value is the discriminator
    actually persisted."""

    NORMAL = "NORMAL"
    PAUSED_LOW_WIN_RATE = "LOW_WIN_RATE"


@dataclass(frozen=True, slots=True)
class WinRateResult:
    """Outcome of a win-rate evaluation."""

    action: Literal["normal", "insufficient_data", "pause"]
    win_rate: float | None  # percent (0–100) or None when insufficient data
    sample_size: int


def _config_block(config: dict[str, Any]) -> dict[str, Any]:
    """Read the `risk.win_rate_floor` block. Empty dict when absent — caller
    treats as feature off."""
    return ((config.get("risk") or {}).get("win_rate_floor") or {})


def _threshold_for(
    strategy: StrategyDefinition, config: dict[str, Any],
) -> float | None:
    """Per-strategy override beats global default. None when feature off."""
    block = _config_block(config)
    if not block.get("enabled", False):
        return None
    override = strategy.params.get("win_rate_floor_pct")
    if override is not None:
        return float(override)
    g = block.get("min_win_rate_pct")
    return float(g) if g is not None else None


async def evaluate(
    repos: Repos,
    strategy_id: str,
    config: dict[str, Any],
    *,
    account_id: str = "primary",
) -> WinRateResult:
    """Pure read. Pulls the last N closed cycles and computes the win rate.

    `final_pnl > 0` = win, `final_pnl <= 0` = loss (matches the spec — a
    pathological zero-P&L close counts as a loss; conservative).

    Returns `insufficient_data` when fewer than `min_closed_cycles` are
    available — fresh strategies are unprotected by this gate until they
    accumulate enough cycles. The drawdown breaker still covers dollar
    losses meanwhile.
    """
    block = _config_block(config)
    n = int(block.get("min_closed_cycles", 10))
    cycles = await repos.cycles.list_closed_for_strategy(
        account_id, strategy_id, limit=n,
    )
    sample = len(cycles)
    if sample < n:
        return WinRateResult(action="insufficient_data", win_rate=None, sample_size=sample)
    wins = sum(1 for c in cycles if c.final_pnl is not None and c.final_pnl > 0)
    rate = wins / sample * 100.0
    threshold = block.get("min_win_rate_pct")
    if threshold is None:
        # Block configured but no threshold → can't decide. Treat as normal.
        return WinRateResult(action="normal", win_rate=rate, sample_size=sample)
    if rate < float(threshold):
        return WinRateResult(action="pause", win_rate=rate, sample_size=sample)
    return WinRateResult(action="normal", win_rate=rate, sample_size=sample)


async def current_pause_state(
    repos: Repos,
    strategy_id: str,
) -> WinRateState:
    """Read the persisted pause state. Does NOT auto-clear — operator must
    run scripts/reenable_strategy.py."""
    row = await repos.strategy_runtime.get(strategy_id)
    if row is None:
        return WinRateState.NORMAL
    raw = (row.get("pause_state") or "").upper()
    if raw == WinRateState.PAUSED_LOW_WIN_RATE.value:
        return WinRateState.PAUSED_LOW_WIN_RATE
    return WinRateState.NORMAL


async def check_and_apply(
    repos: Repos,
    strategy: StrategyDefinition,
    config: dict[str, Any],
    *,
    now: datetime | None = None,
) -> WinRateResult:
    """Evaluate and persist on transitions.

    NORMAL → PAUSED: writes the pause columns + fires
                     `strategy.paused_low_win_rate` Discord (rate-limited
                     24h via alert_rate_limits).
    PAUSED → *     : never auto-recovers. Even if rate climbs back above
                     the floor, the row stays. Logs `win_rate_recovered_pause_held`
                     so the operator sees the signal in the dashboard /
                     logs and can decide to manually reenable.
    No-op (returns `WinRateResult(action="normal", win_rate=None, sample_size=0)`)
    when `risk.win_rate_floor.enabled` is false.
    """
    block = _config_block(config)
    if not block.get("enabled", False):
        return WinRateResult(action="normal", win_rate=None, sample_size=0)

    now = now or datetime.now(UTC).replace(tzinfo=None)
    account_id = config.get("account", {}).get("id", "primary")
    # Re-route through per-strategy override into the threshold check.
    threshold = _threshold_for(strategy, config)
    if threshold is None:
        return WinRateResult(action="normal", win_rate=None, sample_size=0)

    cycles = await repos.cycles.list_closed_for_strategy(
        account_id, strategy.id,
        limit=int(block.get("min_closed_cycles", 10)),
    )
    n_min = int(block.get("min_closed_cycles", 10))
    sample = len(cycles)
    if sample < n_min:
        return WinRateResult(action="insufficient_data", win_rate=None, sample_size=sample)
    wins = sum(1 for c in cycles if c.final_pnl is not None and c.final_pnl > 0)
    rate = wins / sample * 100.0
    target_pause = rate < threshold

    prior = await current_pause_state(repos, strategy.id)

    if target_pause and prior is WinRateState.NORMAL:
        eligible_in = int(block.get("pause_duration_days", DEFAULT_PAUSE_DURATION_DAYS))
        reason = (
            f"win_rate_floor: {rate:.1f}% over last {sample} closed cycles "
            f"< floor {threshold:.1f}%"
        )
        await repos.strategy_runtime.mark_paused(
            strategy.id,
            reason=reason,
            now=now,
            eligible_in_days=eligible_in,
        )
        log_checkpoint(
            "win_rate_pause_triggered",
            status="ok",
            strategy=strategy.id,
            win_rate=rate,
            threshold=threshold,
            sample_size=sample,
            eligible_in_days=eligible_in,
        )
        if await repos.alert_rate_limits.try_fire(
            f"win_rate_floor_{strategy.id}",
            cooldown_hours=PAUSE_ALERT_COOLDOWN_HOURS,
        ):
            await notify(
                "strategy.paused_low_win_rate",
                f"{strategy.id} paused — low win rate",
                strategy=strategy.id,
                win_rate=rate,
                threshold=threshold,
                sample_size=sample,
                eligible_in_days=eligible_in,
                reason=reason,
            )
        return WinRateResult(action="pause", win_rate=rate, sample_size=sample)

    if prior is WinRateState.PAUSED_LOW_WIN_RATE:
        # Recovery does NOT auto-clear. Log so the operator can see the
        # signal in /risk and the bot logs.
        log_checkpoint(
            "win_rate_recovered_pause_held" if not target_pause else "win_rate_pause_held",
            status="ok",
            strategy=strategy.id,
            win_rate=rate,
            threshold=threshold,
            sample_size=sample,
            note="pause held — operator must manually reenable" if not target_pause
                 else "still below floor",
        )
        return WinRateResult(
            action="pause", win_rate=rate, sample_size=sample,
        )

    # target NORMAL, prior NORMAL — quiet path.
    return WinRateResult(action="normal", win_rate=rate, sample_size=sample)


async def get_win_rate_overview(
    repos: Repos,
    strategies: list[StrategyDefinition],
    config: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, dict[str, Any]]:
    """Dashboard-ready per-strategy overview. Used by `/risk` (Win-rate
    panel) and `/performance` (per-strategy badge).

    Returns `{strategy_id: {state, win_rate, sample_size, threshold,
    paused_at, paused_reenable_eligible_at, eligible_in_days}}`.
    Recomputes the win rate fresh — does NOT trust a stale persisted value
    (a strategy can be PAUSED with a recovered rate; the dashboard should
    show both truths).
    """
    now = now or datetime.now(UTC).replace(tzinfo=None)
    account_id = config.get("account", {}).get("id", "primary")
    out: dict[str, dict[str, Any]] = {}
    for s in strategies:
        result = await evaluate(repos, s.id, config, account_id=account_id)
        state = await current_pause_state(repos, s.id)
        row = await repos.strategy_runtime.get(s.id) or {}
        eligible_at_raw = row.get("paused_reenable_eligible_at")
        eligible_in_days: int | None = None
        if eligible_at_raw:
            try:
                eligible_at = datetime.fromisoformat(eligible_at_raw)
                eligible_in_days = max(0, (eligible_at - now).days)
            except ValueError:
                eligible_in_days = None
        out[s.id] = {
            "state": state.value,
            "win_rate": result.win_rate,
            "sample_size": result.sample_size,
            "threshold": _threshold_for(s, config),
            "paused_at": row.get("paused_at"),
            "paused_reason": row.get("paused_reason"),
            "paused_reenable_eligible_at": eligible_at_raw,
            "eligible_in_days": eligible_in_days,
            "action": result.action,
        }
    return out
