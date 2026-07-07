"""Per-strategy expectancy floor (2026-07-06 sprint).

Companion to `risk/win_rate_floor.py`, closing its blind spot: win rate is
the wrong pause metric for asymmetric payoffs. A spread strategy with $54
average wins and -$150 stops can pass a 60% win-rate floor while losing
money (live put_spread: 57% WR, negative P&L), and a wheel with lumpy wins
can sit just under the floor while profitable (monthly_wheel: 56% WR, +$150).
This gauge pauses on the thing we actually care about:

    expectancy = mean(final_pnl) over the last N closed cycles

When `expectancy < min_expectancy_usd` (default $0), the strategy is PAUSED —
new entries are skipped on every tick; closes keep running. Same recovery
contract as the win-rate floor: manual operator reenable only
(`scripts/reenable_strategy.py`), with an advisory eligible-at date.

Shares the `strategy_runtime_state.pause_state` column with the win-rate
floor (a strategy is paused or it isn't; two literals say why):

    NULL                    no pause
    'LOW_WIN_RATE'          win-rate floor (TICKET-009)
    'NEGATIVE_EXPECTANCY'   this module

Neither floor overwrites the other's pause — first tripper wins the column,
the other logs a held checkpoint. The run_bot opens-gate treats ANY non-NULL
pause_state as entry-suppressing, so which floor fired first never affects
trading behavior, only the recorded reason.
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

PAUSE_ALERT_COOLDOWN_HOURS = 24.0
DEFAULT_PAUSE_DURATION_DAYS = 14


class ExpectancyState(StrEnum):
    """`.value` matches the DB `pause_state` literal (NULL = NORMAL)."""

    NORMAL = "NORMAL"
    PAUSED_NEGATIVE_EXPECTANCY = "NEGATIVE_EXPECTANCY"


@dataclass(frozen=True, slots=True)
class ExpectancyResult:
    action: Literal["normal", "insufficient_data", "pause", "held_other_pause"]
    expectancy: float | None  # mean $/cycle, None when insufficient data
    sample_size: int


def _config_block(config: dict[str, Any]) -> dict[str, Any]:
    return ((config.get("risk") or {}).get("expectancy_floor") or {})


def _threshold_for(
    strategy: StrategyDefinition, config: dict[str, Any],
) -> float | None:
    """Per-strategy `expectancy_floor_usd` override beats the global
    `min_expectancy_usd`. None when the feature is off."""
    block = _config_block(config)
    if not block.get("enabled", False):
        return None
    override = strategy.params.get("expectancy_floor_usd")
    if override is not None:
        return float(override)
    g = block.get("min_expectancy_usd")
    return float(g) if g is not None else None


async def evaluate(
    repos: Repos,
    strategy_id: str,
    config: dict[str, Any],
    *,
    account_id: str = "primary",
) -> ExpectancyResult:
    """Pure read: mean final_pnl over the last N closed cycles.
    `list_closed_for_strategy` already excludes NULL-final_pnl rows (same
    contract as the win-rate floor and drawdown breaker), so
    partially-recorded closes never bias the mean; the `or 0.0` below is
    belt-and-suspenders only."""
    block = _config_block(config)
    n = int(block.get("min_closed_cycles", 10))
    cycles = await repos.cycles.list_closed_for_strategy(
        account_id, strategy_id, limit=n,
    )
    sample = len(cycles)
    if sample < n:
        return ExpectancyResult(action="insufficient_data", expectancy=None, sample_size=sample)
    mean = sum((c.final_pnl or 0.0) for c in cycles) / sample
    return ExpectancyResult(action="normal", expectancy=mean, sample_size=sample)


async def current_pause_state(repos: Repos, strategy_id: str) -> ExpectancyState:
    row = await repos.strategy_runtime.get(strategy_id)
    if row is None:
        return ExpectancyState.NORMAL
    raw = (row.get("pause_state") or "").upper()
    if raw == ExpectancyState.PAUSED_NEGATIVE_EXPECTANCY.value:
        return ExpectancyState.PAUSED_NEGATIVE_EXPECTANCY
    return ExpectancyState.NORMAL


async def check_and_apply(
    repos: Repos,
    strategy: StrategyDefinition,
    config: dict[str, Any],
    *,
    now: datetime | None = None,
) -> ExpectancyResult:
    """Evaluate and persist on transitions. Mirrors win_rate_floor:

    NORMAL → PAUSED : writes pause columns + Discord (24h rate-limited).
    PAUSED → *      : never auto-recovers; logs recovery so the operator
                      can decide to reenable.
    Another pause already set (e.g. LOW_WIN_RATE): held — do NOT overwrite.
    No-op when `risk.expectancy_floor.enabled` is false.
    """
    threshold = _threshold_for(strategy, config)
    if threshold is None:
        return ExpectancyResult(action="normal", expectancy=None, sample_size=0)

    now = now or datetime.now(UTC).replace(tzinfo=None)
    account_id = config.get("account", {}).get("id", "primary")
    result = await evaluate(repos, strategy.id, config, account_id=account_id)
    if result.action == "insufficient_data":
        return result
    assert result.expectancy is not None
    target_pause = result.expectancy < threshold

    row = await repos.strategy_runtime.get(strategy.id) or {}
    raw_pause = (row.get("pause_state") or "").upper()

    if raw_pause == ExpectancyState.PAUSED_NEGATIVE_EXPECTANCY.value:
        log_checkpoint(
            "expectancy_recovered_pause_held" if not target_pause else "expectancy_pause_held",
            status="ok",
            strategy=strategy.id,
            expectancy=result.expectancy,
            threshold=threshold,
            sample_size=result.sample_size,
            note="pause held — operator must manually reenable" if not target_pause
                 else "still below floor",
        )
        return ExpectancyResult(
            action="pause", expectancy=result.expectancy, sample_size=result.sample_size,
        )

    if raw_pause:
        # A different gauge (win-rate floor) owns the pause. Entries are
        # already suppressed; don't overwrite its reason columns.
        if target_pause:
            log_checkpoint(
                "expectancy_pause_deferred_other_pause",
                status="skip",
                strategy=strategy.id,
                expectancy=result.expectancy,
                threshold=threshold,
                other_pause=raw_pause,
            )
        return ExpectancyResult(
            action="held_other_pause",
            expectancy=result.expectancy,
            sample_size=result.sample_size,
        )

    if target_pause:
        eligible_in = int(
            _config_block(config).get("pause_duration_days", DEFAULT_PAUSE_DURATION_DAYS)
        )
        reason = (
            f"expectancy_floor: ${result.expectancy:.2f}/cycle over last "
            f"{result.sample_size} closed cycles < floor ${threshold:.2f}"
        )
        await repos.strategy_runtime.mark_paused(
            strategy.id,
            reason=reason,
            now=now,
            eligible_in_days=eligible_in,
            state=ExpectancyState.PAUSED_NEGATIVE_EXPECTANCY.value,
        )
        log_checkpoint(
            "expectancy_pause_triggered",
            status="ok",
            strategy=strategy.id,
            expectancy=result.expectancy,
            threshold=threshold,
            sample_size=result.sample_size,
            eligible_in_days=eligible_in,
        )
        if await repos.alert_rate_limits.try_fire(
            f"expectancy_floor_{strategy.id}",
            cooldown_hours=PAUSE_ALERT_COOLDOWN_HOURS,
        ):
            await notify(
                "strategy.paused_negative_expectancy",
                f"{strategy.id} paused — negative expectancy",
                strategy=strategy.id,
                expectancy=result.expectancy,
                threshold=threshold,
                sample_size=result.sample_size,
                eligible_in_days=eligible_in,
                reason=reason,
            )
        return ExpectancyResult(
            action="pause", expectancy=result.expectancy, sample_size=result.sample_size,
        )

    return result


async def get_expectancy_overview(
    repos: Repos,
    strategies: list[StrategyDefinition],
    config: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, dict[str, Any]]:
    """Dashboard-ready per-strategy overview for `/risk`. Recomputes the
    expectancy fresh — a strategy can be PAUSED with a recovered mean and the
    panel should show both truths."""
    now = now or datetime.now(UTC).replace(tzinfo=None)
    account_id = config.get("account", {}).get("id", "primary")
    out: dict[str, dict[str, Any]] = {}
    for s in strategies:
        result = await evaluate(repos, s.id, config, account_id=account_id)
        state = await current_pause_state(repos, s.id)
        row = await repos.strategy_runtime.get(s.id) or {}
        out[s.id] = {
            "state": state.value,
            "expectancy": result.expectancy,
            "sample_size": result.sample_size,
            "threshold": _threshold_for(s, config),
            "paused_at": row.get("paused_at"),
            "paused_reason": row.get("paused_reason"),
            "action": result.action,
        }
    return out
