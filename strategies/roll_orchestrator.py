"""Roll orchestrator — runs the rule-based roll advisor for the reconciler.

Until 2026-10-27 this combined the rule with an (always-disabled) Opus+Haiku
ensemble and halted on disagreement. The LLM path was removed in the
two-strategy cleanup; the rule in `strategies/roll_advisor.py` is the whole
decision. Below-trigger positions return `RollOutcome(action=None)` — the
caller does nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from core.broker import Broker
from core.checkpoint import log_checkpoint
from db.repo import Repos
from strategies.roll_advisor import (
    RollAction,
    RollContext,
    RollDecision,
    evaluate_roll,
)


@dataclass(frozen=True, slots=True)
class RollOutcome:
    """Surfaced to the reconciler. `action=None` means below-trigger."""

    action: RollAction | None
    rule: RollDecision | None
    halted: bool
    reason: str


async def evaluate(
    *,
    broker: Broker,
    repos: Repos,
    ctx: RollContext,
    position_id: int | None,
    config: dict[str, Any],
    universe: dict[str, Any],
    today: date | None = None,
) -> RollOutcome:
    today = today or date.today()
    rule = await evaluate_roll(
        broker=broker, ctx=ctx, config=config, universe=universe, today=today
    )
    if rule is None:
        return RollOutcome(action=None, rule=None, halted=False, reason="below_trigger")
    log_checkpoint(
        "roll_orchestrator",
        status="ok",
        symbol=ctx.symbol,
        position_id=position_id,
        action=rule.action.value,
    )
    return RollOutcome(action=rule.action, rule=rule, halted=False, reason="rule")
