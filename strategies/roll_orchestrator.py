"""Roll orchestrator — combines rule-based advisor with LLM ensemble.

Spec §9.3: rule-based recommendation is canonical; LLM is a "second pair of
eyes." On disagreement, halt the position for human review (write a
`state_log` row, fire a `roll.disagreement` notification, no orders placed).

Behavior matrix:

  rule != None, llm disabled                    → execute rule
  rule != None, llm == None (skip/budget)       → execute rule + log "llm skipped"
  rule == llm                                   → execute rule (logged "agreed")
  rule != llm                                   → halt, no orders, notify

Below-trigger positions return `RollOutcome(action=None)` — caller does nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from core.broker import Broker
from core.checkpoint import log_checkpoint
from core.models import (
    OptionType,
    PositionState,
    StateLog,
    StateLogTrigger,
)
from core.notify import notify
from db.repo import Repos
from intelligence.anthropic_client import AnthropicClient
from intelligence.roll_advisor_llm import LlmRollResult, llm_roll_decision
from strategies.roll_advisor import (
    RollAction,
    RollContext,
    RollDecision,
    evaluate_roll,
)


@dataclass(frozen=True, slots=True)
class RollOutcome:
    """Surfaced to the reconciler. `action=None` means below-trigger or halt."""

    action: RollAction | None
    rule: RollDecision | None
    llm: LlmRollResult | None
    halted: bool
    reason: str


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


async def evaluate(
    *,
    broker: Broker,
    repos: Repos,
    anthropic: AnthropicClient | None,
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
        return RollOutcome(action=None, rule=None, llm=None, halted=False, reason="below_trigger")

    intel = config.get("intelligence", {}) or {}
    use_llm = bool(intel.get("llm_roll_advisor_enabled", False)) and anthropic is not None
    llm: LlmRollResult | None = None
    if use_llm:
        llm = await llm_roll_decision(client=anthropic, ctx=ctx, config=config)

    if llm is None or llm.decision is None:
        # LLM disabled, no consensus, or budget — fall back to rule alone.
        log_checkpoint(
            "roll_orchestrator",
            status="ok",
            symbol=ctx.symbol,
            outcome="rule_only",
            rule=rule.action.value,
        )
        return RollOutcome(
            action=rule.action,
            rule=rule,
            llm=llm,
            halted=False,
            reason="rule_only" if not use_llm else "llm_no_consensus",
        )

    if rule.action == llm.decision:
        log_checkpoint(
            "roll_orchestrator",
            status="ok",
            symbol=ctx.symbol,
            outcome="agreed",
            decision=rule.action.value,
        )
        return RollOutcome(
            action=rule.action, rule=rule, llm=llm, halted=False, reason="agreed"
        )

    # Disagreement: halt this position only.
    halt_reason = (
        f"rule={rule.action.value} vs llm={llm.decision.value} "
        f"({llm.agreement}); halt for review"
    )
    log_checkpoint(
        "roll_orchestrator",
        status="fail",
        symbol=ctx.symbol,
        outcome="disagreement",
        rule=rule.action.value,
        llm=llm.decision.value,
        agreement=llm.agreement,
    )
    if position_id is not None:
        existing = await repos.positions.get(position_id)
        from_state = existing.state if existing else None
        await repos.positions.update_state(
            position_id,
            PositionState.MANUAL_INTERVENTION,
            halt_reason,
            when=_utcnow(),
        )
        await repos.state_log.insert(
            StateLog(
                position_id=position_id,
                from_state=from_state,
                to_state=PositionState.MANUAL_INTERVENTION,
                reason=halt_reason,
                triggered_by=StateLogTrigger.STRATEGY,
                metadata={
                    "rule_decision": rule.action.value,
                    "llm_decision": llm.decision.value,
                    "llm_agreement": llm.agreement,
                    "rule_rationale": rule.rationale,
                    "llm_rationale": llm.rationale,
                },
                created_at=_utcnow(),
            )
        )
    await notify(
        "roll.disagreement",
        f"{ctx.symbol} roll halted",
        symbol=ctx.symbol,
        rule=rule.action.value,
        rule_rationale=rule.rationale,
        llm=llm.decision.value,
        llm_agreement=llm.agreement,
        llm_rationale=llm.rationale,
    )
    return RollOutcome(action=None, rule=rule, llm=llm, halted=True, reason=halt_reason)
