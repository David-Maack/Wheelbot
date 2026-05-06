"""LLM ensemble roll advisor — spec §9.3 Phase 2.

Defaults to disabled (`intelligence.llm_roll_advisor_enabled: false`). Enable
once you've collected ~3 months of cycle data per spec §12 Phase 5 — that's
the point at which the audit-trail query
`SELECT outcome FROM llm_decisions WHERE acted_on=TRUE` can tell you whether
the LLM helped or hurt.

Returns the same `RollAction` values as the rule-based advisor. The
orchestrator (`strategies/roll_orchestrator.py`) compares them; on
disagreement, it halts the position and posts for human review.

Decision JSON contract (model is instructed to emit exactly this):

    {
      "decision": "ROLL" | "LET_ASSIGN" | "CLOSE",
      "rationale": "<= 240 chars",
      "confidence": 0.0-1.0
    }
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.checkpoint import log_checkpoint
from core.models import LlmDecisionType
from intelligence.anthropic_client import AnthropicClient
from intelligence.budget import BudgetExceeded
from intelligence.ensemble import EnsembleResult, ensemble_vote
from strategies.roll_advisor import RollAction, RollContext


SYSTEM_PROMPT = """You are a second-opinion options trader reviewing a wheel
position whose short option has gone in-the-money. You will see the position
context: ticker, the current short leg (strike, DTE, current mid, original
premium), the underlying price, and the rule-based advisor's recommendation
isn't shown to you — you must reach an independent verdict.

Choose exactly one of:
  ROLL        — extend the short into a later expiration at a strike that
                pays a net credit. Choose this when the position can be
                defended without booking a loss.
  LET_ASSIGN  — for a short put: take the shares (we wanted to own this
                stock at this price anyway). Choose this when assignment is
                a fine outcome.
  CLOSE       — buy to close at a loss. Choose this only when the position
                has deteriorated enough that neither rolling nor accepting
                assignment is acceptable.

Be conservative — when in doubt, lean LET_ASSIGN for puts, ROLL for calls.

Return exactly one JSON object, no markdown:

{
  "decision": "ROLL" | "LET_ASSIGN" | "CLOSE",
  "rationale": "<= 240 chars",
  "confidence": 0.0-1.0
}
"""


@dataclass(frozen=True, slots=True)
class LlmRollResult:
    decision: RollAction | None
    agreement: str  # "unanimous" | "majority" | "no_consensus"
    rationale: str = ""


def _build_payload(ctx: RollContext) -> dict[str, Any]:
    sc = ctx.short_contract
    side_value = sc.option_type.value if hasattr(sc.option_type, "value") else str(sc.option_type)
    return {
        "symbol": ctx.symbol,
        "side": "PUT" if side_value == "PUT" else "CALL",
        "short_strike": sc.strike,
        "short_expiration": str(sc.expiration),
        "short_quantity": ctx.short_quantity,
        "short_delta_now": sc.delta,
        "short_mid_now": ctx.current_short_mid,
        "short_premium_collected": ctx.short_premium_collected_per_share,
        "underlying_price": ctx.underlying_price,
    }


def _normalize_decision(raw: Any) -> str | None:
    s = str(raw or "").strip().upper()
    return s if s in {"ROLL", "LET_ASSIGN", "CLOSE"} else None


async def llm_roll_decision(
    *,
    client: AnthropicClient,
    ctx: RollContext,
    config: dict[str, Any],
) -> LlmRollResult:
    """Run the ensemble. Always returns a result — never raises (BudgetExceeded
    surfaces as no_consensus + status=skipped)."""
    intel = config.get("intelligence", {}) or {}
    if not bool(intel.get("llm_roll_advisor_enabled", False)):
        log_checkpoint("llm_roll_disabled", status="skip", symbol=ctx.symbol)
        return LlmRollResult(decision=None, agreement="no_consensus", rationale="LLM advisor disabled")

    models = list(intel.get("roll_advisor_models") or [])
    if not models:
        return LlmRollResult(decision=None, agreement="no_consensus", rationale="no models configured")
    quorum = int(intel.get("ensemble_quorum", 2))

    try:
        result: EnsembleResult = await ensemble_vote(
            client,
            decision_type=LlmDecisionType.ROLL_ADVISE,
            system=SYSTEM_PROMPT,
            user_payload=_build_payload(ctx),
            models=models,
            quorum=quorum,
            max_output_tokens=int(intel.get("roll_advisor_max_tokens", 384)),
            context={"symbol": ctx.symbol},
        )
    except BudgetExceeded as exc:
        log_checkpoint("llm_roll_budget_skip", status="skip", symbol=ctx.symbol, error=str(exc))
        return LlmRollResult(decision=None, agreement="no_consensus", rationale="budget exhausted")

    normalized = _normalize_decision(result.decision)
    action = RollAction(normalized) if normalized else None
    rationale = ""
    # Surface the most-recent rationale we can find for the picked decision.
    for member in result.per_model:
        raw = member.get("raw") or {}
        parsed = raw.get("parsed") or {}
        if _normalize_decision(parsed.get("decision")) == normalized:
            rationale = str(parsed.get("rationale", ""))[:400]
            break
    log_checkpoint(
        "llm_roll_decide",
        status="ok",
        symbol=ctx.symbol,
        decision=action.value if action else None,
        agreement=result.agreement,
    )
    return LlmRollResult(decision=action, agreement=result.agreement, rationale=rationale)
