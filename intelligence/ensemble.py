"""Multi-model voting helper.

Sends the same prompt to multiple models in parallel and collapses the answers
into one of three states:

    "unanimous"     — every model returned the same `decision` string
    "majority"      — at least `quorum` models agreed; that decision wins
    "no_consensus"  — neither — caller should halt for human review

Mirrors the PolyTrader weather-strategy ensemble pattern from spec §9.4.

Used by:
  - The Phase-2 LLM roll advisor (Sprint 8).
  - Anywhere we want a "second pair of eyes" — happy to wire to news_check
    later if the spec evolves.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass
from typing import Any

from core.checkpoint import log_checkpoint
from core.models import LlmDecisionType
from intelligence.anthropic_client import AnthropicClient


@dataclass(frozen=True, slots=True)
class EnsembleResult:
    decision: str | None
    agreement: str  # "unanimous" | "majority" | "no_consensus"
    per_model: list[dict[str, Any]]


async def ensemble_vote(
    client: AnthropicClient,
    *,
    decision_type: LlmDecisionType,
    system: str,
    user_payload: dict[str, Any] | str,
    models: list[str],
    quorum: int = 2,
    max_output_tokens: int = 512,
    context: dict[str, Any] | None = None,
) -> EnsembleResult:
    """Run the prompt against every model in parallel and tally the decisions."""
    if not models:
        return EnsembleResult(decision=None, agreement="no_consensus", per_model=[])

    async def _one(model: str) -> dict[str, Any]:
        try:
            result = await client.call(
                decision_type=decision_type,
                model=model,
                system=system,
                user_payload=user_payload,
                max_output_tokens=max_output_tokens,
                context=context,
            )
            decision = (result["parsed"] or {}).get("decision")
            return {"model": model, "decision": decision, "raw": result}
        except Exception as exc:
            log_checkpoint(
                "ensemble_member_fail",
                status="fail",
                model=model,
                error=str(exc),
            )
            return {"model": model, "decision": None, "error": str(exc)}

    per_model = await asyncio.gather(*(_one(m) for m in models))
    decisions = [r["decision"] for r in per_model if r["decision"] is not None]
    if not decisions:
        return EnsembleResult(decision=None, agreement="no_consensus", per_model=per_model)

    counts = Counter(decisions)
    top, top_count = counts.most_common(1)[0]
    if len(set(decisions)) == 1 and len(decisions) == len(per_model):
        return EnsembleResult(decision=top, agreement="unanimous", per_model=per_model)
    if top_count >= quorum:
        return EnsembleResult(decision=top, agreement="majority", per_model=per_model)
    return EnsembleResult(decision=None, agreement="no_consensus", per_model=per_model)
