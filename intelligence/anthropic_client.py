"""Single chokepoint for every Anthropic call we make.

Responsibilities:

  - Hold the AsyncAnthropic client.
  - Consult BudgetTracker before each call; raise BudgetExceeded when over.
  - Record every call to `llm_decisions` with token usage + dollar cost so the
    daily budget gate has data to sum, and the dashboard `/decisions` view has
    something to render.

Callers (screener, news_check, ensemble, future roll advisor) don't talk to the
SDK directly. They build a structured prompt + decision_type and ask the
client to send it.

Failure modes the caller cares about:
    BudgetExceeded             — pre-call, no API hit, no DB write.
    AnthropicAPIError          — network/auth/etc. Logged with status=fail.
    Output JSON parse failure  — caller's problem; client returns the raw text.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any

from anthropic import APIError as AnthropicAPIError
from anthropic import AsyncAnthropic

from core.checkpoint import log_checkpoint
from core.models import LlmDecision, LlmDecisionType
from db.repo import LlmDecisionsRepo
from intelligence.budget import BudgetExceeded, BudgetTracker


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def estimate_tokens(text: str) -> int:
    """Rough estimate (~4 chars per token). Used pre-call for the budget gate.

    Anthropic offers a precise count_tokens endpoint; the rough estimate is
    cheaper and good enough for a worst-case-budget check.
    """
    return max(1, len(text) // 4)


class AnthropicClient:
    def __init__(
        self,
        decisions_repo: LlmDecisionsRepo,
        budget: BudgetTracker,
        *,
        api_key: str | None = None,
        client: AsyncAnthropic | None = None,
    ) -> None:
        self._repo = decisions_repo
        self._budget = budget
        if client is not None:
            self._client = client
        else:
            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError("ANTHROPIC_API_KEY not set")
            self._client = AsyncAnthropic(api_key=key)

    async def call(
        self,
        *,
        decision_type: LlmDecisionType,
        model: str,
        system: str,
        user_payload: dict[str, Any] | str,
        max_output_tokens: int = 1024,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send a single message to Anthropic. Returns a dict with the parsed
        JSON output (or raw text under `text` if parsing failed) plus usage info.

        On budget-exceeded: the call is *not* made and the tracker raises.
        Callers handle BudgetExceeded by skipping the LLM-driven step (e.g.
        news_check fails-open to "proceed").
        """
        prompt_text = (
            json.dumps(user_payload, default=str)
            if isinstance(user_payload, dict)
            else user_payload
        )
        estimated_in = estimate_tokens(system) + estimate_tokens(prompt_text)
        await self._budget.check(
            model,
            prompt_tokens_estimate=estimated_in,
            max_output_tokens=max_output_tokens,
        )

        try:
            response = await self._client.messages.create(
                model=model,
                max_tokens=max_output_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt_text}],
            )
        except AnthropicAPIError as exc:
            log_checkpoint(
                "llm_call_fail",
                status="fail",
                model=model,
                decision_type=decision_type.value,
                error=str(exc),
            )
            raise

        text = response.content[0].text if response.content else ""
        parsed: dict[str, Any] = {}
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            parsed = {"text": text}

        cost = self._budget.cost_for(
            model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        record = LlmDecision(
            decision_type=decision_type,
            context=context or (
                user_payload if isinstance(user_payload, dict) else {"prompt": prompt_text[:2000]}
            ),
            model=model,
            response=parsed,
            decision=str(parsed.get("decision", "")) or None,
            confidence=parsed.get("confidence"),
            acted_on=None,
            outcome=None,
            created_at=_utcnow(),
            tokens_in=response.usage.input_tokens,
            tokens_out=response.usage.output_tokens,
            cost_usd=cost.total,
        )
        decision_id = await self._repo.insert(record)

        log_checkpoint(
            "llm_call",
            status="ok",
            model=model,
            decision_type=decision_type.value,
            tokens_in=response.usage.input_tokens,
            tokens_out=response.usage.output_tokens,
            cost_usd=round(cost.total, 6),
        )
        return {
            "decision_id": decision_id,
            "parsed": parsed,
            "raw_text": text,
            "tokens_in": response.usage.input_tokens,
            "tokens_out": response.usage.output_tokens,
            "cost_usd": cost.total,
        }
