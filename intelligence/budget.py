"""Daily LLM spend gate.

Wheelbot consults `BudgetTracker.check(model, max_output_tokens, prompt_tokens)`
before every Anthropic call. If today's running total + this call's
*worst-case* cost exceeds `intelligence.daily_budget_usd`, the tracker raises
`BudgetExceeded` and the LLM call is skipped.

Cost is computed from the SDK's `usage` object after the call returns and
written to `llm_decisions.cost_usd`. The next budget check sums those rows.

Default prices (per spec config defaults):

    claude-opus-4-7      $15.00 / $75.00 per 1M tokens (in / out)
    claude-haiku-4-5      $1.00 /  $5.00 per 1M tokens

Override via `intelligence.pricing.<model>.{input_per_mtok, output_per_mtok}`
in config.yaml when Anthropic adjusts pricing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from core.notify import notify
from db.repo import LlmDecisionsRepo


DEFAULT_PRICING_USD_PER_MTOK = {
    "claude-opus-4-7":  {"input_per_mtok": 15.0, "output_per_mtok": 75.0},
    "claude-sonnet-4-6": {"input_per_mtok": 3.0,  "output_per_mtok": 15.0},
    "claude-haiku-4-5": {"input_per_mtok": 1.0,  "output_per_mtok": 5.0},
}


class BudgetExceeded(Exception):
    """Raised pre-call when the daily cap would be breached."""


@dataclass(frozen=True, slots=True)
class CostEstimate:
    input_cost_usd: float
    output_cost_usd: float

    @property
    def total(self) -> float:
        return self.input_cost_usd + self.output_cost_usd


class BudgetTracker:
    def __init__(
        self,
        decisions_repo: LlmDecisionsRepo,
        config: dict[str, Any],
    ) -> None:
        self._repo = decisions_repo
        intel = config.get("intelligence", {}) or {}
        self._daily_cap = float(intel.get("daily_budget_usd", 1.0))
        overrides = (intel.get("pricing") or {})
        self._pricing: dict[str, dict[str, float]] = {**DEFAULT_PRICING_USD_PER_MTOK}
        for model, prices in overrides.items():
            base = self._pricing.get(model, {"input_per_mtok": 0.0, "output_per_mtok": 0.0})
            self._pricing[model] = {**base, **prices}
        self._notified_for_date: date | None = None

    @property
    def daily_cap_usd(self) -> float:
        return self._daily_cap

    def price(self, model: str) -> dict[str, float]:
        return self._pricing.get(
            model, {"input_per_mtok": 0.0, "output_per_mtok": 0.0}
        )

    def cost_for(
        self, model: str, *, input_tokens: int, output_tokens: int
    ) -> CostEstimate:
        p = self.price(model)
        return CostEstimate(
            input_cost_usd=(input_tokens / 1_000_000) * p["input_per_mtok"],
            output_cost_usd=(output_tokens / 1_000_000) * p["output_per_mtok"],
        )

    async def total_today(self) -> float:
        return await self._repo.total_cost_today()

    async def check(
        self,
        model: str,
        *,
        prompt_tokens_estimate: int,
        max_output_tokens: int,
    ) -> None:
        """Raise BudgetExceeded if running this call could breach the cap.

        Worst-case cost = input_tokens × in_price + max_output_tokens × out_price.
        """
        spent = await self.total_today()
        worst_case = self.cost_for(
            model,
            input_tokens=prompt_tokens_estimate,
            output_tokens=max_output_tokens,
        ).total
        if spent + worst_case > self._daily_cap:
            today = datetime.now(UTC).date()
            if self._notified_for_date != today:
                self._notified_for_date = today
                await notify(
                    "risk.budget_exceeded",
                    "Daily LLM budget hit",
                    cap_usd=self._daily_cap,
                    spent_usd=round(spent, 4),
                    blocked_call_model=model,
                )
            raise BudgetExceeded(
                f"projected ${spent + worst_case:.4f} exceeds daily cap "
                f"${self._daily_cap:.2f} (already spent ${spent:.4f})"
            )
