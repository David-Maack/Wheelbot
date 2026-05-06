"""intelligence/budget — pricing, daily total, refusal at cap."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from core.models import LlmDecision, LlmDecisionType
from intelligence.budget import BudgetExceeded, BudgetTracker


def _utc() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _config(cap: float = 1.0, **overrides) -> dict:
    base = {"intelligence": {"daily_budget_usd": cap}}
    base["intelligence"].update(overrides)
    return base


@pytest.mark.asyncio
async def test_pricing_defaults_for_known_models(db_repos):
    tracker = BudgetTracker(db_repos.llm_decisions, _config())
    assert tracker.price("claude-opus-4-7")["input_per_mtok"] == pytest.approx(15.0)
    assert tracker.price("claude-haiku-4-5")["output_per_mtok"] == pytest.approx(5.0)


@pytest.mark.asyncio
async def test_cost_for_arithmetic(db_repos):
    tracker = BudgetTracker(db_repos.llm_decisions, _config())
    cost = tracker.cost_for("claude-haiku-4-5", input_tokens=1_000_000, output_tokens=500_000)
    # 1M in × $1 + 0.5M out × $5 = $1 + $2.5 = $3.50
    assert cost.total == pytest.approx(3.5)


@pytest.mark.asyncio
async def test_total_today_sums_today_only(db_repos):
    tracker = BudgetTracker(db_repos.llm_decisions, _config())
    await db_repos.llm_decisions.insert(
        LlmDecision(decision_type=LlmDecisionType.SCREEN, created_at=_utc(), cost_usd=0.25)
    )
    await db_repos.llm_decisions.insert(
        LlmDecision(decision_type=LlmDecisionType.NEWS_CHECK, created_at=_utc(), cost_usd=0.10)
    )
    assert await tracker.total_today() == pytest.approx(0.35)


@pytest.mark.asyncio
async def test_check_passes_when_under_budget(db_repos):
    tracker = BudgetTracker(db_repos.llm_decisions, _config(cap=1.0))
    await tracker.check(
        "claude-haiku-4-5",
        prompt_tokens_estimate=1000,
        max_output_tokens=500,
    )  # ~$0.001 + $0.0025 — well under $1


@pytest.mark.asyncio
async def test_check_raises_when_projected_exceeds_cap(db_repos):
    tracker = BudgetTracker(db_repos.llm_decisions, _config(cap=0.001))
    with pytest.raises(BudgetExceeded):
        await tracker.check(
            "claude-opus-4-7",
            prompt_tokens_estimate=10_000,
            max_output_tokens=2_000,
        )


@pytest.mark.asyncio
async def test_pricing_overrides_in_config(db_repos):
    tracker = BudgetTracker(
        db_repos.llm_decisions,
        _config(pricing={"claude-haiku-4-5": {"input_per_mtok": 99.0}}),
    )
    assert tracker.price("claude-haiku-4-5")["input_per_mtok"] == pytest.approx(99.0)
    # Output price falls back to default.
    assert tracker.price("claude-haiku-4-5")["output_per_mtok"] == pytest.approx(5.0)
