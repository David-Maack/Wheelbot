"""intelligence/anthropic_client — chokepoint behavior."""

from __future__ import annotations

from typing import Any

import pytest

from core.models import LlmDecisionType
from intelligence.anthropic_client import AnthropicClient
from intelligence.budget import BudgetExceeded, BudgetTracker


class _FakeUsage:
    def __init__(self, in_tokens: int, out_tokens: int):
        self.input_tokens = in_tokens
        self.output_tokens = out_tokens


class _FakeContent:
    def __init__(self, text: str):
        self.text = text


class _FakeMessage:
    def __init__(self, text: str, in_tokens: int, out_tokens: int):
        self.content = [_FakeContent(text)]
        self.usage = _FakeUsage(in_tokens, out_tokens)


class _FakeMessages:
    def __init__(self, response: _FakeMessage):
        self._response = response
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _FakeMessage:
        self.calls.append(kwargs)
        return self._response


class _FakeAnthropic:
    def __init__(self, response: _FakeMessage):
        self.messages = _FakeMessages(response)


def _config() -> dict:
    return {"intelligence": {"daily_budget_usd": 1.0}}


@pytest.mark.asyncio
async def test_call_records_usage_and_cost(db_repos):
    fake = _FakeAnthropic(_FakeMessage('{"decision":"proceed"}', in_tokens=1234, out_tokens=42))
    budget = BudgetTracker(db_repos.llm_decisions, _config())
    client = AnthropicClient(db_repos.llm_decisions, budget, client=fake)
    result = await client.call(
        decision_type=LlmDecisionType.NEWS_CHECK,
        model="claude-haiku-4-5",
        system="sys",
        user_payload={"hello": "world"},
        max_output_tokens=128,
    )
    assert result["parsed"] == {"decision": "proceed"}
    assert result["tokens_in"] == 1234
    assert result["tokens_out"] == 42
    rows = await db_repos.llm_decisions.list_recent()
    assert rows[0].cost_usd is not None and rows[0].cost_usd > 0


@pytest.mark.asyncio
async def test_call_returns_text_when_response_not_json(db_repos):
    fake = _FakeAnthropic(_FakeMessage("not even close to JSON", in_tokens=10, out_tokens=10))
    budget = BudgetTracker(db_repos.llm_decisions, _config())
    client = AnthropicClient(db_repos.llm_decisions, budget, client=fake)
    result = await client.call(
        decision_type=LlmDecisionType.NEWS_CHECK,
        model="claude-haiku-4-5",
        system="sys",
        user_payload="just text",
    )
    assert result["parsed"] == {"text": "not even close to JSON"}


@pytest.mark.asyncio
async def test_budget_exceeded_short_circuits_before_api(db_repos):
    fake = _FakeAnthropic(_FakeMessage('{"decision":"proceed"}', 1, 1))
    budget = BudgetTracker(db_repos.llm_decisions, {"intelligence": {"daily_budget_usd": 0.0}})
    client = AnthropicClient(db_repos.llm_decisions, budget, client=fake)
    with pytest.raises(BudgetExceeded):
        await client.call(
            decision_type=LlmDecisionType.NEWS_CHECK,
            model="claude-haiku-4-5",
            system="sys",
            user_payload="hi",
        )
    assert fake.messages.calls == []
