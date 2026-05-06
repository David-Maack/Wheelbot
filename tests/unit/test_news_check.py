"""intelligence/news_check — fail-open + decision parsing."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from intelligence.news import Headline, NewsSource, NewsSourceUnavailable
from intelligence.news_check import news_check


class _StubAnthropic:
    """Returns a pre-set parsed response. Tracks call count."""

    def __init__(self, response: dict[str, Any] | Exception):
        self._response = response
        self.calls = 0

    async def call(self, **kwargs):
        self.calls += 1
        if isinstance(self._response, Exception):
            raise self._response
        return {
            "parsed": self._response,
            "raw_text": "",
            "tokens_in": 1,
            "tokens_out": 1,
            "cost_usd": 0.0001,
            "decision_id": 1,
        }


class _FakeNews(NewsSource):
    def __init__(self, headlines: list[Headline] | Exception):
        self._headlines = headlines

    @property
    def name(self) -> str:
        return "fake"

    async def recent(self, symbol, *, hours=48):
        if isinstance(self._headlines, Exception):
            raise self._headlines
        return self._headlines


def _h(text: str) -> Headline:
    return Headline(
        symbol="F",
        headline=text,
        summary=None,
        url=None,
        published_at=datetime.now(UTC).replace(tzinfo=None),
        source="fake",
    )


@pytest.mark.asyncio
async def test_no_news_source_proceeds():
    result = await news_check(
        symbol="F", news=None, anthropic=None, config={"intelligence": {}}
    )
    assert result.decision == "proceed"
    assert "no news source" in result.rationale.lower()


@pytest.mark.asyncio
async def test_news_source_rate_limited_proceeds():
    news = _FakeNews(NewsSourceUnavailable("429"))
    result = await news_check(
        symbol="F", news=news, anthropic=_StubAnthropic({}), config={"intelligence": {}}
    )
    assert result.decision == "proceed"
    assert result.source == "skipped:news_unavailable"


@pytest.mark.asyncio
async def test_no_headlines_proceeds_without_calling_llm():
    anthropic = _StubAnthropic({"decision": "block"})
    result = await news_check(
        symbol="F", news=_FakeNews([]), anthropic=anthropic, config={"intelligence": {}}
    )
    assert result.decision == "proceed"
    assert anthropic.calls == 0


@pytest.mark.asyncio
async def test_block_decision_propagates():
    anthropic = _StubAnthropic({"decision": "block", "rationale": "FDA warning", "confidence": 0.9})
    result = await news_check(
        symbol="F",
        news=_FakeNews([_h("FDA warns of safety issue")]),
        anthropic=anthropic,
        config={"intelligence": {}},
    )
    assert result.decision == "block"
    assert "FDA" in result.rationale


@pytest.mark.asyncio
async def test_caution_decision_propagates():
    anthropic = _StubAnthropic({"decision": "caution", "rationale": "guidance miss"})
    result = await news_check(
        symbol="F",
        news=_FakeNews([_h("Q4 guidance misses")]),
        anthropic=anthropic,
        config={"intelligence": {}},
    )
    assert result.decision == "caution"


@pytest.mark.asyncio
async def test_malformed_response_falls_back_to_proceed():
    anthropic = _StubAnthropic({"text": "not parseable"})  # missing decision key
    result = await news_check(
        symbol="F",
        news=_FakeNews([_h("noise")]),
        anthropic=anthropic,
        config={"intelligence": {}},
    )
    assert result.decision == "proceed"


@pytest.mark.asyncio
async def test_llm_failure_falls_back_to_proceed():
    anthropic = _StubAnthropic(RuntimeError("boom"))
    result = await news_check(
        symbol="F",
        news=_FakeNews([_h("noise")]),
        anthropic=anthropic,
        config={"intelligence": {}},
    )
    assert result.decision == "proceed"
    assert result.source == "skipped:llm_error"


@pytest.mark.asyncio
async def test_disabled_in_config_proceeds_without_anything():
    result = await news_check(
        symbol="F",
        news=_FakeNews([_h("noise")]),
        anthropic=_StubAnthropic({"decision": "block"}),
        config={"intelligence": {"llm_news_check_enabled": False}},
    )
    assert result.decision == "proceed"
    assert result.source == "skipped:disabled"
