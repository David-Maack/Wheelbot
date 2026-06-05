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


# -- TICKET-014 Phase 2 step 1: wheel-CSP regression lock ------------------
#
# Written FIRST against current behavior. Must pass before any news_check
# generalization touches the module, and MUST CONTINUE to pass after the
# bullish_csp/bullish_long/neutral_range/neutral_pin profile dispatch
# lands. The wheel CSP path has been running in paper for months; this
# pins its contract byte-for-byte so the generalization can't silently
# regress it.


def test_wheel_csp_system_prompt_locked():
    """The bullish_csp prompt is what the wheel has been running. Pin its
    anchor phrases so a generalization refactor can't silently rewrite it.
    If you legitimately need to change the prompt, update this test in the
    same commit and document the change in the runbook."""
    from intelligence.news_check import SYSTEM_PROMPT

    # Anchor: this is the CSP-specific prompt, not a generic one.
    assert "30-day cash-secured put" in SYSTEM_PROMPT
    # Anchor: tomorrow-morning entry context (vs. a generic "this trade").
    assert "tomorrow morning is a reasonable idea" in SYSTEM_PROMPT
    # Decision schema — exact strings the parser normalizes on.
    assert '"decision": "proceed" | "caution" | "block"' in SYSTEM_PROMPT
    assert '"rationale": "<= 240 chars"' in SYSTEM_PROMPT
    assert '"confidence": 0.0-1.0' in SYSTEM_PROMPT
    # The three definition lines — each one a specific phrase the model
    # has been trained against in production.
    assert "fresh material catalyst" in SYSTEM_PROMPT
    assert "halving the position size" in SYSTEM_PROMPT
    assert "makes the trade unwise" in SYSTEM_PROMPT
    # Conservative bias anchor.
    assert 'when in doubt, "caution"' in SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_wheel_csp_decision_path_unchanged():
    """Full happy-path round trip on the wheel-CSP contract: configured news
    source returns headlines, LLM returns a structured decision, function
    returns NewsCheckResult with the expected shape. Locks the result
    fields the wheel orchestrator depends on."""
    anthropic = _StubAnthropic({
        "decision": "caution",
        "rationale": "Earnings next week may cause IV crush",
        "confidence": 0.7,
    })
    result = await news_check(
        symbol="F",
        news=_FakeNews([_h("Ford Q4 earnings preview")]),
        anthropic=anthropic,
        config={"intelligence": {}},
    )
    # Decision normalized lowercase, one of {proceed, caution, block}.
    assert result.decision == "caution"
    # Rationale carried through verbatim.
    assert result.rationale == "Earnings next week may cause IV crush"
    # Source distinguishes LLM vs skipped:*.
    assert result.source == "llm"
    # Confidence preserved when present.
    assert result.confidence == 0.7
    # LLM was actually called once.
    assert anthropic.calls == 1


@pytest.mark.asyncio
async def test_wheel_csp_fail_open_sources_locked():
    """All seven fail-open source tags must be exactly these strings — the
    router and dashboard branch on them. A rename in the generalization
    would silently break advisory-mode handling."""
    # Disabled
    r = await news_check(
        symbol="F", news=_FakeNews([_h("x")]), anthropic=_StubAnthropic({}),
        config={"intelligence": {"llm_news_check_enabled": False}},
    )
    assert r.source == "skipped:disabled"
    # No source
    r = await news_check(
        symbol="F", news=None, anthropic=_StubAnthropic({}),
        config={"intelligence": {}},
    )
    assert r.source == "skipped:no_source"
    # Source unavailable
    r = await news_check(
        symbol="F", news=_FakeNews(NewsSourceUnavailable("429")),
        anthropic=_StubAnthropic({}), config={"intelligence": {}},
    )
    assert r.source == "skipped:news_unavailable"
    # No headlines
    r = await news_check(
        symbol="F", news=_FakeNews([]), anthropic=_StubAnthropic({}),
        config={"intelligence": {}},
    )
    assert r.source == "skipped:no_headlines"
    # LLM error
    r = await news_check(
        symbol="F", news=_FakeNews([_h("x")]),
        anthropic=_StubAnthropic(RuntimeError("boom")),
        config={"intelligence": {}},
    )
    assert r.source == "skipped:llm_error"


# -- TICKET-014 Phase 2: profile dispatch ----------------------------------


def test_profile_prompts_registered():
    """All four profiles named in the scope doc must exist in PROFILE_PROMPTS."""
    from intelligence.news_check import PROFILE_PROMPTS
    assert set(PROFILE_PROMPTS.keys()) == {
        "bullish_csp", "bullish_long", "neutral_range", "neutral_pin",
    }


def test_bullish_csp_profile_is_byte_identical_to_legacy_prompt():
    """The wheel-CSP regression contract: PROFILE_PROMPTS['bullish_csp'] is
    the EXACT same string as the legacy SYSTEM_PROMPT export. The wheel
    path imports both depending on caller; if these drift the wheel's
    LLM call subtly changes."""
    from intelligence.news_check import PROFILE_PROMPTS, SYSTEM_PROMPT, BULLISH_CSP_PROMPT
    assert PROFILE_PROMPTS["bullish_csp"] == SYSTEM_PROMPT
    assert PROFILE_PROMPTS["bullish_csp"] == BULLISH_CSP_PROMPT


def test_each_profile_has_distinct_prompt_anchors():
    """Each profile must contain its distinguishing phrase. If a refactor
    points all profiles at the same prompt by mistake, this catches it."""
    from intelligence.news_check import PROFILE_PROMPTS
    assert "cash-secured put" in PROFILE_PROMPTS["bullish_csp"]
    assert "long-dated call" in PROFILE_PROMPTS["bullish_long"]
    assert "iron condor" in PROFILE_PROMPTS["neutral_range"]
    assert "calendar spread" in PROFILE_PROMPTS["neutral_pin"]


@pytest.mark.asyncio
async def test_profile_dispatch_sends_right_prompt_to_anthropic():
    """The profile parameter routes the call to the right system prompt."""
    captured: dict[str, str] = {}

    class _CapturingAnthropic:
        calls = 0
        async def call(self, **kwargs):
            self.__class__.calls += 1
            captured["system"] = kwargs.get("system", "")
            captured["profile"] = kwargs.get("context", {}).get("profile")
            return {
                "parsed": {"decision": "proceed"}, "raw_text": "",
                "tokens_in": 1, "tokens_out": 1, "cost_usd": 0.0,
                "decision_id": 1,
            }

    for profile_name, anchor in [
        ("bullish_csp", "cash-secured put"),
        ("bullish_long", "long-dated call"),
        ("neutral_range", "iron condor"),
        ("neutral_pin", "calendar spread"),
    ]:
        captured.clear()
        await news_check(
            symbol="F",
            news=_FakeNews([_h("x")]),
            anthropic=_CapturingAnthropic(),
            config={"intelligence": {}},
            profile=profile_name,
        )
        assert anchor in captured["system"], (
            f"profile={profile_name!r} should dispatch to a prompt containing "
            f"{anchor!r}, got: {captured['system'][:200]}"
        )
        assert captured["profile"] == profile_name, (
            f"profile context not threaded through for {profile_name}"
        )


@pytest.mark.asyncio
async def test_unknown_profile_falls_back_to_bullish_csp():
    """Defensive: an unknown profile name doesn't crash. Falls back to
    bullish_csp and logs a warning checkpoint."""
    captured: dict[str, str] = {}

    class _CapturingAnthropic:
        async def call(self, **kwargs):
            captured["system"] = kwargs.get("system", "")
            return {
                "parsed": {"decision": "proceed"}, "raw_text": "",
                "tokens_in": 1, "tokens_out": 1, "cost_usd": 0.0,
                "decision_id": 1,
            }

    result = await news_check(
        symbol="F",
        news=_FakeNews([_h("x")]),
        anthropic=_CapturingAnthropic(),
        config={"intelligence": {}},
        profile="nonsense_profile_name",
    )
    assert result.decision == "proceed"
    assert "cash-secured put" in captured["system"]  # bullish_csp fallback


@pytest.mark.asyncio
async def test_default_profile_is_bullish_csp():
    """When `profile` is not passed, news_check uses bullish_csp — locks the
    wheel-CSP call site behavior (which never passes profile)."""
    captured: dict[str, str] = {}

    class _CapturingAnthropic:
        async def call(self, **kwargs):
            captured["system"] = kwargs.get("system", "")
            return {
                "parsed": {"decision": "proceed"}, "raw_text": "",
                "tokens_in": 1, "tokens_out": 1, "cost_usd": 0.0,
                "decision_id": 1,
            }

    await news_check(
        symbol="F",
        news=_FakeNews([_h("x")]),
        anthropic=_CapturingAnthropic(),
        config={"intelligence": {}},
    )
    assert "cash-secured put" in captured["system"]
