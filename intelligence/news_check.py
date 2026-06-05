"""Pre-trade news check — spec §13 #32 / §9.2.

Before placing a new CSP, ask Claude Haiku: "given these last 48h headlines on
this ticker, is there a catalyst that would make selling a 30-day put unwise?"

Decision values:
    "proceed"  — no concerning headlines; place as proposed.
    "caution"  — non-trivial signal; reduce size by 50%.
    "block"    — clear catalyst; cancel the order.

Fail-open everywhere a real production system might silently lose info:
  - News source down  → proceed (logged with status=skip).
  - Budget exhausted  → proceed (logged).
  - Malformed model   → proceed (logged).

The bias is intentional: a working risk gate plus rule-based limits should
already be doing the heavy lifting. The LLM is a last-line catalyst sniffer,
not a primary safety mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.checkpoint import log_checkpoint
from core.models import LlmDecisionType
from intelligence.anthropic_client import AnthropicClient
from intelligence.budget import BudgetExceeded, ModelNotPriced
from intelligence.news import Headline, NewsSource, NewsSourceUnavailable

VALID_DECISIONS = {"proceed", "caution", "block"}

# TICKET-014: per-strategy news_check prompts. Each profile asks the LLM
# the question that matches the position's risk shape. The wheel CSP path
# uses `bullish_csp` — that prompt is preserved BYTE-FOR-BYTE from the
# original (locked by test_wheel_csp_system_prompt_locked) because it's
# been running in paper for months. The other three profiles ship with
# this ticket and get exercised by iron_condor (and later PMCC / calendar).
#
# Decision schema is identical across all profiles so the parser stays one
# code path. Decision semantics per the runbook §6 / scope doc:
#   proceed → no fresh material catalyst, place at full size
#   caution → non-trivial signal — halve size, or skip if qty=1
#   block   → clear catalyst — cancel the order

BULLISH_CSP_PROMPT = """You are a risk reviewer. Given the last 48 hours of news
headlines for one stock ticker, decide whether selling a 30-day cash-secured put
on that ticker tomorrow morning is a reasonable idea.

Reply with exactly one JSON object, no markdown:

{
  "decision": "proceed" | "caution" | "block",
  "rationale": "<= 240 chars",
  "confidence": 0.0-1.0
}

"proceed"  — nothing in these headlines is a fresh material catalyst.
"caution"  — there is a non-trivial signal worth halving the position size for.
"block"    — there is a clear catalyst that makes the trade unwise (earnings
             surprise, FDA action, criminal probe, fraud allegations, dilution
             event, leadership crisis, etc.).
Be conservative — when in doubt, "caution".
"""

# Backwards-compat alias. External code (and the wheel-CSP regression test)
# imports SYSTEM_PROMPT. Keep it pointing at the bullish_csp content.
SYSTEM_PROMPT = BULLISH_CSP_PROMPT

BULLISH_LONG_PROMPT = """You are a risk reviewer. Given the last 48 hours of news
headlines for one stock ticker, decide whether opening a bullish long-dated call
position on that ticker tomorrow morning is a reasonable idea.

Reply with exactly one JSON object, no markdown:

{
  "decision": "proceed" | "caution" | "block",
  "rationale": "<= 240 chars",
  "confidence": 0.0-1.0
}

"proceed"  — nothing in these headlines threatens a multi-month bullish thesis.
"caution"  — there is a non-trivial signal worth halving the position size for.
"block"    — there is a clear catalyst that makes a long bullish position unwise
             (impending earnings tail risk, FDA/regulatory action, dilution
             event, leadership crisis, criminal probe, etc.).
Be conservative — when in doubt, "caution".
"""

NEUTRAL_RANGE_PROMPT = """You are a risk reviewer. Given the last 48 hours of news
headlines for one ticker, decide whether opening a neutral-range options position
(iron condor: short put spread + short call spread, defined risk) on that ticker
tomorrow morning is a reasonable idea. A neutral-range position loses if the
underlying moves SHARPLY in EITHER direction within 30-45 days.

For broad-market indices (SPY, QQQ, IWM, DIA), single-name news is less relevant;
focus on macro catalysts (FOMC decisions, CPI/NFP releases, major geopolitical
events, sovereign-rating actions).

Reply with exactly one JSON object, no markdown:

{
  "decision": "proceed" | "caution" | "block",
  "rationale": "<= 240 chars",
  "confidence": 0.0-1.0
}

"proceed"  — no catalyst likely to drive a sharp move in either direction.
"caution"  — there is a non-trivial signal worth halving the position size for.
"block"    — there is a clear catalyst likely to cause a large directional move
             (FOMC surprise, major data release inside the holding period,
             geopolitical shock, earnings inside the window for single names).
Be conservative — when in doubt, "caution".
"""

NEUTRAL_PIN_PROMPT = """You are a risk reviewer. Given the last 48 hours of news
headlines for one ticker, decide whether opening a calendar spread (short
front-month, long back-month, same strike) on that ticker tomorrow morning is a
reasonable idea. A calendar spread profits when the underlying stays NEAR the
strike through front-month expiration; it loses if the underlying moves
significantly away from the strike before the front leg expires.

Reply with exactly one JSON object, no markdown:

{
  "decision": "proceed" | "caution" | "block",
  "rationale": "<= 240 chars",
  "confidence": 0.0-1.0
}

"proceed"  — no catalyst likely to drive a significant move away from current price.
"caution"  — there is a non-trivial signal worth halving the position size for.
"block"    — there is a clear catalyst likely to move the underlying meaningfully
             away from current price before front-month expiration (FOMC, major
             data, earnings inside the front-month, geopolitical shock).
Be conservative — when in doubt, "caution".
"""

PROFILE_PROMPTS: dict[str, str] = {
    "bullish_csp": BULLISH_CSP_PROMPT,
    "bullish_long": BULLISH_LONG_PROMPT,
    "neutral_range": NEUTRAL_RANGE_PROMPT,
    "neutral_pin": NEUTRAL_PIN_PROMPT,
}


@dataclass(frozen=True, slots=True)
class NewsCheckResult:
    decision: str  # "proceed" | "caution" | "block"
    rationale: str
    source: str  # "llm" | "skipped:<reason>"
    confidence: float | None = None


def _normalize(decision: Any) -> str:
    s = str(decision or "").strip().lower()
    return s if s in VALID_DECISIONS else "proceed"


async def news_check(
    *,
    symbol: str,
    news: NewsSource | None,
    anthropic: AnthropicClient,
    config: dict[str, Any],
    profile: str = "bullish_csp",
) -> NewsCheckResult:
    """Run the pre-trade check. Always returns a result — never raises.

    `profile` selects which prompt to use; defaults to `bullish_csp` for
    backwards-compat with the wheel CSP path. Unknown profiles fall back
    to bullish_csp (defensive — surface as a warning instead of crashing
    if a new profile is referenced before its prompt is added)."""
    intel = config.get("intelligence", {}) or {}
    if not bool(intel.get("llm_news_check_enabled", True)):
        return NewsCheckResult("proceed", "news_check disabled in config", "skipped:disabled")
    if news is None:
        log_checkpoint("news_check_no_source", status="skip", symbol=symbol)
        return NewsCheckResult("proceed", "no news source configured", "skipped:no_source")

    headlines: list[Headline] = []
    try:
        headlines = await news.recent(symbol, hours=48)
    except NewsSourceUnavailable as exc:
        log_checkpoint("news_check_source_skip", status="skip", symbol=symbol, error=str(exc))
        return NewsCheckResult("proceed", f"news source unavailable: {exc}", "skipped:news_unavailable")

    if not headlines:
        log_checkpoint("news_check_no_headlines", status="ok", symbol=symbol)
        return NewsCheckResult("proceed", "no recent headlines", "skipped:no_headlines")

    payload = {
        "symbol": symbol,
        "headlines": [
            {
                "headline": h.headline,
                "summary": (h.summary or "")[:500],
                "published_at": h.published_at.isoformat(),
                "source": h.source,
            }
            for h in headlines[:15]
        ],
    }
    model = intel.get("news_check_model", "claude-haiku-4-5")
    system_prompt = PROFILE_PROMPTS.get(profile)
    if system_prompt is None:
        log_checkpoint(
            "news_check_unknown_profile",
            status="fail",
            symbol=symbol,
            profile=profile,
            note="falling back to bullish_csp",
        )
        system_prompt = BULLISH_CSP_PROMPT
    try:
        result = await anthropic.call(
            decision_type=LlmDecisionType.NEWS_CHECK,
            model=model,
            system=system_prompt,
            user_payload=payload,
            max_output_tokens=int(intel.get("news_check_max_tokens", 256)),
            context={"symbol": symbol, "n_headlines": len(headlines), "profile": profile},
        )
    except BudgetExceeded as exc:
        log_checkpoint("news_check_budget_skip", status="skip", symbol=symbol, error=str(exc))
        return NewsCheckResult("proceed", "budget exhausted; degrading", "skipped:budget")
    except ModelNotPriced as exc:
        # Loud log — fail-open per news_check policy, but this is a config error
        # (typo'd model name or deprecated model) that needs attention.
        log_checkpoint(
            "news_check_model_not_priced",
            status="fail",
            symbol=symbol,
            model=model,
            error=str(exc),
            note="config error: model missing from pricing table — fix and redeploy",
        )
        return NewsCheckResult(
            "proceed",
            f"news_check_model {model!r} not in pricing table; degrading",
            "skipped:model_not_priced",
        )
    except Exception as exc:
        log_checkpoint("news_check_llm_fail", status="fail", symbol=symbol, error=str(exc))
        return NewsCheckResult("proceed", f"LLM call failed: {exc}", "skipped:llm_error")

    parsed = result["parsed"] or {}
    decision = _normalize(parsed.get("decision"))
    return NewsCheckResult(
        decision=decision,
        rationale=str(parsed.get("rationale", "")),
        source="llm",
        confidence=parsed.get("confidence"),
    )
