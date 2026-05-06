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
from intelligence.budget import BudgetExceeded
from intelligence.news import Headline, NewsSource, NewsSourceUnavailable

VALID_DECISIONS = {"proceed", "caution", "block"}

SYSTEM_PROMPT = """You are a risk reviewer. Given the last 48 hours of news
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
) -> NewsCheckResult:
    """Run the pre-trade check. Always returns a result — never raises."""
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
    try:
        result = await anthropic.call(
            decision_type=LlmDecisionType.NEWS_CHECK,
            model=model,
            system=SYSTEM_PROMPT,
            user_payload=payload,
            max_output_tokens=int(intel.get("news_check_max_tokens", 256)),
            context={"symbol": symbol, "n_headlines": len(headlines)},
        )
    except BudgetExceeded as exc:
        log_checkpoint("news_check_budget_skip", status="skip", symbol=symbol, error=str(exc))
        return NewsCheckResult("proceed", "budget exhausted; degrading", "skipped:budget")
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
