"""Daily LLM screener — spec §13 #31.

For each tier-1 / tier-2 ticker (skip banned, skip tier-3 unless explicitly
enabled) gather the structured payload from §9.1:

  current price, IVR, IV percentile, recent price action (5/20/50 SMAs),
  upcoming earnings date, recent news headlines (last 7 days)

Send one Opus call with the whole payload, ask for a ranked JSON shortlist
with rationale per ticker. Parse, write top-N rows to `candidates`.

Output JSON contract (the model is instructed to produce exactly this):

    {
      "decision": "screen_complete",
      "candidates": [
        {
          "rank": 1,
          "symbol": "F",
          "score": 78.2,
          "rationale": "..."
        }
      ]
    }

Hard rule (spec §9.1): screener output is a *suggestion*. Risk gates and tier
rules still apply downstream. Tier 2 candidates marked acted_on=False until a
human or the wheel orchestrator decides.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

from core.broker import Broker
from core.checkpoint import checkpoint, log_checkpoint
from core.models import Candidate, LlmDecisionType, OptionType
from core.watchlists import effective_universe
from data.earnings import next_earnings
from data.ivr import IVRProvider
from data.yf_helpers import safe_history
from db.repo import Repos
from intelligence.anthropic_client import AnthropicClient
from intelligence.budget import BudgetExceeded
from intelligence.news import NewsSource, NewsSourceUnavailable

SYSTEM_PROMPT = """You are a quantitative analyst screening an options-wheel
trading universe. The user hands you a JSON payload of tickers with current
metrics; you rank them.

Pick candidates that are good fits for selling 30-45 DTE cash-secured puts at
delta 0.20-0.30 with the goal of either collecting premium or being assigned
shares the user is willing to hold.

Strong candidates have: high IVR (premium is rich), liquid options market,
no upcoming earnings inside the typical expiry window, no acute negative news
catalysts in the last 7 days, technical posture that wouldn't punish assignment
(e.g. SMA stack not screaming "knife falling").

Weak candidates have: low IVR (cheap premium not worth the risk), upcoming
earnings inside the expiry window, fresh negative catalyst headlines, broken
technical structure.

Return ONLY a JSON object. Do not wrap in markdown. The exact shape is:

{
  "decision": "screen_complete",
  "candidates": [
    {"rank": 1, "symbol": "TICKER", "score": 0-100, "rationale": "<= 240 chars"}
  ]
}

`score` is your conviction 0-100. Lower-conviction picks should still be ranked
but with lower scores. Include all input tickers — never silently drop one.
"""


@dataclass(slots=True)
class TickerSnapshot:
    symbol: str
    tier: int
    price: float | None = None
    ivr: float | None = None
    iv_pct: float | None = None
    sma_5: float | None = None
    sma_20: float | None = None
    sma_50: float | None = None
    next_earnings: str | None = None
    headlines: list[str] = field(default_factory=list)


async def _fetch_snapshot(
    broker: Broker,
    ivr: IVRProvider,
    news: NewsSource | None,
    symbol: str,
    tier: int,
) -> TickerSnapshot:
    snap = TickerSnapshot(symbol=symbol, tier=tier)

    try:
        quote = await broker.get_quote(symbol)
        snap.price = quote.mid if quote.mid is not None else (quote.last or quote.bid or quote.ask)
    except Exception:
        snap.price = None

    stats = await ivr.stats(symbol)
    if stats is not None:
        snap.ivr = stats.rank
        snap.iv_pct = stats.percentile

    snap.sma_5, snap.sma_20, snap.sma_50 = _smas_from_yfinance(symbol)

    earnings = next_earnings(symbol)
    snap.next_earnings = str(earnings.next_date) if earnings.next_date else None

    if news is not None:
        try:
            headlines = await news.recent(symbol, hours=7 * 24)
            snap.headlines = [h.headline for h in headlines[:8]]
        except NewsSourceUnavailable as exc:
            log_checkpoint("screener_news_skip", status="skip", symbol=symbol, error=str(exc))
    return snap


def _smas_from_yfinance(symbol: str) -> tuple[float | None, float | None, float | None]:
    df = safe_history(symbol, period="3mo")
    if df.empty or "Close" not in df:
        return None, None, None
    close = df["Close"].dropna()
    s5 = float(close.tail(5).mean()) if len(close) >= 5 else None
    s20 = float(close.tail(20).mean()) if len(close) >= 20 else None
    s50 = float(close.tail(50).mean()) if len(close) >= 50 else None
    return s5, s20, s50


def _payload_from(snapshots: list[TickerSnapshot]) -> dict[str, Any]:
    return {
        "tickers": [
            {
                "symbol": s.symbol,
                "tier": s.tier,
                "price": s.price,
                "ivr": s.ivr,
                "iv_percentile": s.iv_pct,
                "sma_5": s.sma_5,
                "sma_20": s.sma_20,
                "sma_50": s.sma_50,
                "next_earnings": s.next_earnings,
                "headlines": s.headlines,
            }
            for s in snapshots
        ]
    }


ADVERSARIAL_SYSTEM_PROMPT = """You are the risk review desk for an options
screener. For EACH candidate below, argue the strongest BULL case, then the
strongest BEAR case, then give a conviction adjustment between -15 and +15
(negative = the bear case dominates). Be adversarial with the given score —
your job is to catch single-pass overconfidence, not to agree.

Return ONLY JSON: {"reviews": [{"symbol": "T", "bull_case": "<=120 chars",
"bear_case": "<=120 chars", "adjust": -15..15}]}"""


def _apply_adversarial(parsed: dict[str, Any], reviews: Any) -> dict[str, Any]:
    """Apply clamped score adjustments + annotate rationales. Pure, fail-open:
    malformed reviews leave the original parsed untouched."""
    if not isinstance(reviews, list):
        return parsed
    by_symbol = {}
    for r in reviews:
        if isinstance(r, dict) and r.get("symbol"):
            by_symbol[str(r["symbol"]).upper()] = r
    for item in parsed.get("candidates") or []:
        if not isinstance(item, dict):
            continue
        r = by_symbol.get(str(item.get("symbol", "")).upper())
        if r is None:
            continue
        try:
            adjust = max(-15.0, min(15.0, float(r.get("adjust", 0))))
        except (TypeError, ValueError):
            continue
        if item.get("score") is not None:
            item["score"] = max(0.0, min(100.0, float(item["score"]) + adjust))
        bear = str(r.get("bear_case") or "")[:120]
        if bear:
            item["rationale"] = f"{item.get('rationale') or ''} | adv({adjust:+.0f}): {bear}"[:240]
    return parsed


async def _adversarial_pass(
    anthropic: AnthropicClient, intel: dict[str, Any], parsed: dict[str, Any], run_date: date,
) -> dict[str, Any]:
    """AI audit item #5 (the one TradingAgents component worth stealing): a
    bull/bear adversarial second read of the screener's TOP-3, adjusting scores
    ±15. One extra cheap call; every failure fails open to the original scores."""
    try:
        cands = [c for c in (parsed.get("candidates") or [])
                 if isinstance(c, dict) and c.get("score") is not None]
        top = sorted(cands, key=lambda c: float(c["score"]), reverse=True)[:3]
        if not top:
            return parsed
        result = await anthropic.call(
            decision_type=LlmDecisionType.SCREEN,
            model=str(intel.get("adversarial_model", "claude-haiku-4-5")),
            system=ADVERSARIAL_SYSTEM_PROMPT,
            user_payload={"candidates": [
                {"symbol": c.get("symbol"), "score": c.get("score"),
                 "rationale": c.get("rationale")} for c in top
            ]},
            max_output_tokens=600,
            context={"run_date": str(run_date), "adversarial": True},
        )
        return _apply_adversarial(parsed, (result.get("parsed") or {}).get("reviews"))
    except Exception as exc:  # noqa: BLE001 — advisory layer: never break the screen
        log_checkpoint("screener_adversarial_fail", status="fail", error=str(exc))
        return parsed


async def _persist_candidates(
    repos: Repos,
    snapshots: list[TickerSnapshot],
    parsed: dict[str, Any],
    run_date: date,
) -> int:
    by_symbol = {s.symbol.upper(): s for s in snapshots}
    raw = parsed.get("candidates") or []
    if not isinstance(raw, list):
        return 0
    written = 0
    for item in raw:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", "")).upper()
        snap = by_symbol.get(symbol)
        if snap is None:
            continue
        await repos.candidates.insert(
            Candidate(
                run_date=run_date,
                symbol=symbol,
                score=item.get("score"),
                rank=item.get("rank"),
                rationale=item.get("rationale"),
                ivr=snap.ivr,
                iv_pct=snap.iv_pct,
                price=snap.price,
                bp_required=(snap.price * 100) if snap.price else None,
                suggested_strike=None,
                suggested_dte=None,
                raw_llm_response=item,
                acted_on=False,
            )
        )
        written += 1
    return written


async def run_screener(
    *,
    broker: Broker,
    repos: Repos,
    ivr: IVRProvider,
    news: NewsSource | None,
    anthropic: AnthropicClient,
    config: dict[str, Any],
    run_date: date | None = None,
) -> dict[str, Any]:
    intel = config.get("intelligence", {}) or {}
    if not bool(intel.get("llm_screener_enabled", True)):
        log_checkpoint("screener_disabled", status="skip")
        return {"skipped": "disabled"}

    run_date = run_date or datetime.now(UTC).date()
    # Watchlist-aware: symbols the universe refresh added (synthesized at
    # tier 2) must get a daily screener row, or the tier2_screen risk-gate rule
    # would block them forever. Falls back to plain universe.yaml when the
    # refresh feature is disabled or nothing is applied.
    universe = await effective_universe(repos, config)
    tickers = [t for t in universe["tickers"] if t.tier in (1, 2)]
    if not tickers:
        log_checkpoint("screener_no_tickers", status="skip")
        return {"skipped": "no_tickers"}

    with checkpoint("screener_run", n_tickers=len(tickers)) as ctx:
        snapshots = await asyncio.gather(
            *(_fetch_snapshot(broker, ivr, news, t.symbol, t.tier) for t in tickers)
        )
        payload = _payload_from(snapshots)
        model = intel.get("screener_model", "claude-opus-4-7")
        try:
            result = await anthropic.call(
                decision_type=LlmDecisionType.SCREEN,
                model=model,
                system=SYSTEM_PROMPT,
                user_payload=payload,
                max_output_tokens=int(intel.get("screener_max_tokens", 2048)),
                context={"run_date": str(run_date)},
            )
        except BudgetExceeded as exc:
            log_checkpoint("screener_budget_skip", status="skip", error=str(exc))
            return {"skipped": "budget_exceeded"}

        parsed = result["parsed"]
        # AI audit item #5: adversarial bull/bear second pass on the top-3
        # (gated; fails open to the single-pass scores).
        if bool(intel.get("adversarial_screener_enabled", False)):
            parsed = await _adversarial_pass(anthropic, intel, parsed, run_date)
        n_written = await _persist_candidates(repos, snapshots, parsed, run_date)
        ctx["candidates_written"] = n_written
        ctx["cost_usd"] = round(result["cost_usd"], 4)
        # A successful API call that yields ZERO candidates is a silent failure:
        # the model returned non-JSON / truncated output (parsed falls back to
        # {"text": ...}), or an empty list. Without this, every tier-2 entry the
        # NEXT day fails the screen gate ("no LLM screener row") with no obvious
        # cause. Surface it loudly so the cron's exit/log makes the problem visible.
        if n_written == 0:
            parsed = result.get("parsed") or {}
            log_checkpoint(
                "screener_zero_candidates",
                status="fail",
                reason="model returned no parseable candidates",
                had_text_fallback="text" in parsed,
                raw_preview=str(result.get("raw_text", ""))[:200],
            )
        return {
            "candidates_written": n_written,
            "cost_usd": result["cost_usd"],
            "decision_id": result["decision_id"],
        }
