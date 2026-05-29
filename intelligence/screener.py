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

import yfinance as yf

from core.broker import Broker
from core.checkpoint import checkpoint, log_checkpoint
from core.config import load_universe
from core.models import Candidate, LlmDecisionType, OptionType
from data.earnings import next_earnings
from data.ivr import IVRProvider
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


def _flatten_yf_columns(df: Any) -> Any:
    """yfinance >= 0.2.34 returns MultiIndex columns even for single-ticker
    downloads, so `df["Close"]` becomes a sub-DataFrame and `.mean()` returns
    a Series instead of a scalar. Collapse to flat columns by taking the
    field name (level 0). Mirror of the helper in risk/regime.py."""
    cols = getattr(df, "columns", None)
    if cols is not None and hasattr(cols, "levels"):
        df.columns = cols.get_level_values(0)
    return df


def _smas_from_yfinance(symbol: str) -> tuple[float | None, float | None, float | None]:
    try:
        df = yf.download(symbol, period="3mo", interval="1d", progress=False, auto_adjust=False)
    except Exception:
        return None, None, None
    if df is None or df.empty or "Close" not in df:
        return None, None, None
    df = _flatten_yf_columns(df)
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
    universe = load_universe()
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

        n_written = await _persist_candidates(repos, snapshots, result["parsed"], run_date)
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
