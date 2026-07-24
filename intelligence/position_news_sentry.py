"""Position news sentry — mid-cycle headline check on OPEN positions.

The entry-time news_check runs once, at order placement, for wheel CSPs only.
Once a position is open the bot is news-blind: management is purely mechanical
(profit %, DTE, stop multiple) and the only mid-cycle event awareness is the
earnings recheck — a SCHEDULED catalyst. This module is the unscheduled
counterpart: every ~hour, each open position's recent headlines get a
position-aware Haiku read.

Design stance (deliberate, evidence-backed):

    NOTIFY-ONLY. The sentry never places, sizes, or closes anything.

The Sprint-2 stop-loss backtest showed reactive early exits on defined-risk
structures destroy expectancy (58% of 2x stop-outs recovered by the DTE-21
close), and LLM headline sentiment tops out ~70-75% accurate — a signal that's
wrong one time in four, firing on the most volatile days, must not realize
losses autonomously. So: `alert` / `exit_advisory` verdicts go to Discord
(rate-limited) and the operator decides — one tap via MCP flatten_position.
Every verdict is logged to llm_decisions with the position's cycle_id so a
future hit-rate report (the news_hit_rate pattern) can measure whether
exit_advisory positions actually end worse. Autonomous action is earned with
that evidence or not at all.

Fail-open contract: any failure (no source, no key, budget, model error,
malformed reply) degrades to `hold` for that position and never breaks the
tick loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from core.checkpoint import log_checkpoint
from core.models import LlmDecisionType, Order, OrderStatus, Position, PositionState
from core.notify import notify
from db.repo import Repos
from intelligence.anthropic_client import AnthropicClient
from intelligence.budget import BudgetExceeded, ModelNotPriced
from intelligence.news import NewsSource, NewsSourceUnavailable

VERDICT_HOLD = "hold"
VERDICT_ALERT = "alert"
VERDICT_EXIT_ADVISORY = "exit_advisory"
VALID_VERDICTS = {VERDICT_HOLD, VERDICT_ALERT, VERDICT_EXIT_ADVISORY}

# States the sentry watches — open, managed exposure. Pendings are in-flight
# (nothing to defend yet); MANUAL_INTERVENTION is the operator's problem
# already; IDLE has nothing at risk.
WATCHED_STATES = (
    PositionState.CSP_OPEN,
    PositionState.CC_OPEN,
    PositionState.SPREAD_OPEN,
    PositionState.PMCC_LONG_OPEN,
    PositionState.PMCC_BOTH_OPEN,
    PositionState.SWING_OPEN,
)

SENTRY_PROMPT = """You are the news risk sentry for an automated options income bot.

You are given ONE open position (strategy, direction, key strikes and \
expirations where known) and that symbol's recent headlines. The bot already \
has mechanical exits (profit target, time close at 21 DTE, stop). Your ONLY \
question: do these headlines describe a MATERIAL, REALIZED adverse change to \
this specific position's risk — something its mechanical exits are too slow \
to see?

Respond ONLY with JSON:
{"verdict": "hold" | "alert" | "exit_advisory", "confidence": 0.0-1.0, \
"rationale": "<= 2 sentences"}

Rules:
- Default to "hold". Routine volatility, price-target changes, analyst \
opinions, rumors, and old news are ALL "hold".
- "alert" = a real development the operator should look at (e.g. guidance cut, \
regulatory action, executive departure) that plausibly pressures the position.
- "exit_advisory" = a realized, position-relevant event (not speculation) that \
materially threatens max loss before the scheduled exits — reserve for the \
clearest cases.
- You advise a human. Nothing closes automatically on your verdict."""


@dataclass(frozen=True, slots=True)
class SentryResult:
    position_id: int
    symbol: str
    strategy_id: str | None
    verdict: str
    confidence: float | None
    rationale: str
    source: str  # "llm" | "skipped:<reason>"


def _normalize(verdict: Any) -> str:
    s = str(verdict or "").strip().lower()
    return s if s in VALID_VERDICTS else VERDICT_HOLD


def _leg_summary(order: Order | None) -> dict[str, Any] | None:
    if order is None:
        return None
    return {
        "strike": order.strike,
        "expiration": order.expiration.isoformat() if order.expiration else None,
        "option_type": str(order.option_type) if order.option_type else None,
    }


async def _position_brief(repos: Repos, pos: Position) -> dict[str, Any]:
    """Best-effort position description for the prompt. Missing leg data
    degrades to strategy/state only — the sentry still runs."""
    brief: dict[str, Any] = {
        "symbol": pos.symbol,
        "strategy": pos.strategy_id,
        "state": str(pos.state),
    }
    if pos.current_cycle_id is None:
        return brief
    try:
        c = await repos.db.connect()
        async with c.execute(
            "SELECT * FROM orders WHERE cycle_id = ? AND status = ? "
            "ORDER BY placed_at DESC LIMIT 6",
            (pos.current_cycle_id, OrderStatus.FILLED.value),
        ) as cur:
            rows = await cur.fetchall()
        from db.repo import JSON_FIELDS_BY_TABLE, _row_to_dict
        legs = []
        for r in rows:
            o = Order(**_row_to_dict(r, JSON_FIELDS_BY_TABLE["orders"]))
            if o.raw_request and isinstance(o.raw_request, dict) and o.raw_request.get("legs"):
                for leg in o.raw_request["legs"]:
                    legs.append({
                        "action": leg.get("action"),
                        "strike": leg.get("strike"),
                        "expiration": leg.get("expiration"),
                        "option_type": leg.get("option_type"),
                    })
                break  # the multi-leg open describes the whole structure
            summary = _leg_summary(o)
            if summary is not None and summary["strike"] is not None:
                legs.append({"action": str(o.order_type), **summary})
        if legs:
            brief["legs"] = legs[:6]
    except Exception as exc:  # noqa: BLE001 — brief is best-effort
        log_checkpoint(
            "news_sentry_brief_fail", status="skip",
            symbol=pos.symbol, position_id=pos.id, error=str(exc),
        )
    return brief


async def run_position_news_sentry(
    *,
    repos: Repos,
    news: NewsSource | None,
    anthropic: AnthropicClient | None,
    config: dict[str, Any],
    sentry_state: dict[str, int],
    today: date | None = None,  # noqa: ARG001 — parity with earnings_recheck signature
) -> list[SentryResult]:
    """Main entry point, called from run_bot._post_tick every tick.

    Self-rate-limited via `sentry_state["ticks_since_check"]` (same pattern as
    the earnings recheck) so the LLM work runs ~hourly, not every 5 minutes.
    """
    cfg = (config.get("intelligence", {}) or {}).get("position_news_sentry", {}) or {}
    if not bool(cfg.get("enabled", False)):
        return []
    interval = int(cfg.get("check_interval_ticks", 12))
    sentry_state["ticks_since_check"] = sentry_state.get("ticks_since_check", 0) + 1
    if sentry_state["ticks_since_check"] < interval:
        return []
    sentry_state["ticks_since_check"] = 0

    if news is None or anthropic is None:
        log_checkpoint(
            "news_sentry_skip", status="skip",
            reason="no news source" if news is None else "no anthropic client",
        )
        return []

    account_id = (config.get("account") or {}).get("id", "primary")
    lookback_hours = int(cfg.get("lookback_hours", 24))
    cooldown_hours = float(cfg.get("notify_cooldown_hours", 4))
    max_positions = int(cfg.get("max_positions_per_pass", 10))
    model = cfg.get("model", "claude-haiku-4-5")

    watched = [
        p for p in await repos.positions.list_active(account_id)
        if p.state in WATCHED_STATES and p.id is not None
    ][:max_positions]

    results: list[SentryResult] = []
    for pos in watched:
        try:
            result = await _check_one(
                repos=repos, news=news, anthropic=anthropic, pos=pos,
                lookback_hours=lookback_hours, cooldown_hours=cooldown_hours,
                model=model,
            )
        except Exception as exc:  # noqa: BLE001 — one bad symbol must not kill the pass
            log_checkpoint(
                "news_sentry_position_fail", status="fail",
                symbol=pos.symbol, position_id=pos.id, error=str(exc),
            )
            continue
        if result is not None:
            results.append(result)
    if results:
        log_checkpoint(
            "news_sentry_pass", status="ok",
            n_positions=len(watched),
            n_alerts=sum(1 for r in results if r.verdict != VERDICT_HOLD),
        )
    return results


async def _check_one(
    *,
    repos: Repos,
    news: NewsSource,
    anthropic: AnthropicClient,
    pos: Position,
    lookback_hours: int,
    cooldown_hours: float,
    model: str,
) -> SentryResult | None:
    assert pos.id is not None
    try:
        headlines = await news.recent(pos.symbol, hours=lookback_hours)
    except NewsSourceUnavailable as exc:
        log_checkpoint(
            "news_sentry_source_skip", status="skip",
            symbol=pos.symbol, error=str(exc),
        )
        return None
    if not headlines:
        return None  # nothing new — no LLM spend, no result row

    brief = await _position_brief(repos, pos)
    payload = {
        "position": brief,
        "headlines": [
            {
                "headline": h.headline,
                "summary": (h.summary or "")[:400],
                "published_at": h.published_at.isoformat(),
                "source": h.source,
            }
            for h in headlines[:12]
        ],
    }
    try:
        result = await anthropic.call(
            decision_type=LlmDecisionType.POSITION_NEWS,
            model=model,
            system=SENTRY_PROMPT,
            user_payload=payload,
            max_output_tokens=256,
            context={
                "symbol": pos.symbol,
                "strategy": pos.strategy_id,
                "position_id": pos.id,
                # cycle_id makes the future hit-rate join (news_hit_rate
                # pattern) trivial: verdict -> cycle -> final_pnl.
                "cycle_id": pos.current_cycle_id,
                "n_headlines": len(headlines),
            },
        )
    except BudgetExceeded as exc:
        log_checkpoint("news_sentry_budget_skip", status="skip",
                       symbol=pos.symbol, error=str(exc))
        return None
    except ModelNotPriced as exc:
        log_checkpoint("news_sentry_model_not_priced", status="fail",
                       symbol=pos.symbol, model=model, error=str(exc),
                       note="config error: fix model id and redeploy")
        return None
    except Exception as exc:  # noqa: BLE001
        log_checkpoint("news_sentry_llm_fail", status="fail",
                       symbol=pos.symbol, error=str(exc))
        return None

    parsed = result["parsed"] or {}
    verdict = _normalize(parsed.get("verdict"))
    rationale = str(parsed.get("rationale", ""))
    confidence = parsed.get("confidence")
    out = SentryResult(
        position_id=pos.id, symbol=pos.symbol, strategy_id=pos.strategy_id,
        verdict=verdict, confidence=confidence, rationale=rationale, source="llm",
    )
    if verdict == VERDICT_HOLD:
        return out

    log_checkpoint(
        "news_sentry_flag", status="ok",
        symbol=pos.symbol, strategy=pos.strategy_id, position_id=pos.id,
        verdict=verdict, confidence=confidence, rationale=rationale,
    )
    if await repos.alert_rate_limits.try_fire(
        f"news_sentry_{pos.id}", cooldown_hours=cooldown_hours,
    ):
        await notify(
            "position.news_sentry",
            f"{pos.symbol} news sentry: {verdict.upper()}",
            symbol=pos.symbol,
            strategy=pos.strategy_id,
            position_id=pos.id,
            verdict=verdict,
            confidence=confidence,
            rationale=rationale,
            action_hint="review /decisions; close via MCP flatten_position if you agree",
        )
    return out
