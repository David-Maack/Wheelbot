"""Daily LLM regime exception veto (2026-07-01 AI audit item #2).

Design (research-backed shape): the REGIME is computed numerically (risk/
regime.py — SMA trend, VIX, choppiness); the LLM is confined to the one thing
it's demonstrably good at — reading unstructured news — and is REDUCE-ONLY:

    "Given today's numeric market dossier and these general-market headlines,
     is there a material risk event the numbers can't see yet (war escalation,
     bank failure, surprise policy move, credit event)? YES/NO + reason."

YES → `llm_risk_veto` is stamped on today's regime snapshot and the bot halves
entry sizes for the day (scripts/run_bot.py reads it next tick). The veto can
never INCREASE risk, and every failure mode — no key, budget exceeded, network,
parse failure — fails open to no-veto. Runs once daily from scripts/run_regime
(after the snapshot write), gated by `intelligence.llm_regime_veto_enabled`
(ships false). Cost ≈ one Haiku call/day.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import httpx

from core.checkpoint import log_checkpoint
from core.models import LlmDecisionType, RegimeSnapshot

DEFAULT_VETO_MODEL = "claude-haiku-4-5"

VETO_SYSTEM_PROMPT = """You are a risk officer for a small defined-risk options bot.
The bot's market regime is computed NUMERICALLY and is not your job. Your ONLY
job: given the numeric dossier and today's general-market headlines, decide
whether the headlines contain a MATERIAL risk event the numbers cannot see yet
— e.g. war escalation, sovereign/bank credit event, major bank failure,
surprise central-bank action, market-infrastructure failure.

Routine noise (analyst opinions, single-stock moves, scheduled data previews,
political bickering) is NOT a veto. When in doubt: no veto. A veto halves the
bot's position sizes for one day; it can never increase them.

Reply with ONLY a JSON object: {"veto": true|false, "reason": "<one line>"}"""


def fetch_general_news(api_key: str, *, limit: int = 15) -> list[dict[str, Any]] | None:
    """Top general-market headlines from Finnhub. None on any failure
    (caller fails open). Mirrors data/macro_calendar._finnhub_fetch."""
    try:
        resp = httpx.get(
            "https://finnhub.io/api/v1/news",
            params={"category": "general", "token": api_key},
            timeout=10.0,
        )
        if resp.status_code in (401, 403, 429):
            log_checkpoint("regime_veto_news_auth", status="fail", code=resp.status_code)
            return None
        resp.raise_for_status()
        items = resp.json() or []
    except httpx.HTTPError as exc:
        log_checkpoint("regime_veto_news_fail", status="fail", error=str(exc))
        return None
    return [
        {"headline": i.get("headline"), "source": i.get("source")}
        for i in items[:limit]
        if i.get("headline")
    ]


def build_dossier(
    snap: RegimeSnapshot, events: list[Any], headlines: list[dict[str, Any]]
) -> dict[str, Any]:
    """The numeric context the LLM must defer to, plus the headlines it reads."""
    return {
        "numeric_regime": {
            "regime": str(snap.regime) if snap.regime else None,
            "vix_close": snap.vix_close,
            "spy_above_200sma": snap.spy_above_sma,
            "choppiness": snap.choppiness,
        },
        "scheduled_macro_events_next_7d": [
            {"date": str(e.event_date), "type": e.event_type} for e in events
        ],
        "headlines": headlines,
    }


def parse_veto(parsed: dict[str, Any] | None) -> tuple[bool, str | None]:
    """Strict, fail-open parse: anything other than an explicit true is no-veto."""
    if not isinstance(parsed, dict):
        return False, None
    veto = parsed.get("veto")
    reason = parsed.get("reason")
    if veto is True or (isinstance(veto, str) and veto.strip().lower() == "true"):
        return True, str(reason) if reason else "unspecified"
    return False, None


def veto_size_multiplier(snap: RegimeSnapshot | None) -> float:
    """Sizing hook for the bot loop: 0.5 on a vetoed day, else 1.0."""
    return 0.5 if (snap is not None and snap.llm_risk_veto) else 1.0


async def run_regime_veto(repos, config: dict[str, Any], client, *, today: date | None = None) -> bool:
    """Evaluate + stamp today's veto. Returns the veto verdict. Fails open."""
    import os

    today = today or date.today()
    try:
        snap = await repos.regime.get_for_date(today)
        if snap is None:
            log_checkpoint("regime_veto_skip", status="skip", reason="no snapshot today")
            return False
        events = await repos.macro_events.between(today, today + timedelta(days=7))
        api_key = os.environ.get("FINNHUB_API_KEY", "")
        headlines = fetch_general_news(api_key) if api_key else None
        if not headlines:
            log_checkpoint("regime_veto_skip", status="skip", reason="no headlines")
            return False

        intel = config.get("intelligence", {}) or {}
        model = str(intel.get("regime_veto_model", DEFAULT_VETO_MODEL))
        result = await client.call(
            decision_type=LlmDecisionType.REGIME_VETO,
            model=model,
            system=VETO_SYSTEM_PROMPT,
            user_payload=build_dossier(snap, events, headlines),
            max_output_tokens=200,
        )
        # 2026-07-23 review fix: parse_veto takes the PARSED payload, not the
        # client's envelope ({"decision_id", "parsed", "raw_text", ...}). The
        # envelope was being passed since arming (2026-07-01), so
        # parsed.get("veto") was always None and the veto NEVER fired —
        # masked by a test stub that returned the unwrapped dict.
        veto, reason = parse_veto(result["parsed"])
        c = await repos.db.connect()
        await c.execute(
            "UPDATE regime_snapshots SET llm_risk_veto = ?, llm_veto_reason = ? "
            "WHERE snapshot_date = ?",
            (1 if veto else 0, reason, today.isoformat()),
        )
        await c.commit()
        log_checkpoint(
            "regime_veto_done", status="ok", veto=veto, reason=reason, model=model,
        )
        return veto
    except Exception as exc:  # noqa: BLE001 — reduce-only: ANY failure = no veto
        log_checkpoint("regime_veto_fail", status="fail", error=f"{type(exc).__name__}: {exc}")
        return False
