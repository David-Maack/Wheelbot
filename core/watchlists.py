"""Watchlist overlay — dynamic universe membership (universe-refresh sprint).

The universe-refresh job (intelligence/universe_refresh.py) proposes per-strategy
watchlists; once a run is APPLIED, `effective_universe` overlays its membership
onto universe.yaml. The overlay ONLY changes which strategies a symbol belongs
to — tiers, overrides, banned list/rules, and every downstream risk gate are
untouched. Symbols the refresh introduces that have no universe.yaml entry are
synthesized at tier 2, so the daily-screener score gate (risk/limits.py
tier2_screen rule) applies before any trade on an auto-added name.

Fail-open contract: any error (DB unavailable, malformed rows) returns plain
universe.yaml membership — the bot never halts because the overlay broke.
"""

from __future__ import annotations

from typing import Any

from core.checkpoint import log_checkpoint
from core.config import load_universe
from core.models import UniverseEntry


def overlay_universe(
    universe: dict[str, Any], membership: dict[str, list[str]]
) -> dict[str, Any]:
    """Pure overlay: for each strategy in `membership`, that list IS the
    strategy's universe; strategies not in `membership` keep their
    universe.yaml tags. Banned symbols are excluded no matter what the
    watchlist says."""
    banned = {str(b).upper() for b in universe.get("banned", [])}
    # Tier-3 = banned-tier (spec §6: never tradable). A tier-3 yaml entry must
    # never regain membership via a watchlist — tier 3 also BYPASSES the
    # tier2_screen risk rule, so re-adding one would trade ungated.
    banned |= {t.symbol.upper() for t in universe.get("tickers", []) if t.tier == 3}
    want: dict[str, set[str]] = {
        sid: {s.upper() for s in symbols if s.upper() not in banned}
        for sid, symbols in membership.items()
    }
    refreshed = set(want)

    known = {t.symbol.upper() for t in universe.get("tickers", [])}
    tickers: list[UniverseEntry] = []
    for t in universe.get("tickers", []):
        sym = t.symbol.upper()
        kept = [sid for sid in t.strategies if sid not in refreshed]
        added = [sid for sid in want if sym in want[sid] and sid not in kept]
        tickers.append(t.model_copy(update={"strategies": kept + added}))

    # Watchlist symbols with no universe.yaml entry: synthesize at tier 2 so
    # the LLM-screener score gate applies before any trade.
    unknown: dict[str, list[str]] = {}
    for sid, symbols in want.items():
        for sym in symbols:
            if sym not in known:
                unknown.setdefault(sym, []).append(sid)
    for sym, sids in sorted(unknown.items()):
        tickers.append(
            UniverseEntry(
                symbol=sym,
                name=sym,
                tier=2,
                strategies=sorted(sids),
                notes="auto-added by universe refresh (no universe.yaml entry)",
            )
        )

    return {
        "tickers": tickers,
        "banned": universe.get("banned", []),
        "banned_rules": universe.get("banned_rules", []),
    }


async def effective_universe(
    repos: Any, config: dict[str, Any], config_dir: Any = None
) -> dict[str, Any]:
    """universe.yaml with the currently-applied watchlist run overlaid.
    Returns plain universe.yaml when the feature is disabled, no run is
    applied, or anything fails (fail-open)."""
    universe = load_universe(config_dir)
    ur = config.get("universe_refresh", {}) or {}
    if not bool(ur.get("enabled", False)):
        return universe
    try:
        membership = await repos.watchlists.applied_membership()
    except Exception as exc:  # noqa: BLE001 — overlay must never break callers
        log_checkpoint("watchlist_overlay_fail", status="fail", error=str(exc))
        return universe
    if not membership:
        return universe
    return overlay_universe(universe, membership)
