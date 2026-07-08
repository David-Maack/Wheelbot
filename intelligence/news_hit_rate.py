"""News-check hit-rate report (2026-07-07 sprint).

The evidence gate for flipping `intelligence.news_check_advisory` to false:
in advisory mode BOTH "proceed" and "caution" entries trade at full size, so
the paper account is a natural experiment — if caution-flagged entries
realize worse P&L than proceed entries, the Haiku catalyst sniff has
predictive value and enforcement (caution halves size / 1-lot caution
blocks) is justified. If they perform the same, advisory mode is the right
setting and enforcement would just tax entries randomly.

Method: join every NEWS_CHECK llm_decisions row to the FIRST wheel cycle on
the same symbol whose `started_at` falls within `match_window_days` after
the decision (news_check fires at router time right before the CSP order;
the cycle starts on the fill, normally the same day). Buckets:

    proceed / caution  -> realized cycle P&L (closed cycles only)
    block              -> no trade exists; counted + listed (counterfactual
                          would need price modeling — out of scope v1)
    unparsed           -> pre-2026-05-29 fence-bug rows (decision NULL)

Read-only: no DB writes, no state. Used by scripts/news_check_hit_rate.py
(CLI) and the /decisions dashboard panel.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from db.repo import Repos

MIN_SAMPLE_FOR_VERDICT = 10  # per bucket, before the comparison means much


@dataclass
class BucketStats:
    n_decisions: int = 0
    n_matched: int = 0        # matched to a cycle (open or closed)
    n_closed: int = 0         # matched to a CLOSED cycle → counted in P&L
    wins: int = 0
    total_pnl: float = 0.0
    pnls: list[float] = field(default_factory=list)

    @property
    def win_rate(self) -> float | None:
        return (self.wins / self.n_closed) if self.n_closed else None

    @property
    def avg_pnl(self) -> float | None:
        return (self.total_pnl / self.n_closed) if self.n_closed else None


def _symbol_from_context(raw: Any) -> str | None:
    if raw is None:
        return None
    try:
        ctx = json.loads(raw) if isinstance(raw, str) else raw
        sym = ctx.get("symbol")
        return str(sym).upper() if sym else None
    except (ValueError, AttributeError):
        return None


async def compute_hit_rate(
    repos: Repos,
    *,
    match_window_days: int = 3,
) -> dict[str, Any]:
    """Aggregate NEWS_CHECK decisions vs realized cycle outcomes.

    Returns {"buckets": {decision: BucketStats}, "blocks": [...],
    "n_unparsed": int, "n_total": int, "verdict": str}.
    """
    c = await repos.db.connect()
    async with c.execute(
        "SELECT id, decision, confidence, created_at, context "
        "FROM llm_decisions WHERE decision_type = 'NEWS_CHECK' "
        "ORDER BY created_at",
    ) as cur:
        rows = await cur.fetchall()

    buckets: dict[str, BucketStats] = {
        "proceed": BucketStats(), "caution": BucketStats(), "block": BucketStats(),
    }
    blocks: list[dict[str, Any]] = []
    n_unparsed = 0

    for row in rows:
        decision = (row["decision"] or "").strip().lower()
        if decision not in buckets:
            n_unparsed += 1
            continue
        stats = buckets[decision]
        stats.n_decisions += 1
        symbol = _symbol_from_context(row["context"])

        if decision == "block":
            blocks.append({
                "symbol": symbol or "?",
                "created_at": row["created_at"],
                "confidence": row["confidence"],
            })
            continue  # nothing traded — no realized outcome to join
        if symbol is None:
            continue

        # First cycle on this symbol started inside the match window. The
        # decision fires at router time; the cycle starts on the fill.
        async with c.execute(
            "SELECT final_pnl, ended_at FROM wheel_cycles "
            "WHERE symbol = ? AND started_at >= ? "
            "  AND started_at <= datetime(?, ?) "
            "ORDER BY started_at LIMIT 1",
            (symbol, row["created_at"], row["created_at"],
             f"+{int(match_window_days)} days"),
        ) as cur2:
            cycle = await cur2.fetchone()
        if cycle is None:
            continue
        stats.n_matched += 1
        if cycle["ended_at"] is None or cycle["final_pnl"] is None:
            continue  # still open — no realized P&L yet
        pnl = float(cycle["final_pnl"])
        stats.n_closed += 1
        stats.pnls.append(pnl)
        stats.total_pnl += pnl
        if pnl > 0:
            stats.wins += 1

    return {
        "buckets": buckets,
        "blocks": blocks,
        "n_unparsed": n_unparsed,
        "n_total": len(rows),
        "verdict": _verdict(buckets),
    }


def _verdict(buckets: dict[str, BucketStats]) -> str:
    """One-line reading of the evidence. Deliberately conservative: below
    MIN_SAMPLE_FOR_VERDICT closed cycles per bucket it refuses to conclude."""
    proceed, caution = buckets["proceed"], buckets["caution"]
    if (proceed.n_closed < MIN_SAMPLE_FOR_VERDICT
            or caution.n_closed < MIN_SAMPLE_FOR_VERDICT):
        return (
            f"insufficient sample (proceed {proceed.n_closed}, caution "
            f"{caution.n_closed} closed; need {MIN_SAMPLE_FOR_VERDICT} each) "
            "— keep news_check_advisory: true and keep collecting"
        )
    assert proceed.avg_pnl is not None and caution.avg_pnl is not None
    if caution.avg_pnl < proceed.avg_pnl:
        return (
            f"caution entries underperform (${caution.avg_pnl:.2f} vs "
            f"${proceed.avg_pnl:.2f}/cycle) — evidence FOR flipping "
            "news_check_advisory: false"
        )
    return (
        f"caution entries do NOT underperform (${caution.avg_pnl:.2f} vs "
        f"${proceed.avg_pnl:.2f}/cycle) — no evidence for enforcement; "
        "keep advisory"
    )
