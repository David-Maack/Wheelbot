"""intelligence/news_hit_rate — NEWS_CHECK decisions joined to cycle outcomes."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from core.models import CycleOutcome, WheelCycle
from intelligence.news_hit_rate import (
    MIN_SAMPLE_FOR_VERDICT,
    BucketStats,
    _verdict,
    compute_hit_rate,
)


def _utc() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


async def _insert_decision(
    db_repos, *, symbol: str, decision: str | None, created_at: datetime,
    confidence: float = 0.7,
) -> None:
    c = await db_repos.db.connect()
    await c.execute(
        "INSERT INTO llm_decisions "
        "(decision_type, context, decision, confidence, created_at) "
        "VALUES ('NEWS_CHECK', ?, ?, ?, ?)",
        (json.dumps({"symbol": symbol, "profile": "bullish_csp"}),
         decision, confidence, created_at.isoformat()),
    )
    await c.commit()


async def _insert_cycle(
    db_repos, *, symbol: str, started_at: datetime,
    final_pnl: float | None, closed: bool = True,
) -> None:
    await db_repos.cycles.insert(
        WheelCycle(
            account_id="test", symbol=symbol, strategy_id="monthly_wheel",
            started_at=started_at,
            ended_at=(started_at + timedelta(days=12)) if closed else None,
            final_pnl=final_pnl,
            cycle_outcome=CycleOutcome.CSP_EXPIRED if closed else None,
        )
    )


@pytest.mark.asyncio
async def test_proceed_and_caution_join_to_closed_cycles(db_repos):
    t0 = _utc() - timedelta(days=30)
    await _insert_decision(db_repos, symbol="F", decision="proceed", created_at=t0)
    await _insert_cycle(db_repos, symbol="F", started_at=t0 + timedelta(hours=2), final_pnl=60.0)
    await _insert_decision(db_repos, symbol="NOK", decision="caution", created_at=t0)
    await _insert_cycle(db_repos, symbol="NOK", started_at=t0 + timedelta(hours=3), final_pnl=-40.0)

    report = await compute_hit_rate(db_repos)
    proceed, caution = report["buckets"]["proceed"], report["buckets"]["caution"]
    assert proceed.n_decisions == proceed.n_matched == proceed.n_closed == 1
    assert proceed.wins == 1 and proceed.total_pnl == pytest.approx(60.0)
    assert caution.n_closed == 1 and caution.wins == 0
    assert caution.avg_pnl == pytest.approx(-40.0)


@pytest.mark.asyncio
async def test_block_listed_without_join(db_repos):
    t0 = _utc() - timedelta(days=5)
    await _insert_decision(db_repos, symbol="HOOD", decision="block", created_at=t0)
    # A cycle exists in-window (some OTHER strategy traded HOOD) — a block
    # must still never join to it.
    await _insert_cycle(db_repos, symbol="HOOD", started_at=t0 + timedelta(hours=1), final_pnl=99.0)

    report = await compute_hit_rate(db_repos)
    assert report["buckets"]["block"].n_decisions == 1
    assert report["buckets"]["block"].n_matched == 0
    assert report["blocks"][0]["symbol"] == "HOOD"


@pytest.mark.asyncio
async def test_unparsed_rows_excluded(db_repos):
    """Pre-2026-05-29 fence-bug rows have decision NULL — excluded, counted."""
    t0 = _utc() - timedelta(days=40)
    await _insert_decision(db_repos, symbol="F", decision=None, created_at=t0)
    report = await compute_hit_rate(db_repos)
    assert report["n_unparsed"] == 1
    assert report["n_total"] == 1
    assert report["buckets"]["proceed"].n_decisions == 0


@pytest.mark.asyncio
async def test_cycle_outside_window_not_matched(db_repos):
    t0 = _utc() - timedelta(days=30)
    await _insert_decision(db_repos, symbol="F", decision="proceed", created_at=t0)
    await _insert_cycle(db_repos, symbol="F", started_at=t0 + timedelta(days=10), final_pnl=60.0)
    report = await compute_hit_rate(db_repos, match_window_days=3)
    proceed = report["buckets"]["proceed"]
    assert proceed.n_decisions == 1
    assert proceed.n_matched == 0


@pytest.mark.asyncio
async def test_open_cycle_matched_but_not_counted(db_repos):
    t0 = _utc() - timedelta(days=2)
    await _insert_decision(db_repos, symbol="F", decision="proceed", created_at=t0)
    await _insert_cycle(
        db_repos, symbol="F", started_at=t0 + timedelta(hours=1),
        final_pnl=None, closed=False,
    )
    report = await compute_hit_rate(db_repos)
    proceed = report["buckets"]["proceed"]
    assert proceed.n_matched == 1
    assert proceed.n_closed == 0
    assert proceed.avg_pnl is None


@pytest.mark.asyncio
async def test_first_cycle_in_window_wins(db_repos):
    """Two cycles inside the window → join the EARLIER one (the fill the
    decision actually gated)."""
    t0 = _utc() - timedelta(days=30)
    await _insert_decision(db_repos, symbol="F", decision="proceed", created_at=t0)
    await _insert_cycle(db_repos, symbol="F", started_at=t0 + timedelta(hours=2), final_pnl=10.0)
    await _insert_cycle(db_repos, symbol="F", started_at=t0 + timedelta(days=2), final_pnl=-999.0)
    report = await compute_hit_rate(db_repos)
    assert report["buckets"]["proceed"].total_pnl == pytest.approx(10.0)


# -- verdict ------------------------------------------------------------------


def _stats(n_closed: int, avg: float) -> BucketStats:
    s = BucketStats(n_decisions=n_closed, n_matched=n_closed, n_closed=n_closed)
    s.total_pnl = avg * n_closed
    s.wins = n_closed if avg > 0 else 0
    return s


def test_verdict_insufficient_sample():
    buckets = {
        "proceed": _stats(MIN_SAMPLE_FOR_VERDICT, 50.0),
        "caution": _stats(MIN_SAMPLE_FOR_VERDICT - 1, -10.0),
        "block": BucketStats(),
    }
    assert "insufficient sample" in _verdict(buckets)


def test_verdict_caution_underperforms():
    buckets = {
        "proceed": _stats(MIN_SAMPLE_FOR_VERDICT, 50.0),
        "caution": _stats(MIN_SAMPLE_FOR_VERDICT, -10.0),
        "block": BucketStats(),
    }
    assert "FOR flipping" in _verdict(buckets)


def test_verdict_caution_no_edge():
    buckets = {
        "proceed": _stats(MIN_SAMPLE_FOR_VERDICT, 20.0),
        "caution": _stats(MIN_SAMPLE_FOR_VERDICT, 30.0),
        "block": BucketStats(),
    }
    assert "keep advisory" in _verdict(buckets)
