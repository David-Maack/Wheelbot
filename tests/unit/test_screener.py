"""intelligence/screener — LLM call + candidate persistence."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import pytest

from core.models import IvHistory, OptionType, Quote, UniverseEntry
from intelligence.screener import (
    _payload_from,
    _persist_candidates,
    run_screener,
)
from platforms.paper_broker import PaperBroker


# yfinance MultiIndex flattening tests moved to tests/unit/test_yf_helpers.py
# when the helper was centralised in data/yf_helpers.py (TICKET-001).


class _StubAnthropic:
    def __init__(self, response: dict[str, Any]):
        self._response = response
        self.calls: list[dict[str, Any]] = []

    async def call(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "parsed": self._response,
            "raw_text": "",
            "tokens_in": 1000,
            "tokens_out": 500,
            "cost_usd": 0.05,
            "decision_id": 1,
        }


class _StubIvr:
    async def stats(self, symbol):
        from data.ivr import IvStats

        return IvStats(current=0.30, low=0.20, high=0.40, n_points=30, rank=50.0, percentile=60.0)

    async def iv_rank(self, symbol):
        return 50.0

    async def iv_percentile(self, symbol):
        return 60.0


def _utc():
    return datetime.now(UTC).replace(tzinfo=None)


@pytest.mark.asyncio
async def test_persist_writes_one_candidate_per_input(db_repos):
    from intelligence.screener import TickerSnapshot

    snapshots = [
        TickerSnapshot(symbol="F", tier=1, price=10.0, ivr=50.0),
        TickerSnapshot(symbol="BAC", tier=1, price=40.0, ivr=30.0),
    ]
    parsed = {
        "candidates": [
            {"rank": 1, "symbol": "F", "score": 80, "rationale": "good"},
            {"rank": 2, "symbol": "BAC", "score": 60, "rationale": "ok"},
        ]
    }
    n = await _persist_candidates(db_repos, snapshots, parsed, date(2025, 6, 1))
    assert n == 2
    rows = await db_repos.candidates.list_for_date(date(2025, 6, 1))
    assert {r.symbol for r in rows} == {"F", "BAC"}


@pytest.mark.asyncio
async def test_persist_drops_unknown_symbols(db_repos):
    from intelligence.screener import TickerSnapshot

    snapshots = [TickerSnapshot(symbol="F", tier=1)]
    parsed = {
        "candidates": [
            {"rank": 1, "symbol": "F", "score": 80},
            {"rank": 2, "symbol": "ZZZ", "score": 50},  # not in input
        ]
    }
    n = await _persist_candidates(db_repos, snapshots, parsed, date(2025, 6, 1))
    assert n == 1


@pytest.mark.asyncio
async def test_persist_returns_zero_on_malformed_response(db_repos):
    """A truncated / non-JSON model reply falls back to {"text": ...} with no
    `candidates` key — persists nothing. run_screener now logs
    `screener_zero_candidates` (status=fail) on this so it isn't silent."""
    from intelligence.screener import TickerSnapshot

    snapshots = [TickerSnapshot(symbol="F", tier=1)]
    # Parse-failure fallback shape from AnthropicClient.call.
    assert await _persist_candidates(db_repos, snapshots, {"text": "not json"}, date(2025, 6, 1)) == 0
    # Empty list shape.
    assert await _persist_candidates(db_repos, snapshots, {"candidates": []}, date(2025, 6, 1)) == 0


@pytest.mark.asyncio
async def test_run_screener_skipped_when_disabled(db_repos):
    broker = PaperBroker()
    config = {"intelligence": {"llm_screener_enabled": False}}
    result = await run_screener(
        broker=broker,
        repos=db_repos,
        ivr=_StubIvr(),  # type: ignore[arg-type]
        news=None,
        anthropic=_StubAnthropic({}),  # type: ignore[arg-type]
        config=config,
    )
    assert result == {"skipped": "disabled"}


def test_payload_from_collects_fields():
    from intelligence.screener import TickerSnapshot

    snap = TickerSnapshot(symbol="F", tier=1, price=10.0, ivr=50.0, headlines=["a", "b"])
    payload = _payload_from([snap])
    assert payload["tickers"][0]["symbol"] == "F"
    assert payload["tickers"][0]["headlines"] == ["a", "b"]
