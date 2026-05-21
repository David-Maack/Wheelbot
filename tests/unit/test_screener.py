"""intelligence/screener — LLM call + candidate persistence."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import pytest

from core.models import IvHistory, OptionType, Quote, UniverseEntry
from intelligence.screener import (
    _flatten_yf_columns,
    _payload_from,
    _persist_candidates,
    run_screener,
)
from platforms.paper_broker import PaperBroker


# -- yfinance MultiIndex column workaround ----------------------------------


def test_flatten_yf_columns_collapses_multiindex():
    """Regression for the 2026-05-21 screener crash: newer yfinance returns
    MultiIndex columns even for single-ticker downloads, so `df["Close"]`
    became a sub-DataFrame and `.mean()` returned a Series → float(...) failed."""
    pd = pytest.importorskip("pandas")
    cols = pd.MultiIndex.from_tuples([("Close", "F"), ("High", "F"), ("Low", "F")])
    df = pd.DataFrame([[10.0, 10.5, 9.5], [11.0, 11.2, 10.8]], columns=cols)
    flat = _flatten_yf_columns(df)
    # After flatten, df["Close"] is a Series and .mean() returns a scalar.
    assert float(flat["Close"].mean()) == pytest.approx(10.5)


def test_flatten_yf_columns_noop_on_flat_columns():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"Close": [10.0, 11.0], "High": [10.5, 11.5]})
    flat = _flatten_yf_columns(df)
    assert float(flat["Close"].mean()) == pytest.approx(10.5)


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
