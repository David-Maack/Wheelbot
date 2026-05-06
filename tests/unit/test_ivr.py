"""data/ivr.py against synthetic IV histories.

We don't spin up a real DB here — a tiny fake repo keeps the test purely about
the math.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable

import pytest

from core.models import IvHistory
from data.ivr import IVRProvider


class FakeIvHistoryRepo:
    def __init__(self, ivs: Iterable[float]):
        # Construct rows oldest → newest, ending today.
        self._rows: list[IvHistory] = []
        ivs_list = list(ivs)
        today = date.today()
        for i, iv in enumerate(ivs_list):
            d = today - timedelta(days=len(ivs_list) - 1 - i)
            self._rows.append(IvHistory(symbol="F", snapshot_date=d, iv_30d=iv))

    async def history_for(self, symbol: str, days: int = 365) -> list[IvHistory]:
        return list(self._rows)


@pytest.mark.asyncio
async def test_returns_none_when_no_history():
    provider = IVRProvider(FakeIvHistoryRepo([]))
    assert await provider.iv_rank("F") is None
    assert await provider.iv_percentile("F") is None


@pytest.mark.asyncio
async def test_under_min_points_returns_none_rank_but_keeps_stats():
    provider = IVRProvider(FakeIvHistoryRepo([0.20, 0.25, 0.30]), min_points=20)
    s = await provider.stats("F")
    assert s is not None
    assert s.rank is None and s.percentile is None
    assert s.n_points == 3


@pytest.mark.asyncio
async def test_iv_rank_at_max_is_100():
    series = [0.10] * 25 + [0.40]  # current at top
    provider = IVRProvider(FakeIvHistoryRepo(series), min_points=20)
    rank = await provider.iv_rank("F")
    assert rank == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_iv_rank_at_min_is_zero():
    series = [0.40] * 25 + [0.10]  # current at bottom
    provider = IVRProvider(FakeIvHistoryRepo(series), min_points=20)
    rank = await provider.iv_rank("F")
    assert rank == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_iv_percentile_counts_points_below_current():
    series = [0.10, 0.20, 0.30, 0.40, 0.25, 0.15] * 4 + [0.30]  # current = 0.30
    provider = IVRProvider(FakeIvHistoryRepo(series), min_points=20)
    pct = await provider.iv_percentile("F")
    assert pct is not None
    # Count of values < 0.30, divided by len
    below = sum(1 for v in series if v < 0.30)
    assert pct == pytest.approx(below / len(series) * 100)


@pytest.mark.asyncio
async def test_constant_series_yields_50_rank():
    """When low == high we can't rank meaningfully; convention: return 50."""
    provider = IVRProvider(FakeIvHistoryRepo([0.25] * 30), min_points=20)
    rank = await provider.iv_rank("F")
    assert rank == pytest.approx(50.0)
