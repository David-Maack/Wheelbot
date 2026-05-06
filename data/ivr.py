"""IV Rank / IV Percentile from rolling 52w history.

IVR  (IV Rank)        = (current_iv - 52w_low) / (52w_high - 52w_low) * 100
IVP  (IV Percentile)  = % of trading days in the lookback where iv < current_iv

Both consume rows from the `iv_history` table — backfilled by
`scripts/ingest_history.py`. Until enough days accumulate (`min_points`,
default 20) the provider returns `None`, and the CSP selector treats that
as "skip the IVR gate" so paper trading isn't blocked from day one.

If/when paid historical data is available, swap the source in one place by
re-pointing `IVRProvider`'s repo dependency. The math stays the same.
"""

from __future__ import annotations

from dataclasses import dataclass

from db.repo import IvHistoryRepo


@dataclass(frozen=True, slots=True)
class IvStats:
    current: float
    low: float
    high: float
    n_points: int
    rank: float | None
    percentile: float | None


class IVRProvider:
    def __init__(
        self,
        iv_history_repo: IvHistoryRepo,
        *,
        lookback_days: int = 365,
        min_points: int = 20,
    ) -> None:
        self._repo = iv_history_repo
        self._lookback = lookback_days
        self._min_points = min_points

    async def stats(self, symbol: str) -> IvStats | None:
        rows = await self._repo.history_for(symbol, days=self._lookback)
        ivs = [r.iv_30d for r in rows if r.iv_30d is not None]
        if not ivs:
            return None
        if len(ivs) < self._min_points:
            return IvStats(
                current=ivs[-1],
                low=min(ivs),
                high=max(ivs),
                n_points=len(ivs),
                rank=None,
                percentile=None,
            )
        current = ivs[-1]
        lo, hi = min(ivs), max(ivs)
        rank = ((current - lo) / (hi - lo) * 100) if hi > lo else 50.0
        below = sum(1 for v in ivs if v < current)
        percentile = below / len(ivs) * 100
        return IvStats(
            current=current,
            low=lo,
            high=hi,
            n_points=len(ivs),
            rank=rank,
            percentile=percentile,
        )

    async def iv_rank(self, symbol: str) -> float | None:
        s = await self.stats(symbol)
        return s.rank if s else None

    async def iv_percentile(self, symbol: str) -> float | None:
        s = await self.stats(symbol)
        return s.percentile if s else None
