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


def effective_ivr_min(ivr: object, params: dict) -> float:
    """The strategy's IVR floor for this tick — selectors' single entry point.

    Delegates to `IVRProvider.effective_ivr_min` (regime-aware) when the
    provider implements it; falls back to the raw `ivr_min` param for
    provider-like stubs that only implement `iv_rank` (unit tests, older
    call sites). Keeps the five selector gates one-liner-identical without
    forcing every stub to know about regime modulation."""
    fn = getattr(ivr, "effective_ivr_min", None)
    if fn is not None:
        return float(fn(params))
    return float(params.get("ivr_min", 0))


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
        relax_vix_threshold: float | None = None,
        relax_floor_delta: float = 0.0,
    ) -> None:
        self._repo = iv_history_repo
        self._lookback = lookback_days
        self._min_points = min_points
        # Regime-aware floor relax (2026-07-06 sprint): defaults OFF
        # (delta 0 / threshold None) per the new-knob convention; armed via
        # risk.ivr_regime_relax in config, wired at construction in run_bot.
        self._relax_vix_threshold = relax_vix_threshold
        self._relax_floor_delta = relax_floor_delta
        self._regime_vix: float | None = None

    def set_regime_vix(self, vix_close: float | None) -> None:
        """Per-tick regime context from the latest regime snapshot. None =
        unknown (missing snapshot / failed read) → no modulation."""
        self._regime_vix = vix_close

    def effective_ivr_min(self, params: dict) -> float:
        """The strategy's IVR floor, regime-adjusted.

        IVR is a RELATIVE measure: in a sustained high-vol regime it reads
        "low" for weeks while absolute premium is rich everywhere (2022-style
        failure — the rank compares against an elevated 52w high). When VIX
        closes at/above the relax threshold, the floor drops by
        `relax_floor_delta` (clamped at 0) so the bot doesn't skip rich
        premium on a technicality. Calm-regime behavior is deliberately
        UNCHANGED — the 2026-07-01 audit set the calm floor, and the
        selector-level min-credit rule already gates thin premium directly.

        Per-strategy `ivr_relax_vix` / `ivr_relax_delta` params override the
        provider-wide defaults. Base `ivr_min` <= 0 (gate off) stays off.
        """
        base = float(params.get("ivr_min", 0))
        if base <= 0 or self._regime_vix is None:
            return base
        threshold_raw = params.get("ivr_relax_vix", self._relax_vix_threshold)
        delta = float(params.get("ivr_relax_delta", self._relax_floor_delta))
        if threshold_raw is None or delta <= 0:
            return base
        if self._regime_vix >= float(threshold_raw):
            return max(base - delta, 0.0)
        return base

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
