"""risk/regime — classification + idempotent persistence."""

from __future__ import annotations

from datetime import UTC, date

import pytest

from core.models import Regime, RegimeSnapshot
from risk.regime import RegimeInputs, _upsert, classify_regime


def test_classify_high_vol_when_vix_above_max():
    inputs = RegimeInputs(spy_close=500, spy_sma_200=400, vix_close=40, vix_change_pct=5, choppiness=50)
    regime, allowed = classify_regime(inputs, vix_max=35, vix_spike_pct=30)
    assert regime == Regime.HIGH_VOL
    assert allowed is False


def test_classify_high_vol_when_vix_spike():
    inputs = RegimeInputs(spy_close=500, spy_sma_200=400, vix_close=20, vix_change_pct=35, choppiness=50)
    regime, allowed = classify_regime(inputs, vix_max=35, vix_spike_pct=30)
    assert regime == Regime.HIGH_VOL
    assert allowed is False


def test_classify_bear_when_spy_below_sma():
    inputs = RegimeInputs(spy_close=380, spy_sma_200=400, vix_close=18, vix_change_pct=5, choppiness=50)
    regime, allowed = classify_regime(inputs, vix_max=35, vix_spike_pct=30)
    assert regime == Regime.BEAR_TREND
    assert allowed is False


def test_classify_bull_when_calm_uptrend():
    inputs = RegimeInputs(spy_close=500, spy_sma_200=400, vix_close=15, vix_change_pct=2, choppiness=40)
    regime, allowed = classify_regime(inputs, vix_max=35, vix_spike_pct=30)
    assert regime == Regime.BULL_TREND
    assert allowed is True


def test_classify_neutral_when_choppy_uptrend():
    inputs = RegimeInputs(spy_close=500, spy_sma_200=400, vix_close=18, vix_change_pct=2, choppiness=70)
    regime, allowed = classify_regime(inputs, vix_max=35, vix_spike_pct=30)
    assert regime == Regime.NEUTRAL
    assert allowed is True


@pytest.mark.asyncio
async def test_upsert_writes_and_replaces_row(db_repos):
    snap = RegimeSnapshot(
        snapshot_date=date(2025, 6, 1),
        spy_close=500.0, spy_sma_200=480.0, spy_above_sma=True,
        vix_close=15.0, vix_change_pct=2.0, choppiness=45.0,
        regime=Regime.BULL_TREND, csps_allowed=True, notes=None,
    )
    await _upsert(db_repos, snap)

    snap2 = RegimeSnapshot(
        snapshot_date=date(2025, 6, 1),
        spy_close=510.0, spy_sma_200=480.0, spy_above_sma=True,
        vix_close=20.0, vix_change_pct=33.0, choppiness=45.0,
        regime=Regime.HIGH_VOL, csps_allowed=False, notes=None,
    )
    await _upsert(db_repos, snap2)

    c = await db_repos.db.connect()
    async with c.execute("SELECT regime, csps_allowed, vix_close FROM regime_snapshots WHERE snapshot_date = ?", ("2025-06-01",)) as cur:
        row = await cur.fetchone()
    assert row["regime"] == "HIGH_VOL"
    assert row["csps_allowed"] == 0
    assert row["vix_close"] == 20.0
