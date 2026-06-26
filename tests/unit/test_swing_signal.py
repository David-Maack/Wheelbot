"""Pure-signal tests for the SPY swing strategy (multi-TF VWAP/EMA crossover)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.swing_signal import (
    SwingParams,
    TimeframeSpec,
    combine_signal,
    compute_indicators,
    ema,
    entry_crosses,
    rolling_vwap,
    session_vwap,
    trend_direction,
)


def _bars(closes, *, vols=None, start="2026-06-01 09:30", freq="5min", spread=0.0):
    """Build an OHLCV frame from a close series. high/low straddle close by `spread`."""
    idx = pd.date_range(start=start, periods=len(closes), freq=freq)
    closes = np.asarray(closes, dtype=float)
    vols = np.ones(len(closes)) if vols is None else np.asarray(vols, dtype=float)
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes + spread,
            "low": closes - spread,
            "close": closes,
            "volume": vols,
        },
        index=idx,
    )


def test_ema_seeds_on_first_and_tracks():
    out = ema([10, 10, 10, 10], period=3)
    # Constant series → EMA stays flat at the seed.
    assert np.allclose(out, 10.0)
    # Step up → EMA rises but lags the input.
    out2 = ema([10, 20], period=3)  # alpha = 0.5
    assert out2[0] == 10.0
    assert out2[1] == pytest.approx(15.0)


def test_session_vwap_resets_each_day():
    # Day 1 at price 10, day 2 at price 20 — VWAP must not bleed across the day.
    closes = [10, 10, 20, 20]
    idx = pd.to_datetime(
        ["2026-06-01 09:30", "2026-06-01 15:55", "2026-06-02 09:30", "2026-06-02 15:55"]
    )
    df = pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes, "volume": [1, 1, 1, 1]},
        index=idx,
    )
    vwap = session_vwap(df)
    assert vwap.iloc[1] == pytest.approx(10.0)  # day 1 cumulative
    assert vwap.iloc[2] == pytest.approx(20.0)  # day 2 reset, not blended toward 10


def test_rolling_vwap_volume_weights():
    df = _bars([10, 20], vols=[1, 3], freq="1D")
    vwap = rolling_vwap(df, window=2)
    # (10*1 + 20*3) / (1+3) = 70/4 = 17.5
    assert vwap.iloc[1] == pytest.approx(17.5)


def test_trend_direction_bull_bear_neutral():
    params = SwingParams()
    tf = TimeframeSpec("5m", "trigger", vwap_mode="session")
    # Rising series → close above both ema and vwap late in the series.
    up = compute_indicators(_bars(list(range(10, 30)), vols=[1] * 20), tf, params)
    assert trend_direction(up).iloc[-1] == 1
    down = compute_indicators(_bars(list(range(30, 10, -1)), vols=[1] * 20), tf, params)
    assert trend_direction(down).iloc[-1] == -1


def test_entry_crosses_detects_fresh_bullish_cross():
    params = SwingParams()
    tf = TimeframeSpec("5m", "trigger", vwap_mode="session")
    # V-shape: fall then rally — EMA should cross back above VWAP on the way up.
    closes = [20, 18, 16, 14, 12, 14, 16, 18, 20, 22, 24, 26]
    df = compute_indicators(_bars(closes, vols=[1] * len(closes)), tf, params)
    crosses = entry_crosses(df)
    assert crosses.iloc[0] == 0  # no prior bar
    assert (crosses == 1).any()  # a bullish cross occurred during the rally


def test_combine_signal_requires_full_alignment():
    # Trigger long, all higher TFs long → take it.
    assert combine_signal(1, [1, 1]) == 1
    # Trigger long, one higher TF disagrees → no trade (whipsaw filter).
    assert combine_signal(1, [1, -1]) == 0
    # Trigger long, a higher TF neutral → no trade.
    assert combine_signal(1, [1, 0]) == 0
    # No trigger → nothing regardless of alignment.
    assert combine_signal(0, [1, 1]) == 0
    # Short side mirrors.
    assert combine_signal(-1, [-1, -1]) == -1
