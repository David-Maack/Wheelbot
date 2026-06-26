"""Data transforms, no-lookahead as-of merge, and a synthetic end-to-end trade."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.data import atr, normalize_columns, resample_ohlcv, to_rth
from backtest.engine import (
    EngineConfig,
    StructureSpec,
    asof_daily_direction,
    generate_signals,
    price_structure,
    simulate_spy_trades,
)
from backtest.report import summarize
from strategies.swing_signal import SwingParams, TimeframeSpec


# --- data transforms --------------------------------------------------------
def test_normalize_columns_handles_yfinance_shape():
    idx = pd.date_range("2026-06-01", periods=2, freq="1D")
    df = pd.DataFrame(
        {"Open": [1, 2], "High": [2, 3], "Low": [0, 1], "Close": [1.5, 2.5],
         "Adj Close": [1.5, 2.5], "Volume": [10, 20]},
        index=idx,
    )
    out = normalize_columns(df)
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert out["close"].iloc[-1] == 2.5


def test_to_rth_filters_extended_hours():
    idx = pd.to_datetime(["2026-06-01 08:00", "2026-06-01 10:00", "2026-06-01 17:00"])
    df = pd.DataFrame({"open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}, index=idx)
    out = to_rth(df)
    assert len(out) == 1 and out.index[0].hour == 10


def test_resample_ohlcv_aggregates():
    idx = pd.date_range("2026-06-01 09:30", periods=4, freq="5min")
    df = pd.DataFrame(
        {"open": [10, 11, 12, 13], "high": [11, 12, 13, 14], "low": [9, 10, 11, 12],
         "close": [10.5, 11.5, 12.5, 13.5], "volume": [1, 1, 1, 1]},
        index=idx,
    )
    out = resample_ohlcv(df, "1D")
    assert len(out) == 1
    assert out["open"].iloc[0] == 10 and out["high"].iloc[0] == 14
    assert out["low"].iloc[0] == 9 and out["close"].iloc[0] == 13.5
    assert out["volume"].iloc[0] == 4


def test_atr_constant_range():
    idx = pd.date_range("2026-06-01", periods=3, freq="1D")
    df = pd.DataFrame({"open": [10, 11, 12], "high": [11, 12, 13],
                       "low": [9, 10, 11], "close": [10, 11, 12], "volume": [1, 1, 1]}, index=idx)
    assert atr(df).iloc[-1] == pytest.approx(2.0)


# --- no-lookahead as-of merge ----------------------------------------------
def test_asof_daily_direction_uses_prior_day():
    dates = pd.to_datetime(["2026-06-01", "2026-06-02", "2026-06-03"])
    daily_dir = pd.Series([1, -1, 1], index=dates)
    intraday = pd.to_datetime(
        ["2026-06-01 10:00", "2026-06-02 10:00", "2026-06-03 10:00"]
    )
    out = asof_daily_direction(intraday, daily_dir)
    assert out.iloc[0] == 0    # first day: no prior → neutral
    assert out.iloc[1] == 1    # 06-02 sees 06-01's +1
    assert out.iloc[2] == -1   # 06-03 sees 06-02's -1 (never today's own bar)


# --- signal generation on aligned cross ------------------------------------
def _spread_frame(dates, closes, *, hl=0.2, vol=1000):
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {"open": closes, "high": closes + hl, "low": closes - hl, "close": closes,
         "volume": [vol] * len(closes)},
        index=pd.DatetimeIndex(dates),
    )


def test_generate_signals_fires_on_aligned_bullish_cross():
    # Daily: 5 rising sessions → each daily bar leans bullish; 06-05's signal
    # uses 06-04's (+1) direction.
    ddates = pd.date_range("2026-06-01", periods=5, freq="1D")
    daily = _spread_frame(ddates, [90, 92, 94, 96, 98], hl=1.0)
    # 5-min V-shape on 06-05: dip then rally → a fresh bullish EMA/VWAP cross.
    idx = pd.date_range("2026-06-05 09:30", periods=12, freq="5min")
    five = _spread_frame(idx, [98, 97, 96, 95, 96, 97, 98, 99, 100, 101, 102, 103])
    params = SwingParams(timeframes=(
        TimeframeSpec("1D", "direction", vwap_mode="rolling", vwap_window=20),
        TimeframeSpec("5m", "trigger", vwap_mode="session"),
    ))
    sig = generate_signals(five, daily, params)
    assert (sig["signal"] == 1).any()


# --- synthetic end-to-end winning long -------------------------------------
def test_simulate_and_price_winning_long():
    # Daily for ATR (constant range 2) on the entry date and the prior day.
    ddates = pd.date_range("2026-06-01", periods=3, freq="1D")
    daily = pd.DataFrame(
        {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000}, index=ddates
    )
    # Hand-built 5-min signal frame on 06-03: signal fires at bar 1, then rallies
    # through the +3 target (stop_atr=1 * ATR=2 → stop_dist 2 → target entry+3).
    idx = pd.date_range("2026-06-03 09:30", periods=4, freq="5min")
    sig_df = pd.DataFrame(
        {
            "open": [100, 100, 101, 103],
            "high": [100, 100, 101.5, 103.5],
            "low": [100, 100, 100.5, 101.0],
            "close": [100, 100, 101, 103],
            "volume": [1000, 1000, 1000, 1000],
            "cross": [0, 1, 0, 0],
            "signal": [0, 1, 0, 0],
        },
        index=idx,
    )
    cfg = EngineConfig()
    spy_trades = simulate_spy_trades(sig_df, daily, cfg)
    assert len(spy_trades) == 1
    t = spy_trades[0]
    assert t.direction == 1 and t.exit_reason == "target"
    assert t.entry_spot == 100.0 and t.exit_spot == pytest.approx(103.0)

    vix = pd.Series([0.18, 0.18], index=pd.to_datetime(["2026-06-01", "2026-06-02"]))
    for spec in (StructureSpec("ITM", 0.67), StructureSpec("OTM", 0.30)):
        trades = price_structure(spy_trades, spec, vix, cfg)
        assert len(trades) == 1
        assert trades[0].pnl > 0  # a +3 SPY move on a long call must profit
        s = summarize(trades)
        assert s.n_trades == 1 and s.win_rate == 1.0 and s.total_pnl > 0
