"""Live swing signal evaluator (reuses the backtest signal → live == backtest)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.engine import EngineConfig
from strategies.swing import evaluate_swing_signal
from strategies.swing_signal import SwingParams, TimeframeSpec

# 2-timeframe stack (daily direction + 5-min trigger) keeps the test self-contained.
_PARAMS = SwingParams(timeframes=(
    TimeframeSpec("1D", "direction", vwap_mode="rolling", vwap_window=20),
    TimeframeSpec("5m", "trigger", vwap_mode="session"),
))
_CFG = EngineConfig(use_regime=True, regime_sma=3)  # small SMA so warm-up is short


def _frame(dates, closes, *, hl=0.2, vol=1000):
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {"open": closes, "high": closes + hl, "low": closes - hl, "close": closes,
         "volume": [vol] * len(closes)},
        index=pd.DatetimeIndex(dates),
    )


def test_evaluate_returns_signal_on_fresh_last_bar_cross():
    # Daily uptrend → direction +1 and above the 3-SMA (regime +1).
    ddates = pd.date_range("2026-06-01", periods=5, freq="1D")
    daily = _frame(ddates, [90, 92, 94, 96, 98], hl=1.0)
    # 5-min flat all session, then a sharp up-move on the FINAL bar → EMA crosses
    # VWAP exactly on the last bar (a fresh trigger).
    idx = pd.date_range("2026-06-05 09:30", periods=12, freq="5min")
    five = _frame(idx, [100] * 11 + [110])
    sig = evaluate_swing_signal(five, daily, params=_PARAMS, cfg=_CFG)
    assert sig is not None
    assert sig.direction == 1
    assert sig.spot == 110.0
    assert sig.ts == idx[-1]


def test_evaluate_returns_none_when_flat():
    ddates = pd.date_range("2026-06-01", periods=5, freq="1D")
    daily = _frame(ddates, [90, 92, 94, 96, 98], hl=1.0)
    idx = pd.date_range("2026-06-05 09:30", periods=12, freq="5min")
    flat = _frame(idx, [100] * 12)  # no cross → no signal
    assert evaluate_swing_signal(flat, daily, params=_PARAMS, cfg=_CFG) is None


def test_evaluate_handles_empty_frames():
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    assert evaluate_swing_signal(empty, empty, params=_PARAMS, cfg=_CFG) is None
