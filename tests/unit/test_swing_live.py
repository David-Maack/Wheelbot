"""Live swing signal evaluator (reuses the backtest signal → live == backtest)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from datetime import date

from backtest.engine import EngineConfig
from core.models import OptionContract, OptionType
from strategies.swing import (
    evaluate_swing_signal,
    pick_deep_itm,
    swing_exit_decision,
    swing_stop_target,
)
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


# --- deep-ITM option selection (2.2a) --------------------------------------
def _call(strike, delta, bid=10.0, ask=10.2):
    return OptionContract(
        underlying="SPY", occ_symbol=f"SPY___C{strike}", strike=strike,
        expiration=date(2026, 8, 21), option_type=OptionType.CALL,
        bid=bid, ask=ask, delta=delta,
    )


def test_pick_deep_itm_closest_to_target_delta():
    cands = [_call(560, 0.85), _call(550, 0.92), _call(540, 0.97)]
    best = pick_deep_itm(cands, target_delta=0.90)
    assert best.strike == 550  # delta 0.92 is closest to 0.90


def test_pick_deep_itm_skips_unpriced():
    cands = [_call(550, 0.90, bid=None, ask=None), _call(560, 0.80)]
    best = pick_deep_itm(cands, target_delta=0.90)
    assert best.strike == 560  # the 0.90 has no mid → skipped


def test_pick_deep_itm_empty():
    assert pick_deep_itm([], target_delta=0.90) is None


# --- SPY-level stop / target / exit (2.2a) ---------------------------------
def test_swing_stop_target_long():
    stop, target = swing_stop_target(600.0, direction=1, prior_day_level=596.0, reward_risk=1.5)
    assert stop == 596.0
    assert target == 600.0 + 1.5 * 4.0  # entry + 1.5 * (600-596) = 606


def test_swing_stop_target_short():
    stop, target = swing_stop_target(600.0, direction=-1, prior_day_level=604.0, reward_risk=1.5)
    assert stop == 604.0
    assert target == 600.0 - 1.5 * 4.0  # 594


def test_swing_exit_min_hold_suppresses_stop():
    # Long, price below stop, but within min-hold → no stop yet.
    e, r = swing_exit_decision(1, 595.0, 596.0, 606.0, hold_days=0.2,
                               min_hold_days=1.0, max_hold_days=7.0)
    assert e is False and r is None
    # Past min-hold → stop fires.
    e, r = swing_exit_decision(1, 595.0, 596.0, 606.0, hold_days=1.5,
                               min_hold_days=1.0, max_hold_days=7.0)
    assert e is True and r == "swing_stop"


def test_swing_exit_target_and_time():
    e, r = swing_exit_decision(1, 607.0, 596.0, 606.0, hold_days=0.1,
                               min_hold_days=1.0, max_hold_days=7.0)
    assert e is True and r == "swing_target"  # target fires even before min-hold
    e, r = swing_exit_decision(1, 600.0, 596.0, 606.0, hold_days=7.0,
                               min_hold_days=1.0, max_hold_days=7.0)
    assert e is True and r == "swing_time_stop"
