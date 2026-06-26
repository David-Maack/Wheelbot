"""Multi-timeframe VWAP / EMA-crossover signal for the SPY swing strategy.

This encodes David's discretionary method: on each timeframe watch an anchored
VWAP and an EMA, and trade the cross — with higher timeframes setting the
*direction* and a lower timeframe *timing* the entry.

    higher TF (1D / 1W)  -> direction gate (are we allowed long, short, or neither)
    lower  TF (5-min)    -> entry trigger  (EMA(9) crossing VWAP in the gated direction)

Everything here is a PURE function over price bars (a pandas DataFrame indexed
by timestamp with open/high/low/close/volume columns). No I/O, no broker, no
network — so it is reused unchanged by both the backtest harness (sub-sprint 1)
and the live strategy (sub-sprint 2), and is trivially unit-testable.

VWAP anchoring (confirmed with David: anchored from the period open):
- intraday bars  -> session VWAP, reset every trading day.
- daily / weekly -> rolling volume-weighted VWAP over `vwap_window` bars (a
  volume-weighted moving average). This is the automatable stand-in for a
  hand-drawn anchored-VWAP on a higher-timeframe chart; the window is a tunable
  the backtest sweeps, and the assumption is surfaced in the report so it can be
  corrected once we see results.

The cross/alignment rule is split out (`trend_direction`, `entry_crosses`,
`combine_signal`) so the as-of merge of higher-TF state onto the trigger
timeline lives in the engine, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --- column names we expect on every bar frame -----------------------------
OPEN, HIGH, LOW, CLOSE, VOLUME = "open", "high", "low", "close", "volume"


@dataclass(frozen=True, slots=True)
class TimeframeSpec:
    """One timeframe in the stack.

    name:        label for reports ("5m", "1D", "1W").
    role:        "trigger" (entry timing) or "direction" (trend gate).
    vwap_mode:   "session" (reset per calendar day) or "rolling".
    vwap_window: bars for rolling VWAP (ignored when vwap_mode == "session").
    """

    name: str
    role: str
    vwap_mode: str = "rolling"
    vwap_window: int = 20


@dataclass(frozen=True, slots=True)
class SwingParams:
    ema_period: int = 9
    # Default 2-timeframe stack: 1-Day direction + 5-min trigger. The harness
    # swaps this out to sweep 2/3/4 timeframes.
    timeframes: tuple[TimeframeSpec, ...] = (
        TimeframeSpec("1D", "direction", vwap_mode="rolling", vwap_window=20),
        TimeframeSpec("5m", "trigger", vwap_mode="session"),
    )


# --- indicators -------------------------------------------------------------
def ema(values: pd.Series | np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average, seeded with the first value.

    Returned array is the same length as the input and aligned to it.
    """
    arr = np.asarray(values, dtype=float)
    out = np.empty_like(arr)
    if arr.size == 0:
        return out
    alpha = 2.0 / (period + 1.0)
    out[0] = arr[0]
    for i in range(1, arr.size):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def _typical_price(df: pd.DataFrame) -> pd.Series:
    return (df[HIGH] + df[LOW] + df[CLOSE]) / 3.0


def session_vwap(df: pd.DataFrame) -> pd.Series:
    """Cumulative VWAP that resets at each calendar day (RTH session anchor)."""
    tp = _typical_price(df)
    tpv = tp * df[VOLUME]
    day = df.index.normalize()
    cum_tpv = tpv.groupby(day).cumsum()
    cum_vol = df[VOLUME].groupby(day).cumsum()
    return cum_tpv / cum_vol.replace(0, np.nan)


def rolling_vwap(df: pd.DataFrame, window: int) -> pd.Series:
    """Volume-weighted moving average over `window` bars (anchored-VWAP stand-in)."""
    tp = _typical_price(df)
    num = (tp * df[VOLUME]).rolling(window, min_periods=1).sum()
    den = df[VOLUME].rolling(window, min_periods=1).sum()
    return num / den.replace(0, np.nan)


def compute_indicators(df: pd.DataFrame, tf: TimeframeSpec, params: SwingParams) -> pd.DataFrame:
    """Return a copy of `df` with `ema` and `vwap` columns for this timeframe."""
    out = df.copy()
    out["ema"] = ema(out[CLOSE], params.ema_period)
    if tf.vwap_mode == "session":
        out["vwap"] = session_vwap(out)
    else:
        out["vwap"] = rolling_vwap(out, tf.vwap_window)
    return out


# --- direction gate + entry trigger ----------------------------------------
def trend_direction(df_ind: pd.DataFrame) -> pd.Series:
    """+1 bullish / -1 bearish / 0 neutral per bar.

    Bullish when close is above BOTH vwap and ema; bearish when below both;
    neutral (no lean) otherwise. NaN indicators (warm-up) read as neutral.
    """
    close, vwap, ema_ = df_ind[CLOSE], df_ind["vwap"], df_ind["ema"]
    bull = (close > vwap) & (close > ema_)
    bear = (close < vwap) & (close < ema_)
    return pd.Series(np.where(bull, 1, np.where(bear, -1, 0)), index=df_ind.index)


def entry_crosses(df_ind: pd.DataFrame) -> pd.Series:
    """+1 on a fresh bullish EMA-over-VWAP cross, -1 on a bearish cross, else 0."""
    diff = df_ind["ema"] - df_ind["vwap"]
    prev = diff.shift(1)
    bull = (prev <= 0) & (diff > 0)
    bear = (prev >= 0) & (diff < 0)
    cross = pd.Series(np.where(bull, 1, np.where(bear, -1, 0)), index=df_ind.index)
    # First bar has no prior → no cross.
    cross.iloc[0] = 0
    return cross


def combine_signal(trigger_cross: int, htf_directions: list[int]) -> int:
    """Final entry direction: a fresh trigger cross, confirmed by EVERY higher
    timeframe agreeing on the same direction. Returns +1 / -1 / 0.

    This is the whipsaw filter — a 5-min cross is only taken when the daily
    (and, in the 3/4-TF sweeps, the weekly/1-min) all lean the same way.
    """
    if trigger_cross == 0:
        return 0
    if all(d == trigger_cross for d in htf_directions):
        return trigger_cross
    return 0


@dataclass(frozen=True, slots=True)
class Signal:
    ts: pd.Timestamp
    direction: int  # +1 long, -1 short
    spot: float
    htf_directions: tuple[int, ...] = field(default_factory=tuple)
