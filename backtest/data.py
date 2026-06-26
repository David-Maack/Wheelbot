"""Data adapters for the swing backtest.

Two sources, matching what the live bot already uses:
- 5-minute SPY bars from **Alpaca** (`StockHistoricalDataClient`) — years of
  intraday history, the entry-trigger timeframe.
- daily / weekly SPY + daily VIX from **yfinance** (`data.yf_helpers.safe_history`)
  — the direction-gate timeframes and the IV proxy.

The pure transforms (column normalize, RTH filter, resample, ATR) are separated
from the networked loaders so they unit-test without keys. Loaders call
`core.config.load_secrets()` to populate ALPACA_API_KEY/SECRET into the env, the
same way the Alpaca broker does.
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd

_STD_COLS = ["open", "high", "low", "close", "volume"]


# --- pure transforms --------------------------------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case + select the OHLCV columns, whatever the source named them.

    Handles yfinance ("Open"/"Adj Close") and Alpaca ("open"...) alike. Returns
    a frame with exactly the standard columns and a DatetimeIndex.
    """
    if df.empty:
        return pd.DataFrame(columns=_STD_COLS)
    out = df.copy()
    out.columns = [str(c).split()[0].lower() for c in out.columns]
    keep = {c: c for c in _STD_COLS if c in out.columns}
    out = out[list(keep)]
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    return out[[c for c in _STD_COLS if c in out.columns]]


def to_rth(df: pd.DataFrame) -> pd.DataFrame:
    """Keep regular-hours bars only (09:30–16:00 ET). Assumes an ET index."""
    if df.empty:
        return df
    t = df.index.time
    start, end = pd.Timestamp("09:30").time(), pd.Timestamp("16:00").time()
    return df[(t >= start) & (t <= end)]


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV bars to a coarser `rule` (e.g. '1D', '1W')."""
    if df.empty:
        return df
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = df.resample(rule).agg({k: v for k, v in agg.items() if k in df.columns})
    return out.dropna(subset=["close"])


def atr(daily: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range on a daily frame. Indexed like `daily`."""
    if daily.empty:
        return pd.Series(dtype=float)
    high, low, close = daily["high"], daily["low"], daily["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


# --- networked loaders (need keys / network; not unit-tested) ---------------
def load_intraday_alpaca(
    symbol: str,
    start: datetime | date,
    end: datetime | date,
    *,
    minutes: int = 5,
    feed: str = "iex",
    rth_only: bool = True,
) -> pd.DataFrame:
    """5-min SPY bars from Alpaca, normalized to ET OHLCV. Free tier = 'iex' feed."""
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    from core.config import load_secrets

    load_secrets()
    import os

    key, secret = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_API_SECRET")
    if not key or not secret:
        raise RuntimeError(
            "ALPACA_API_KEY/SECRET not set — add them to config/secrets.env "
            "(see config/secrets.env.example) or run on CT 105 where they live."
        )
    client = StockHistoricalDataClient(key, secret)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(minutes, TimeFrameUnit.Minute),
        start=start,
        end=end,
        feed=feed,
    )
    bars = client.get_stock_bars(req)
    df = bars.df
    if df is None or df.empty:
        return pd.DataFrame(columns=_STD_COLS)
    if isinstance(df.index, pd.MultiIndex):  # (symbol, timestamp)
        df = df.xs(symbol, level=0)
    df = normalize_columns(df)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York").tz_localize(None)
    return to_rth(df) if rth_only else df


def load_daily_yf(symbol: str, period: str = "3y", *, weekly: bool = False) -> pd.DataFrame:
    """Daily (or weekly) bars via the shared yfinance chokepoint."""
    from data.yf_helpers import safe_history

    interval = "1wk" if weekly else "1d"
    df = safe_history(symbol, period=period, interval=interval)
    return normalize_columns(df)


def load_vix_daily(period: str = "3y") -> pd.Series:
    """Daily VIX close as a Series indexed by date (decimal IV = close/100)."""
    df = load_daily_yf("^VIX", period=period)
    if df.empty:
        return pd.Series(dtype=float)
    s = df["close"] / 100.0
    s.index = pd.to_datetime(s.index).normalize()
    return s
