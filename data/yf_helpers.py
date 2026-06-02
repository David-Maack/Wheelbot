"""Single chokepoint for yfinance access.

Two helpers callers use:

  - `flatten_yf_columns(df)` collapses the MultiIndex that yfinance >= 0.2.34
    returns even for single-ticker downloads. Without it, `df["Close"]` is a
    sub-DataFrame instead of a Series and downstream `.mean()` / `.iloc[-1]`
    return Series instead of scalars.
  - `safe_history(symbol, period, interval)` wraps `yf.download(...)`, flattens
    the result, and returns an EMPTY DataFrame on any exception (network, auth,
    API change). Callers can `if df.empty: return None` and never see a raise.

Both helpers used to live duplicated inside `intelligence/screener.py` and
`risk/regime.py`. Centralising here so the next yfinance breaking change has
one place to fix.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import yfinance as yf

from core.checkpoint import log_checkpoint


def flatten_yf_columns(df: Any) -> Any:
    """Collapse a MultiIndex columns DataFrame to flat columns by taking
    the field name (level 0). No-op on a non-MultiIndex DataFrame.

    Pure function — returns the same object after potentially mutating its
    `columns` attribute. Behaviour-identical to the two copies it replaces.
    """
    cols = getattr(df, "columns", None)
    if cols is not None and hasattr(cols, "levels"):
        df.columns = cols.get_level_values(0)
    return df


def safe_history(
    symbol: str,
    period: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch a history DataFrame for `symbol`, defensively.

    Matches the call shape the existing screener + regime code relied on:
    `yf.download(symbol, period=..., interval=..., progress=False,
    auto_adjust=False)`. We use `download` rather than `Ticker.history` so the
    refactor is behaviour-preserving (the two flavours can disagree on the
    MultiIndex quirk and on `auto_adjust` defaults).

    On any exception OR if yfinance returns None / an empty frame, returns an
    EMPTY DataFrame and logs a `yf_safe_history_fail` checkpoint. Callers can
    branch on `df.empty` without try/except wrappers of their own.
    """
    try:
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
        )
    except Exception as exc:
        log_checkpoint(
            "yf_safe_history_fail",
            status="fail",
            symbol=symbol,
            period=period,
            error=str(exc),
        )
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    return flatten_yf_columns(df)
