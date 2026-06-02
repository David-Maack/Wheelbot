"""data/yf_helpers — single chokepoint for yfinance access.

Replaces two duplicated _flatten_yf_columns helpers (one in
intelligence/screener.py, one in risk/regime.py). Tests target the helpers
themselves rather than re-running the screener / regime classifier.
"""

from __future__ import annotations

import pandas as pd
import pytest

from data.yf_helpers import flatten_yf_columns, safe_history


def test_flatten_collapses_multiindex_columns():
    """yfinance >= 0.2.34 returns MultiIndex columns; the helper must flatten
    them so callers see `df["Close"]` as a Series, not a sub-DataFrame."""
    cols = pd.MultiIndex.from_tuples(
        [("Open", "SPY"), ("Close", "SPY"), ("Volume", "SPY")]
    )
    df = pd.DataFrame([[1.0, 2.0, 100]], columns=cols)
    out = flatten_yf_columns(df)
    assert list(out.columns) == ["Open", "Close", "Volume"]
    # `df["Close"]` is now a Series, not a sub-DataFrame.
    assert isinstance(out["Close"], pd.Series)


def test_flatten_noop_on_flat_columns():
    """A DataFrame with a plain Index is returned unchanged."""
    df = pd.DataFrame({"Open": [1.0], "Close": [2.0], "Volume": [100]})
    out = flatten_yf_columns(df)
    assert list(out.columns) == ["Open", "Close", "Volume"]
    assert isinstance(out["Close"], pd.Series)


def test_flatten_handles_non_dataframe_gracefully():
    """Defensive: objects without a `.columns` attribute (or a flat Index) are
    returned as-is. The helper never raises."""
    assert flatten_yf_columns(None) is None
    assert flatten_yf_columns("not a frame") == "not a frame"


def test_safe_history_returns_empty_on_network_error(monkeypatch):
    """yf.download raising must surface as an empty DataFrame plus a logged
    checkpoint — callers branch on `df.empty` without try/except."""
    def _boom(*args, **kwargs):
        raise RuntimeError("yfinance network unreachable")

    monkeypatch.setattr("data.yf_helpers.yf.download", _boom)
    df = safe_history("SPY", period="1y")
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_safe_history_returns_empty_when_yf_returns_none(monkeypatch):
    """yfinance occasionally returns None (e.g. delisted ticker, weekend)."""
    monkeypatch.setattr("data.yf_helpers.yf.download", lambda *a, **kw: None)
    df = safe_history("ZZZZ", period="1mo")
    assert df.empty


def test_safe_history_flattens_multiindex_result(monkeypatch):
    """Happy path: a MultiIndex result from yfinance is flattened before return."""
    cols = pd.MultiIndex.from_tuples([("Close", "SPY"), ("Open", "SPY")])
    fake = pd.DataFrame([[100.0, 99.0], [101.0, 100.5]], columns=cols)
    monkeypatch.setattr("data.yf_helpers.yf.download", lambda *a, **kw: fake)

    df = safe_history("SPY", period="1mo")
    assert not df.empty
    assert list(df.columns) == ["Close", "Open"]
    assert isinstance(df["Close"], pd.Series)
