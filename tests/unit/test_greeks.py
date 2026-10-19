"""Sanity checks for data/greeks.py.

Focus on properties that catch BS bugs without pinning to specific reference values:
- Deep OTM ≈ 0
- Deep ITM ≈ intrinsic discounted by carry
- Put-call parity
- Monotonic delta (calls go up with spot, puts go down with strike)
- IV solver round-trip
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from core.models import OptionType
from data.greeks import (
    bs_delta,
    bs_price,
    bs_theta,
    bs_vega,
    fill_greeks,
    implied_volatility,
)


def test_deep_otm_put_near_zero():
    # 30-day put with spot way above strike → nearly worthless.
    price = bs_price(S=100, K=50, T=30 / 365, sigma=0.30, option_type=OptionType.PUT)
    assert 0 <= price < 0.01


def test_deep_itm_put_near_intrinsic():
    # 30-day put with spot way below strike → ~ K - S (discounted).
    price = bs_price(S=10, K=100, T=30 / 365, sigma=0.30, option_type=OptionType.PUT)
    intrinsic = 100 - 10
    assert price == pytest.approx(intrinsic, rel=0.02)


def test_put_call_parity():
    # C - P = S - K * exp(-rT)  for European, no dividends.
    S, K, T, sigma, r = 100, 100, 0.25, 0.30, 0.045
    c = bs_price(S, K, T, sigma, OptionType.CALL, r=r)
    p = bs_price(S, K, T, sigma, OptionType.PUT, r=r)
    expected = S - K * math.exp(-r * T)
    assert (c - p) == pytest.approx(expected, abs=1e-6)


def test_call_delta_in_range_and_monotonic():
    deltas = [
        bs_delta(S=S, K=100, T=0.25, sigma=0.30, option_type=OptionType.CALL)
        for S in (60, 80, 100, 120, 140)
    ]
    assert all(0 <= d <= 1 for d in deltas)
    assert deltas == sorted(deltas)


def test_put_delta_negative_and_monotonic():
    deltas = [
        bs_delta(S=S, K=100, T=0.25, sigma=0.30, option_type=OptionType.PUT)
        for S in (60, 80, 100, 120, 140)
    ]
    assert all(-1 <= d <= 0 for d in deltas)
    assert deltas == sorted(deltas)  # rises (becomes less negative) as spot rises


def test_theta_negative_for_options_with_time():
    t = bs_theta(S=100, K=100, T=0.25, sigma=0.30, option_type=OptionType.CALL)
    assert t < 0


def test_vega_positive():
    v = bs_vega(S=100, K=100, T=0.25, sigma=0.30)
    assert v > 0


def test_iv_solver_round_trip():
    S, K, T, true_iv = 100, 105, 30 / 365, 0.32
    market = bs_price(S, K, T, true_iv, OptionType.PUT)
    solved = implied_volatility(market, S, K, T, OptionType.PUT)
    assert solved is not None
    assert solved == pytest.approx(true_iv, abs=1e-3)


def test_iv_solver_returns_none_for_arb_violation():
    # A put price below intrinsic discounted is impossible → solver returns None.
    solved = implied_volatility(target_price=0.0001, S=10, K=100, T=0.1, option_type=OptionType.PUT)
    assert solved is None


def test_fill_greeks_uses_supplied_iv():
    today = date(2025, 6, 1)
    expiry = today + timedelta(days=30)
    res = fill_greeks(
        underlying_price=100,
        strike=100,
        expiration=expiry,
        option_type=OptionType.PUT,
        market_price=None,
        today=today,
        iv=0.30,
    )
    assert res is not None
    assert res.iv == 0.30
    assert -0.6 < res.delta < -0.4  # ATM put delta hovers ~ -0.5


def test_fill_greeks_solves_iv_when_only_market_price_given():
    today = date(2025, 6, 1)
    expiry = today + timedelta(days=30)
    res = fill_greeks(
        underlying_price=100,
        strike=95,
        expiration=expiry,
        option_type=OptionType.PUT,
        market_price=1.20,
        today=today,
    )
    assert res is not None
    assert res.iv is not None
    assert 0.10 < res.iv < 1.0


# -- 2026-10-19: expiration-day (0DTE) time to expiry -----------------------


def test_years_to_expiry_same_day_uses_time_to_close():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from data.greeks import years_to_expiry

    et = ZoneInfo("America/New_York")
    d = date(2026, 10, 19)
    # Old behaviour preserved when no clock is passed.
    assert years_to_expiry(d, d) == 0.0
    # 10:00 ET -> 6h to the 16:00 close.
    t = years_to_expiry(d, d, now=datetime(2026, 10, 19, 10, 0, tzinfo=et))
    assert t == pytest.approx(6 / (365 * 24))
    # Naive datetimes are read as UTC: 14:00Z == 10:00 EDT.
    assert years_to_expiry(d, d, now=datetime(2026, 10, 19, 14, 0)) == pytest.approx(t)
    # At/after the close -> expired.
    assert years_to_expiry(d, d, now=datetime(2026, 10, 19, 16, 0, tzinfo=et)) == 0.0
    assert years_to_expiry(d, d, now=datetime(2026, 10, 19, 17, 30, tzinfo=et)) == 0.0
    # DTE >= 1 unchanged by `now` (whole days).
    assert years_to_expiry(date(2026, 10, 23), d,
                           now=datetime(2026, 10, 19, 10, 0, tzinfo=et)) == pytest.approx(4 / 365)


def test_fill_greeks_prices_0dte_contract_when_now_given():
    """The bug: on expiry day fill_greeks returned None for every contract,
    so strategies/zero_dte.py never saw a delta. With `now` it round-trips."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    d = date(2026, 10, 19)
    now = datetime(2026, 10, 19, 10, 5, tzinfo=et)
    T = (16 * 60 - (10 * 60 + 5)) / (365 * 24 * 60)
    # 1 sd over ~5.9h at 14% vol is ~$2.80 on SPY 766, so the 764P is ~0.24 delta.
    price = bs_price(766.0, 764.0, T, 0.14, OptionType.PUT)
    assert fill_greeks(766.0, 764.0, d, OptionType.PUT, price, d) is None  # old path
    res = fill_greeks(766.0, 764.0, d, OptionType.PUT, price, d, now=now)
    assert res is not None
    assert res.iv == pytest.approx(0.14, abs=1e-3)
    assert -0.30 < res.delta < -0.18
