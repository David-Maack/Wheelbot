"""Black-Scholes pricing, Greeks, and IV solver.

Pure functions. No state, no I/O. Used in two places:

1. As a fallback inside `data/chain.py` when the broker doesn't supply Greeks.
2. By `scripts/ingest_history.py` to compute ATM 30-day IV from an option price.

Conventions:
- All rates and IVs are decimal (0.05 = 5%, 0.30 = 30%).
- T is years to expiry (DTE / 365).
- No dividend yield in v1 — the universe is dividend-paying but the effect at our
  DTE band (30–45) is small enough to ignore for selection. Add `q` later if needed.
- Risk-free rate defaults to 4.5%; override per call from config.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from zoneinfo import ZoneInfo

from core.models import OptionType

DEFAULT_RISK_FREE_RATE = 0.045
_SQRT_2PI = math.sqrt(2 * math.pi)
_ET = ZoneInfo("America/New_York")
_EXPIRY_CLOSE_ET = time(16, 0)  # equity/ETF options stop trading and settle at the 4pm close
_SECONDS_PER_YEAR = 365.0 * 86400.0


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))


def years_to_expiry(expiration: date, today: date, now: datetime | None = None) -> float:
    """Calendar-year fraction. <=0 means already expired.

    Whole days for DTE >= 1 (unchanged — selection at 30-45 DTE doesn't care
    about hours). On expiration day the whole-day count is 0, which made every
    same-day contract look expired: fill_greeks returned None, no 0DTE contract
    ever got a delta, and strategies/zero_dte.py logged
    zero_dte_no_short_candidates on every tick from July to 2026-10-19. When
    the caller passes `now`, expiration day instead returns the real time left
    until the 16:00 ET close. Without `now` it stays 0 (old behaviour), so
    only callers that opt in see intraday values.
    """
    days = (expiration - today).days
    if days == 0 and now is not None:
        moment = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
        close = datetime.combine(expiration, _EXPIRY_CLOSE_ET, tzinfo=_ET)
        return max((close - moment).total_seconds(), 0.0) / _SECONDS_PER_YEAR
    return max(days, 0) / 365.0


def bs_price(
    S: float,
    K: float,
    T: float,
    sigma: float,
    option_type: OptionType,
    r: float = DEFAULT_RISK_FREE_RATE,
) -> float:
    """Black-Scholes price for a European option. Returns intrinsic when T<=0 or sigma<=0."""
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0.0) if option_type == OptionType.CALL else max(K - S, 0.0)
        return intrinsic
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == OptionType.CALL:
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def bs_delta(
    S: float,
    K: float,
    T: float,
    sigma: float,
    option_type: OptionType,
    r: float = DEFAULT_RISK_FREE_RATE,
) -> float:
    """Per-share delta. Calls in [0,1], puts in [-1,0]."""
    if T <= 0 or sigma <= 0:
        if option_type == OptionType.CALL:
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    d1 = _d1(S, K, T, r, sigma)
    return _norm_cdf(d1) if option_type == OptionType.CALL else _norm_cdf(d1) - 1.0


def bs_theta(
    S: float,
    K: float,
    T: float,
    sigma: float,
    option_type: OptionType,
    r: float = DEFAULT_RISK_FREE_RATE,
) -> float:
    """Per-day theta (price change for 1 day decay). Negative for long options."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    term1 = -(S * _norm_pdf(d1) * sigma) / (2 * math.sqrt(T))
    if option_type == OptionType.CALL:
        annual = term1 - r * K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        annual = term1 + r * K * math.exp(-r * T) * _norm_cdf(-d2)
    return annual / 365.0


def bs_vega(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float = DEFAULT_RISK_FREE_RATE,
) -> float:
    """Vega per 1.00 (100%) change in IV. Divide by 100 for per-1% if you want."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, sigma)
    return S * _norm_pdf(d1) * math.sqrt(T)


def implied_volatility(
    target_price: float,
    S: float,
    K: float,
    T: float,
    option_type: OptionType,
    r: float = DEFAULT_RISK_FREE_RATE,
    *,
    tol: float = 1e-5,
    max_iter: int = 100,
) -> float | None:
    """Solve BS for sigma given a market price. Bisection — robust if slow."""
    if T <= 0 or target_price <= 0:
        return None
    intrinsic = (
        max(S - K * math.exp(-r * T), 0.0)
        if option_type == OptionType.CALL
        else max(K * math.exp(-r * T) - S, 0.0)
    )
    if target_price < intrinsic - tol:
        return None  # arb-violating quote, can't solve

    lo, hi = 1e-4, 5.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        price = bs_price(S, K, T, mid, option_type, r)
        if abs(price - target_price) < tol:
            return mid
        if price < target_price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass(frozen=True, slots=True)
class GreeksResult:
    delta: float
    theta: float
    vega: float
    iv: float | None
    price: float


def fill_greeks(
    underlying_price: float,
    strike: float,
    expiration: date,
    option_type: OptionType,
    market_price: float | None,
    today: date,
    *,
    iv: float | None = None,
    r: float = DEFAULT_RISK_FREE_RATE,
    now: datetime | None = None,
) -> GreeksResult | None:
    """Compute Greeks for a single contract. If `iv` is None, solve from market_price.

    Returns None when we can't get an IV (no market_price + no iv supplied).
    Pass `now` to price expiration-day contracts on time left to the close
    (see years_to_expiry).
    """
    T = years_to_expiry(expiration, today, now)
    if T <= 0:
        return None
    if iv is None:
        if market_price is None or market_price <= 0:
            return None
        iv = implied_volatility(market_price, underlying_price, strike, T, option_type, r)
        if iv is None:
            return None
    return GreeksResult(
        delta=bs_delta(underlying_price, strike, T, iv, option_type, r),
        theta=bs_theta(underlying_price, strike, T, iv, option_type, r),
        vega=bs_vega(underlying_price, strike, T, iv, r),
        iv=iv,
        price=bs_price(underlying_price, strike, T, iv, option_type, r),
    )
