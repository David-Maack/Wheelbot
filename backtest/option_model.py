"""Black-Scholes-modeled option leg for the swing backtest.

We have no historical option-chain data (`chain_snapshots` only holds ~6 weeks),
so the backtest *models* the option a signal would have bought: pick a strike at
a target delta and a target DTE off the SPY spot + implied vol at signal time,
then reprice it with Black-Scholes as SPY and IV move over the hold.

IV input comes from the engine (VIX as an ATM proxy). Everything here is pure —
it just wraps `data.greeks`. Per-contract dollars use the standard 100x
multiplier; commissions/slippage are applied by the engine, not here.

Limitations (documented so the report can flag them):
- Single ATM IV per leg — no vol skew/smile and no term structure. ITM and OTM
  both priced off the same sigma, so the ITM-vs-OTM comparison captures the
  delta/theta/vega *structure* difference but not skew richness. Good enough for
  a go/no-go read; refine with real chains only if we go live.
- European BS (SPY options are American) — early-exercise value is negligible at
  our 20-30 DTE, no-dividend-in-window band.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.models import OptionType
from data.greeks import bs_delta, bs_price

CONTRACT_MULTIPLIER = 100
_R = 0.045  # risk-free, matches data.greeks.DEFAULT_RISK_FREE_RATE


def _t(dte_days: float) -> float:
    return max(dte_days, 0.0) / 365.0


def select_strike(
    spot: float,
    dte_days: float,
    target_delta: float,
    option_type: OptionType,
    iv: float,
    *,
    step: float = 1.0,
    search_frac: float = 0.20,
    r: float = _R,
) -> float:
    """Strike whose |BS delta| is closest to `target_delta`, on a `step` grid.

    SPY lists $1 strikes, so `step=1.0`. Scans spot*(1±search_frac).
    `target_delta` is the magnitude (0.67 for ITM, 0.30 for OTM) regardless of
    call/put.
    """
    if spot <= 0 or iv <= 0 or dte_days <= 0:
        return round(spot / step) * step
    T = _t(dte_days)
    lo = round(spot * (1.0 - search_frac) / step) * step
    hi = round(spot * (1.0 + search_frac) / step) * step
    best_k, best_err = None, float("inf")
    k = lo
    while k <= hi + 1e-9:
        if k > 0:
            d = abs(bs_delta(spot, k, T, iv, option_type, r))
            err = abs(d - target_delta)
            if err < best_err:
                best_err, best_k = err, k
        k += step
    return best_k if best_k is not None else round(spot / step) * step


@dataclass(frozen=True, slots=True)
class OptionLeg:
    option_type: OptionType
    strike: float
    dte_at_entry: float
    entry_spot: float
    entry_iv: float
    entry_price: float  # per share
    contracts: int = 1

    @property
    def cost(self) -> float:
        """Total premium paid (debit) in dollars."""
        return self.entry_price * CONTRACT_MULTIPLIER * self.contracts


def open_leg(
    spot: float,
    dte_days: float,
    target_delta: float,
    option_type: OptionType,
    iv: float,
    contracts: int = 1,
    *,
    r: float = _R,
) -> OptionLeg:
    """Construct a modeled leg at the signal: pick the strike, price the entry."""
    strike = select_strike(spot, dte_days, target_delta, option_type, iv, r=r)
    price = bs_price(spot, strike, _t(dte_days), iv, option_type, r)
    return OptionLeg(
        option_type=option_type,
        strike=strike,
        dte_at_entry=dte_days,
        entry_spot=spot,
        entry_iv=iv,
        entry_price=price,
        contracts=contracts,
    )


def price_leg(leg: OptionLeg, spot: float, days_held: float, iv: float, *, r: float = _R) -> float:
    """Per-share BS value of the leg now: same strike, DTE reduced by the hold,
    repriced at the current spot and IV. Intrinsic at/after expiry."""
    remaining = leg.dte_at_entry - days_held
    return bs_price(spot, leg.strike, _t(remaining), iv, leg.option_type, r)


def leg_pnl(leg: OptionLeg, exit_price: float) -> float:
    """Dollar P&L for a long leg: (exit - entry) * 100 * contracts."""
    return (exit_price - leg.entry_price) * CONTRACT_MULTIPLIER * leg.contracts


def pnl_pct(leg: OptionLeg, exit_price: float) -> float:
    """Return on premium for a long leg (e.g. +0.5 = +50%)."""
    if leg.entry_price <= 0:
        return 0.0
    return (exit_price - leg.entry_price) / leg.entry_price
