"""Tests for the Black-Scholes-modeled option leg used by the swing backtest."""

from __future__ import annotations

import pytest

from backtest.option_model import (
    leg_pnl,
    open_leg,
    pnl_pct,
    price_leg,
    select_strike,
)
from core.models import OptionType


def test_select_strike_itm_call_below_spot():
    # ~0.67 delta call is in the money → strike below spot.
    k = select_strike(600.0, dte_days=25, target_delta=0.67, option_type=OptionType.CALL, iv=0.18)
    assert k < 600.0


def test_select_strike_otm_call_above_spot():
    # ~0.30 delta call is out of the money → strike above spot.
    k = select_strike(600.0, dte_days=25, target_delta=0.30, option_type=OptionType.CALL, iv=0.18)
    assert k > 600.0


def test_select_strike_put_uses_magnitude():
    # 0.67-magnitude put delta is ITM → strike ABOVE spot for a put.
    k = select_strike(600.0, dte_days=25, target_delta=0.67, option_type=OptionType.PUT, iv=0.18)
    assert k > 600.0


def test_open_leg_prices_entry_and_cost():
    leg = open_leg(600.0, 25, 0.67, OptionType.CALL, iv=0.18, contracts=2)
    assert leg.entry_price > 0
    assert leg.cost == pytest.approx(leg.entry_price * 100 * 2)
    assert leg.contracts == 2


def test_price_leg_decays_toward_intrinsic_at_expiry():
    leg = open_leg(600.0, 25, 0.67, OptionType.CALL, iv=0.18)
    # Hold to expiry with spot unchanged → value collapses to intrinsic (spot-strike).
    intrinsic = max(600.0 - leg.strike, 0.0)
    at_expiry = price_leg(leg, spot=600.0, days_held=25, iv=0.18)
    assert at_expiry == pytest.approx(intrinsic, abs=1e-6)
    # And it's worth less than at entry (time value bled off).
    assert at_expiry < leg.entry_price


def test_pnl_sign_and_pct():
    leg = open_leg(600.0, 25, 0.67, OptionType.CALL, iv=0.18)
    up = price_leg(leg, spot=615.0, days_held=2, iv=0.18)  # SPY rallied
    assert leg_pnl(leg, up) > 0
    assert pnl_pct(leg, up) == pytest.approx((up - leg.entry_price) / leg.entry_price)
    down = price_leg(leg, spot=585.0, days_held=2, iv=0.18)  # SPY fell
    assert leg_pnl(leg, down) < 0
