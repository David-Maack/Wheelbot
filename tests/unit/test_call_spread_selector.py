"""Bear call spread selector — selection logic only.

Orchestrator + router multi-leg tests come in sub-sprint 3 when
strategies/spreads.py is generalized to dispatch on direction.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from core.models import OptionContract, OptionType, Quote
from platforms.paper_broker import PaperBroker
from strategies.call_spread_selector import select_bear_call_spread


def _call(strike: float, *, bid: float, ask: float, delta: float = 0.25) -> OptionContract:
    today = date(2025, 6, 1)
    occ = f"F250706C{int(strike * 1000):08d}"
    return OptionContract(
        underlying="F",
        occ_symbol=occ,
        strike=strike,
        expiration=today + timedelta(days=35),
        option_type=OptionType.CALL,
        bid=bid,
        ask=ask,
        delta=delta,
        open_interest=1000,
        volume=200,
    )


def _params(**overrides) -> dict:
    base = {
        "dte_min": 30,
        "dte_max": 45,
        "short_delta_min": 0.20,
        "short_delta_max": 0.30,
        "spread_width_dollars": 1.0,
        "min_credit_pct_of_width": 25.0,
        "open_interest_min": 100,
        "volume_min": 50,
        "bid_ask_spread_max_pct": 10.0,
    }
    base.update(overrides)
    return base


# -- Selection happy path ----------------------------------------------------


@pytest.mark.asyncio
async def test_selector_picks_short_low_long_high_with_target_width():
    """Bear call: short at lower strike (closer to ATM), long at higher strike."""
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F", bid=10.0, ask=10.04))
    broker.seed_chain(
        "F",
        [
            _call(10.0, bid=0.39, ask=0.41, delta=0.25),  # short candidate
            _call(11.0, bid=0.10, ask=0.12, delta=0.10),  # long target
        ],
    )
    candidate = await select_bear_call_spread(
        broker, "F", _params(), today=date(2025, 6, 1)
    )
    assert candidate is not None
    assert candidate.short.strike == 10.0
    assert candidate.long.strike == 11.0
    assert candidate.width_dollars == pytest.approx(1.0)
    # short mid 0.40, long mid 0.11 → credit 0.29; max loss = (1.0 - 0.29) * 100 = 71.
    assert candidate.net_credit_per_spread == pytest.approx(0.29)
    assert candidate.max_loss_per_spread == pytest.approx(71.0)


# -- Credit gate -------------------------------------------------------------


@pytest.mark.asyncio
async def test_selector_rejects_when_credit_below_min_pct():
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F", bid=10.0, ask=10.04))
    broker.seed_chain(
        "F",
        [
            _call(10.0, bid=0.30, ask=0.32, delta=0.25),  # short mid 0.31
            _call(11.0, bid=0.20, ask=0.22, delta=0.10),  # long mid 0.21 → credit 0.10 → 10% < 25%
        ],
    )
    candidate = await select_bear_call_spread(
        broker, "F", _params(), today=date(2025, 6, 1)
    )
    assert candidate is None


# -- Strike fallback when exact width is missing ------------------------------


@pytest.mark.asyncio
async def test_selector_falls_back_to_nearest_strike_above_target():
    """Target was 11.0 with width=1.0, but only 11.5 is listed."""
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F", bid=10.0, ask=10.04))
    broker.seed_chain(
        "F",
        [
            _call(10.0, bid=0.39, ask=0.41, delta=0.25),
            _call(11.5, bid=0.05, ask=0.07, delta=0.05),
        ],
    )
    # 1.5-wide spread, credit 0.34 → 22.7% < 25% default. Loosen the gate.
    candidate = await select_bear_call_spread(
        broker, "F", _params(min_credit_pct_of_width=20.0), today=date(2025, 6, 1)
    )
    assert candidate is not None
    assert candidate.long.strike == 11.5
    assert candidate.width_dollars == pytest.approx(1.5)


# -- Delta band filter (no short candidate survives) --------------------------


@pytest.mark.asyncio
async def test_selector_returns_none_when_no_short_passes():
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F", bid=10.0, ask=10.04))
    broker.seed_chain(
        "F",
        [
            # Calls way OTM — abs(delta) < 0.20 band.
            _call(10.0, bid=0.39, ask=0.41, delta=0.05),
            _call(11.0, bid=0.10, ask=0.12, delta=0.02),
        ],
    )
    candidate = await select_bear_call_spread(
        broker, "F", _params(), today=date(2025, 6, 1)
    )
    assert candidate is None


# -- Long leg constraints ----------------------------------------------------


@pytest.mark.asyncio
async def test_selector_returns_none_when_no_long_above_short():
    """If chain only has strikes ≤ short, there's no valid bear-call long leg."""
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F", bid=10.0, ask=10.04))
    broker.seed_chain(
        "F",
        [
            _call(10.0, bid=0.39, ask=0.41, delta=0.25),
            # Only a lower strike exists — invalid for bear-call (would be a debit spread).
            _call(9.0, bid=1.20, ask=1.22, delta=0.60),
        ],
    )
    candidate = await select_bear_call_spread(
        broker, "F", _params(), today=date(2025, 6, 1)
    )
    assert candidate is None


# -- Yield-based ranking -----------------------------------------------------


@pytest.mark.asyncio
async def test_selector_rejects_when_ivr_below_min():
    """Sprint 12 sub-sprint 5: IVR < ivr_min should block entry."""
    class _StubIvr:
        async def iv_rank(self, symbol):
            return 22.0  # below the 30 threshold

    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F", bid=10.0, ask=10.04))
    broker.seed_chain(
        "F",
        [
            _call(10.0, bid=0.39, ask=0.41, delta=0.25),
            _call(11.0, bid=0.10, ask=0.12, delta=0.10),
        ],
    )
    candidate = await select_bear_call_spread(
        broker, "F", _params(ivr_min=30), today=date(2025, 6, 1), ivr=_StubIvr(),
    )
    assert candidate is None


@pytest.mark.asyncio
async def test_selector_passes_when_ivr_unavailable():
    """No iv_history yet → iv_rank returns None → skip the filter (paper-bringup safety)."""
    class _StubIvr:
        async def iv_rank(self, symbol):
            return None

    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F", bid=10.0, ask=10.04))
    broker.seed_chain(
        "F",
        [
            _call(10.0, bid=0.39, ask=0.41, delta=0.25),
            _call(11.0, bid=0.10, ask=0.12, delta=0.10),
        ],
    )
    candidate = await select_bear_call_spread(
        broker, "F", _params(ivr_min=30), today=date(2025, 6, 1), ivr=_StubIvr(),
    )
    assert candidate is not None  # filter skipped, trade allowed


@pytest.mark.asyncio
async def test_selector_prefers_higher_yield_short_when_multiple_candidates():
    """Two valid short candidates in the delta band — pick the higher yielder."""
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F", bid=10.0, ask=10.04))
    broker.seed_chain(
        "F",
        [
            # Lower strike → higher mid → higher yield. Both in delta band.
            _call(10.0, bid=0.49, ask=0.51, delta=0.30),  # higher yield
            _call(10.5, bid=0.34, ask=0.36, delta=0.22),  # lower yield
            # Long legs (one per candidate).
            _call(11.0, bid=0.10, ask=0.12, delta=0.10),
            _call(11.5, bid=0.05, ask=0.07, delta=0.05),
        ],
    )
    candidate = await select_bear_call_spread(
        broker, "F", _params(), today=date(2025, 6, 1)
    )
    assert candidate is not None
    assert candidate.short.strike == 10.0  # higher-yield short won
