"""data/chain.py: filter behavior + Greeks fallback."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from core.models import OptionContract, OptionType, Quote
from data.chain import ChainFilters, annualized_yield, fetch_filtered_chain
from platforms.paper_broker import PaperBroker


def _put(
    occ: str,
    strike: float,
    days_out: int,
    *,
    delta: float | None = -0.25,
    bid: float | None = 0.50,
    ask: float | None = 0.55,
    oi: int | None = 1000,
    vol: int | None = 200,
) -> OptionContract:
    today = date(2025, 6, 1)
    return OptionContract(
        underlying="F",
        occ_symbol=occ,
        strike=strike,
        expiration=today + timedelta(days=days_out),
        option_type=OptionType.PUT,
        bid=bid,
        ask=ask,
        delta=delta,
        open_interest=oi,
        volume=vol,
    )


@pytest.mark.asyncio
async def test_dte_filter_drops_outside_band():
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F", bid=10.0, ask=10.04))
    broker.seed_chain(
        "F",
        [
            _put("a", 10.0, days_out=10),  # too short
            _put("b", 10.0, days_out=35),  # in band
            _put("c", 10.0, days_out=60),  # too long
        ],
    )
    filt = ChainFilters(dte_min=30, dte_max=45, delta_min=0.10, delta_max=0.40)
    out = await fetch_filtered_chain(broker, "F", "put", filt, today=date(2025, 6, 1))
    assert [c.occ_symbol for c in out] == ["b"]


@pytest.mark.asyncio
async def test_delta_band_uses_absolute_value_for_puts():
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F", bid=10.0, ask=10.04))
    broker.seed_chain(
        "F",
        [
            _put("low", 8.0, days_out=35, delta=-0.10),  # too low
            _put("mid", 10.0, days_out=35, delta=-0.25),  # in band
            _put("hi", 12.0, days_out=35, delta=-0.45),  # too high
        ],
    )
    filt = ChainFilters(dte_min=30, dte_max=45, delta_min=0.20, delta_max=0.30)
    out = await fetch_filtered_chain(broker, "F", "put", filt, today=date(2025, 6, 1))
    assert [c.occ_symbol for c in out] == ["mid"]


@pytest.mark.asyncio
async def test_oi_volume_and_spread_gates():
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F", bid=10.0, ask=10.04))
    broker.seed_chain(
        "F",
        [
            _put("ok", 10.0, days_out=35),
            _put("low_oi", 10.0, days_out=35, oi=10),
            _put("low_vol", 10.0, days_out=35, vol=10),
            _put("wide", 10.0, days_out=35, bid=0.50, ask=1.00),  # 67% spread
        ],
    )
    filt = ChainFilters(
        dte_min=30,
        dte_max=45,
        delta_min=0.10,
        delta_max=0.40,
        open_interest_min=500,
        volume_min=100,
        bid_ask_spread_max_pct=10.0,
    )
    out = await fetch_filtered_chain(broker, "F", "put", filt, today=date(2025, 6, 1))
    assert [c.occ_symbol for c in out] == ["ok"]


@pytest.mark.asyncio
async def test_greeks_fallback_when_broker_omits_delta():
    """Missing delta should be filled from BS via the underlying quote."""
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F", bid=10.00, ask=10.04, last=10.02))
    broker.seed_chain(
        "F",
        [
            _put("solveme", strike=9.0, days_out=35, delta=None, bid=0.10, ask=0.12),
            _put("solve2", strike=10.0, days_out=35, delta=None, bid=0.30, ask=0.35),
        ],
    )
    filt = ChainFilters(dte_min=30, dte_max=45, delta_min=0.05, delta_max=0.60)
    out = await fetch_filtered_chain(broker, "F", "put", filt, today=date(2025, 6, 1))
    assert len(out) >= 1
    for c in out:
        assert c.delta is not None and -1 <= c.delta <= 0


@pytest.mark.asyncio
async def test_empty_chain_returns_empty():
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F", bid=10.0, ask=10.04))
    broker.seed_chain("F", [])
    filt = ChainFilters(dte_min=30, dte_max=45, delta_min=0.20, delta_max=0.30)
    out = await fetch_filtered_chain(broker, "F", "put", filt, today=date(2025, 6, 1))
    assert out == []


def test_annualized_yield_matches_formula():
    today = date(2025, 6, 1)
    c = _put("y", strike=10.0, days_out=30, bid=0.20, ask=0.30)
    # mid = 0.25, mid/strike = 0.025, * (365/30) ≈ 0.304
    assert annualized_yield(c, today) == pytest.approx(0.025 * (365 / 30), rel=1e-4)


def test_annualized_yield_none_for_expired():
    today = date(2025, 6, 1)
    c = _put("y", strike=10.0, days_out=0, bid=0.20, ask=0.30)
    assert annualized_yield(c, today) is None
