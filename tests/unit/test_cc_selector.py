"""strategies/cc_selector — same shape as CSP, plus the strike-floor rule."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from core.models import OptionContract, OptionType, Quote, UniverseEntry
from platforms.paper_broker import PaperBroker
from strategies.cc_selector import select_cc


def _call(occ: str, strike: float, days_out: int, *, delta: float, mid: float) -> OptionContract:
    today = date(2025, 6, 1)
    spread = max(mid * 0.02, 0.01)
    return OptionContract(
        underlying="F",
        occ_symbol=occ,
        strike=strike,
        expiration=today + timedelta(days=days_out),
        option_type=OptionType.CALL,
        bid=mid - spread / 2,
        ask=mid + spread / 2,
        delta=delta,
        open_interest=1000,
        volume=200,
    )


def _config() -> dict:
    return {
        "account": {"id": "test", "broker": "paper"},
        "wheel": {
            "cc_delta_min": 0.20,
            "cc_delta_max": 0.30,
            "dte_min": 30,
            "dte_max": 45,
            "open_interest_min": 100,
            "volume_min": 50,
            "bid_ask_spread_max_pct": 10.0,
        },
    }


def _universe() -> dict:
    entry = UniverseEntry(symbol="F", name="Ford", tier=1, overrides={})
    return {"tickers": [entry], "banned": [], "banned_rules": []}


@pytest.mark.asyncio
async def test_rejects_strikes_below_cost_basis_even_if_higher_yield():
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F", bid=10.0, ask=10.04))
    broker.seed_chain(
        "F",
        [
            _call("under_basis", 9.5, 35, delta=0.25, mid=0.80),  # higher yield but under cb
            _call("at_basis", 10.0, 35, delta=0.25, mid=0.40),
            _call("above_basis", 10.5, 35, delta=0.22, mid=0.30),
        ],
    )
    chosen = await select_cc(broker, "F", 10.0, _config(), _universe(), today=date(2025, 6, 1))
    assert chosen is not None
    # at_basis has higher yield than above_basis; both legal.
    assert chosen.occ_symbol == "at_basis"


@pytest.mark.asyncio
async def test_returns_none_when_all_strikes_below_basis():
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F", bid=10.0, ask=10.04))
    broker.seed_chain(
        "F",
        [
            _call("under1", 9.0, 35, delta=0.25, mid=0.40),
            _call("under2", 9.5, 35, delta=0.22, mid=0.30),
        ],
    )
    chosen = await select_cc(broker, "F", 10.0, _config(), _universe(), today=date(2025, 6, 1))
    assert chosen is None


@pytest.mark.asyncio
async def test_picks_highest_yield_among_legal_strikes():
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F", bid=10.0, ask=10.04))
    broker.seed_chain(
        "F",
        [
            _call("good_yield", 10.5, 35, delta=0.25, mid=0.50),
            _call("better_yield", 11.0, 35, delta=0.23, mid=0.60),  # strictly higher mid/strike
            _call("low_yield", 12.0, 35, delta=0.22, mid=0.20),
        ],
    )
    chosen = await select_cc(broker, "F", 10.0, _config(), _universe(), today=date(2025, 6, 1))
    assert chosen is not None
    assert chosen.occ_symbol == "better_yield"
