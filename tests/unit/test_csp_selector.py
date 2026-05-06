"""strategies/csp_selector against PaperBroker fixtures."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from core.models import OptionContract, OptionType, Quote, UniverseEntry
from data.ivr import IVRProvider
from platforms.paper_broker import PaperBroker
from strategies.csp_selector import select_csp


class _NullIvr:
    """Stand-in that returns None for every symbol so the IVR gate is a no-op."""

    async def iv_rank(self, symbol: str) -> float | None:
        return None

    async def iv_percentile(self, symbol: str) -> float | None:
        return None


class _HighIvr:
    async def iv_rank(self, symbol: str) -> float | None:
        return 80.0

    async def iv_percentile(self, symbol: str) -> float | None:
        return 80.0


class _LowIvr:
    async def iv_rank(self, symbol: str) -> float | None:
        return 5.0

    async def iv_percentile(self, symbol: str) -> float | None:
        return 5.0


def _put(occ: str, strike: float, days_out: int, *, delta: float, mid: float) -> OptionContract:
    today = date(2025, 6, 1)
    spread = max(mid * 0.02, 0.01)
    return OptionContract(
        underlying="F",
        occ_symbol=occ,
        strike=strike,
        expiration=today + timedelta(days=days_out),
        option_type=OptionType.PUT,
        bid=mid - spread / 2,
        ask=mid + spread / 2,
        delta=delta,
        open_interest=1000,
        volume=200,
    )


def _config(**overrides) -> dict:
    base = {
        "account": {"id": "test", "broker": "paper"},
        "wheel": {
            "csp_delta_min": 0.20,
            "csp_delta_max": 0.30,
            "cc_delta_min": 0.20,
            "cc_delta_max": 0.30,
            "dte_min": 30,
            "dte_max": 45,
            "ivr_min": 30,
            "open_interest_min": 100,
            "volume_min": 50,
            "bid_ask_spread_max_pct": 10.0,
        },
    }
    base["wheel"].update(overrides)
    return base


def _universe(overrides_for_F: dict | None = None) -> dict:
    entry = UniverseEntry(symbol="F", name="Ford", tier=1, overrides=overrides_for_F or {})
    return {"tickers": [entry], "banned": [], "banned_rules": []}


@pytest.mark.asyncio
async def test_selects_highest_yield_in_band():
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F", bid=10.0, ask=10.04))
    broker.seed_chain(
        "F",
        [
            _put("low_yield", 9.0, 35, delta=-0.25, mid=0.20),
            _put("high_yield", 9.5, 35, delta=-0.27, mid=0.40),  # better mid → higher yield
        ],
    )
    chosen = await select_csp(broker, "F", _config(), _universe(), _NullIvr(), today=date(2025, 6, 1))
    assert chosen is not None
    assert chosen.occ_symbol == "high_yield"


@pytest.mark.asyncio
async def test_returns_none_when_nothing_passes_filters():
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F", bid=10.0, ask=10.04))
    broker.seed_chain(
        "F",
        [
            _put("too_low", 9.0, 35, delta=-0.10, mid=0.05),  # below delta band
            _put("too_short", 9.0, 10, delta=-0.25, mid=0.30),  # below DTE
        ],
    )
    chosen = await select_csp(broker, "F", _config(), _universe(), _NullIvr(), today=date(2025, 6, 1))
    assert chosen is None


@pytest.mark.asyncio
async def test_skips_when_ivr_below_min():
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F", bid=10.0, ask=10.04))
    broker.seed_chain("F", [_put("ok", 9.5, 35, delta=-0.27, mid=0.40)])
    chosen = await select_csp(
        broker, "F", _config(ivr_min=30), _universe(), _LowIvr(), today=date(2025, 6, 1)
    )
    assert chosen is None


@pytest.mark.asyncio
async def test_proceeds_when_ivr_above_min():
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F", bid=10.0, ask=10.04))
    broker.seed_chain("F", [_put("ok", 9.5, 35, delta=-0.27, mid=0.40)])
    chosen = await select_csp(
        broker, "F", _config(ivr_min=30), _universe(), _HighIvr(), today=date(2025, 6, 1)
    )
    assert chosen is not None
    assert chosen.occ_symbol == "ok"


@pytest.mark.asyncio
async def test_per_ticker_override_tightens_delta_max():
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="F", bid=10.0, ask=10.04))
    broker.seed_chain(
        "F",
        [
            _put("inside_override", 9.0, 35, delta=-0.22, mid=0.20),  # within tight band
            _put("outside_override", 9.5, 35, delta=-0.28, mid=0.40),  # exceeds 0.25 cap
        ],
    )
    universe = _universe(overrides_for_F={"csp_delta_max": 0.25})
    chosen = await select_csp(broker, "F", _config(), universe, _NullIvr(), today=date(2025, 6, 1))
    assert chosen is not None
    assert chosen.occ_symbol == "inside_override"
