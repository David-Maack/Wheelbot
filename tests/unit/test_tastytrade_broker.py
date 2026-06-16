"""TICKET-025 — TastytradeBroker.populate_quotes batch market-data merge.

These are pure unit tests: the SDK's get_market_data_by_type is reached only
through the `_market_data_by_type` seam, which we stub. No tastytrade install
and no network are required. The live integration path is covered (gated) in
tests/integration/test_tastytrade_sandbox.py.
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from core.models import OptionContract, OptionType
from platforms.tastytrade_broker import TastytradeBroker, _occ_dense, _occ_padded


def _broker() -> TastytradeBroker:
    b = TastytradeBroker(provider_secret="x", refresh_token="y")
    # Pre-seed the session/account so _ensure_session() never touches the network.
    b._session = object()
    b._account = object()
    return b


def _contract(occ: str, strike: float = 10.0) -> OptionContract:
    return OptionContract(
        underlying="F",
        occ_symbol=occ,
        strike=strike,
        expiration=date(2025, 6, 20),
        option_type=OptionType.PUT,
    )


def _md(symbol: str, **kw) -> SimpleNamespace:
    base = dict(bid=None, ask=None, mid=None, last=None, volume=None, open_interest=None)
    base.update(kw)
    return SimpleNamespace(symbol=symbol, **base)


@pytest.mark.asyncio
async def test_populate_quotes_fills_all_fields(monkeypatch):
    b = _broker()

    async def fake(session, syms):
        assert syms == ["F250620P00010000"]
        return [
            _md("F250620P00010000", bid=0.50, ask=0.60, mid=0.55,
                last=0.54, volume=120, open_interest=345)
        ]

    monkeypatch.setattr(b, "_market_data_by_type", fake)
    out = await b.populate_quotes([_contract("F250620P00010000")])
    c = out[0]
    assert (c.bid, c.ask, c.mid, c.last) == (0.50, 0.60, 0.55, 0.54)
    assert (c.volume, c.open_interest) == (120, 345)


@pytest.mark.asyncio
async def test_populate_quotes_chunks_large_batches(monkeypatch):
    b = _broker()
    contracts = [_contract(f"F250620P{i:08d}") for i in range(200)]
    calls: list[int] = []

    async def fake(session, syms):
        calls.append(len(syms))
        return [_md(s, bid=1.0, ask=1.1) for s in syms]

    monkeypatch.setattr(b, "_market_data_by_type", fake)
    out = await b.populate_quotes(contracts)
    assert len(out) == 200
    assert all(c.bid == 1.0 and c.ask == 1.1 for c in out)
    # 200 / 90 → three pages of 90, 90, 20.
    assert calls == [90, 90, 20]


@pytest.mark.asyncio
async def test_populate_quotes_missing_symbol_keeps_none(monkeypatch):
    b = _broker()

    async def fake(session, syms):
        return []  # snapshot returns nothing for the requested symbols

    monkeypatch.setattr(b, "_market_data_by_type", fake)
    out = await b.populate_quotes([_contract("F250620P00010000")])
    assert out[0].bid is None and out[0].ask is None


@pytest.mark.asyncio
async def test_populate_quotes_degrades_on_snapshot_error(monkeypatch):
    b = _broker()

    async def boom(session, syms):
        raise RuntimeError("502 Bad Gateway")

    monkeypatch.setattr(b, "_market_data_by_type", boom)
    contracts = [_contract("F250620P00010000")]
    out = await b.populate_quotes(contracts)
    # No raise; original contracts returned unquoted.
    assert out is contracts
    assert out[0].bid is None


@pytest.mark.asyncio
async def test_populate_quotes_empty_input():
    b = _broker()
    assert await b.populate_quotes([]) == []


# -- _occ_padded: TICKET-028 (orders need the space-padded 21-char OCC) -------


def test_occ_padded_pads_root_to_six():
    assert _occ_padded("F260717P00005000") == "F     260717P00005000"
    assert _occ_padded("AAPL260116C00150000") == "AAPL  260116C00150000"
    assert _occ_padded("GOOGL260116C00150000") == "GOOGL 260116C00150000"
    assert _occ_padded("SPXW260116C04000000") == "SPXW  260116C04000000"


def test_occ_padded_roundtrips_with_dense():
    padded = "F     260717P00005000"
    assert _occ_padded(_occ_dense(padded)) == padded
    assert _occ_dense(_occ_padded("F260717P00005000")) == "F260717P00005000"


def test_occ_padded_leaves_tickers_untouched():
    assert _occ_padded("F") == "F"
    assert _occ_padded("AAPL") == "AAPL"
    assert _occ_padded("") == ""
