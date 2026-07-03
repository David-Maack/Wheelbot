"""data/discovery — most-actives sourcing + chain tradability gate."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pytest

from core.models import OptionContract, OptionType
from data.discovery import discover_candidates, has_tradable_chain

CONFIG = {"universe_refresh": {"discovery": {"enabled": True, "top_n": 5}}}


@dataclass
class _Row:
    symbol: str


class _StubScreener:
    def __init__(self, symbols):
        self._symbols = symbols

    def get_most_actives(self, req):
        class _Resp:
            most_actives = [_Row(s) for s in self._symbols]

        return _Resp()


class _BoomScreener:
    def get_most_actives(self, req):
        raise RuntimeError("api down")


@pytest.mark.asyncio
async def test_discover_returns_plain_symbols_only():
    client = _StubScreener(["NVDA", "TSLA", "BRK.B", "FOO.WS", "hood"])
    out = await discover_candidates(CONFIG, client=client)
    # BRK.B / FOO.WS filtered (non-plain tickers); lowercase normalized.
    assert out == ["NVDA", "TSLA", "HOOD"]


@pytest.mark.asyncio
async def test_discover_fails_open_to_empty():
    assert await discover_candidates(CONFIG, client=_BoomScreener()) == []


@pytest.mark.asyncio
async def test_discover_no_keys_fails_open(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET", raising=False)
    assert await discover_candidates(CONFIG) == []


def _contract(bid, ask):
    return OptionContract(
        underlying="X", occ_symbol="X260717P00010000", strike=10.0,
        expiration=date(2026, 7, 17), option_type=OptionType.PUT, bid=bid, ask=ask,
    )


class _ChainBroker:
    def __init__(self, contracts=None, raise_exc=False):
        self._contracts = contracts or []
        self._raise = raise_exc

    async def get_option_chain(self, underlying, expiration=None, option_type=None):
        if self._raise:
            raise RuntimeError("chain fetch failed")
        return self._contracts


@pytest.mark.asyncio
async def test_chain_tradable_when_any_tight_quote():
    broker = _ChainBroker([_contract(0.10, 0.90), _contract(1.00, 1.05)])  # 133% then ~5%
    assert await has_tradable_chain(broker, "X", spread_max_pct=15.0) is True


@pytest.mark.asyncio
async def test_chain_not_tradable_when_all_wide_or_unquoted():
    broker = _ChainBroker([_contract(0.10, 0.90), _contract(None, 1.0), _contract(0.0, 0.5)])
    assert await has_tradable_chain(broker, "X", spread_max_pct=15.0) is False


@pytest.mark.asyncio
async def test_chain_fetch_error_reads_as_not_tradable():
    assert await has_tradable_chain(_ChainBroker(raise_exc=True), "X", 15.0) is False
