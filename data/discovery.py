"""Market discovery — tier 0 of the weekly universe refresh.

Pulls the top-N most-active US stocks from Alpaca's screener API so the
refresh can consider names nobody hand-fed it. Volume ranking is the
liquidity proxy; everything else (price band, median volume, earnings
distance, option-chain tradability, the LLM's strategy-fit judgment, churn
caps, and the human approval gate) happens downstream in
intelligence/universe_refresh.py.

Fail-open contract: any error returns an empty list — a broken screener can
only shrink the candidate pool back to the hand-curated one, never block the
refresh.
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any

from core.broker import Broker
from core.checkpoint import log_checkpoint

# Plain 1-5 letter tickers only — skips units/warrants/preferreds (BRK.B,
# FOO.WS, ...) whose option chains are absent or untradeable anyway.
_PLAIN_SYMBOL = re.compile(r"^[A-Z]{1,5}$")


async def discover_candidates(config: dict[str, Any], *, client: Any = None) -> list[str]:
    """Top-N most-active US stock symbols by volume, most-active first.

    `client` is injectable for tests; by default an alpaca-py ScreenerClient is
    constructed from the same env keys the broker uses. Returns [] on ANY
    failure (missing keys, SDK error, network) — fail-open by design.
    """
    ur = config.get("universe_refresh", {}) or {}
    disc = ur.get("discovery", {}) or {}
    top_n = int(disc.get("top_n", 100))
    try:
        if client is None:
            from alpaca.data.historical.screener import ScreenerClient

            key = os.environ.get("ALPACA_API_KEY")
            secret = os.environ.get("ALPACA_API_SECRET")
            if not key or not secret:
                log_checkpoint("discovery_no_keys", status="skip")
                return []
            client = ScreenerClient(key, secret)
        from alpaca.data.requests import MostActivesRequest

        req = MostActivesRequest(by="volume", top=top_n)
        resp = await asyncio.to_thread(client.get_most_actives, req)
        rows = getattr(resp, "most_actives", None) or []
        symbols = []
        for row in rows:
            sym = str(getattr(row, "symbol", "")).upper()
            if _PLAIN_SYMBOL.match(sym):
                symbols.append(sym)
        log_checkpoint("discovery_most_actives", status="ok", requested=top_n, usable=len(symbols))
        return symbols
    except Exception as exc:  # noqa: BLE001 — discovery must never break the refresh
        log_checkpoint("discovery_fail", status="fail", error=str(exc))
        return []


async def has_tradable_chain(broker: Broker, symbol: str, spread_max_pct: float) -> bool:
    """True iff `symbol` has at least one option contract with a live two-sided
    quote whose bid-ask spread is <= spread_max_pct of the mid. This is the
    'does an options market actually exist here' gate for newly-discovered
    names — per-contract liquidity is still enforced at entry time."""
    try:
        contracts = await broker.get_option_chain(symbol)
    except Exception as exc:  # noqa: BLE001 — treat a failed fetch as not tradable
        log_checkpoint("discovery_chain_fail", status="skip", symbol=symbol, error=str(exc))
        return False
    for c in contracts:
        if c.bid is None or c.ask is None or c.bid <= 0 or c.ask <= 0:
            continue
        mid = (c.bid + c.ask) / 2
        if mid > 0 and (c.ask - c.bid) / mid * 100 <= spread_max_pct:
            return True
    return False
