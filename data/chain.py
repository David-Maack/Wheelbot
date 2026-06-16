"""Option-chain fetch + filter.

The strategy layer never touches the broker directly for chains — it goes through
`fetch_filtered_chain`, which:

1. Calls broker.get_option_chain for the underlying.
2. Fills missing Greeks from data/greeks.py using a current quote.
3. Applies the filters from spec §5: DTE band, delta band, OI floor, volume floor,
   bid-ask spread cap, contract type.

Returns OptionContracts. Selection (highest yield, etc) lives in the selectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

from core.broker import Broker
from core.checkpoint import log_checkpoint
from core.models import OptionContract, OptionType
from data.greeks import DEFAULT_RISK_FREE_RATE, fill_greeks

Side = Literal["put", "call"]


@dataclass(frozen=True, slots=True)
class ChainFilters:
    """Subset of wheel params that gate chain selection. Selectors translate
    the higher-level config into one of these per call."""

    dte_min: int
    dte_max: int
    delta_min: float
    delta_max: float
    open_interest_min: int = 0
    volume_min: int = 0
    bid_ask_spread_max_pct: float = 100.0  # generous default; selectors override
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE


def _spread_pct(c: OptionContract) -> float | None:
    if c.bid is None or c.ask is None or c.ask <= 0:
        return None
    mid = (c.bid + c.ask) / 2
    if mid <= 0:
        return None
    return ((c.ask - c.bid) / mid) * 100.0


def _abs_delta(d: float | None) -> float | None:
    if d is None:
        return None
    return abs(d)


def _within_dte(exp: date, today: date, lo: int, hi: int) -> bool:
    dte = (exp - today).days
    return lo <= dte <= hi


def _passes_liquidity(c: OptionContract, f: ChainFilters) -> bool:
    if (c.open_interest or 0) < f.open_interest_min:
        return False
    if (c.volume or 0) < f.volume_min:
        return False
    spread = _spread_pct(c)
    if spread is None or spread > f.bid_ask_spread_max_pct:
        return False
    return True


def _populate_greeks(
    contracts: list[OptionContract],
    underlying_price: float,
    today: date,
    r: float,
) -> list[OptionContract]:
    """Fill in missing delta/theta/vega/iv from BS using the contract's mid price."""
    out: list[OptionContract] = []
    for c in contracts:
        if c.delta is not None:
            # broker supplied — pass through, but stamp the underlying price for downstream use
            out.append(c.model_copy(update={"underlying_price": underlying_price}) if c.underlying_price is None else c)
            continue
        bid, ask = c.bid, c.ask
        market_price = c.last
        if market_price is None and bid is not None and ask is not None and ask > 0:
            market_price = (bid + ask) / 2
        result = fill_greeks(
            underlying_price=underlying_price,
            strike=c.strike,
            expiration=c.expiration,
            option_type=c.option_type,
            market_price=market_price,
            today=today,
            r=r,
        )
        if result is None:
            out.append(c)
            continue
        out.append(
            c.model_copy(
                update={
                    "delta": result.delta,
                    "theta": result.theta,
                    "vega": result.vega,
                    "iv": result.iv if c.iv is None else c.iv,
                    "underlying_price": underlying_price,
                }
            )
        )
    return out


async def fetch_filtered_chain(
    broker: Broker,
    underlying: str,
    side: Side,
    filters: ChainFilters,
    *,
    today: date | None = None,
    underlying_price: float | None = None,
) -> list[OptionContract]:
    """Return contracts on `underlying` matching `filters`.

    `underlying_price` is the spot used for Greeks fallback; if None, we ask the
    broker for a quote. Failing that, contracts without broker-supplied Greeks
    are dropped (we can't filter on delta we don't have).
    """
    today = today or date.today()
    opt_type = OptionType.PUT if side == "put" else OptionType.CALL

    raw = await broker.get_option_chain(underlying, option_type=opt_type)
    if not raw:
        log_checkpoint("chain_empty", status="ok", underlying=underlying, side=side)
        return []

    # DTE pre-filter (cheap, removes most contracts before Greeks math).
    in_window = [c for c in raw if _within_dte(c.expiration, today, filters.dte_min, filters.dte_max)]
    if not in_window:
        return []

    # Some brokers (Tastytrade) return structure-only chains; fetch quotes for
    # just this DTE window. No-op for brokers that already quote inline (Alpaca,
    # Paper), so the live path is unchanged.
    in_window = await broker.populate_quotes(in_window)

    if underlying_price is None:
        try:
            quote = await broker.get_quote(underlying)
            underlying_price = quote.mid if quote.mid is not None else (quote.last or quote.bid or quote.ask)
        except Exception:
            underlying_price = None

    if underlying_price is not None:
        with_greeks = _populate_greeks(in_window, underlying_price, today, filters.risk_free_rate)
    else:
        with_greeks = in_window

    out: list[OptionContract] = []
    for c in with_greeks:
        d = _abs_delta(c.delta)
        if d is None or d < filters.delta_min or d > filters.delta_max:
            continue
        if not _passes_liquidity(c, filters):
            continue
        out.append(c)
    return out


def annualized_yield(contract: OptionContract, today: date) -> float | None:
    """Premium yield annualized: (mid / strike) * (365 / DTE).

    Used by selectors to rank candidates. Returns None when mid or DTE invalid.
    """
    if contract.bid is None or contract.ask is None:
        return None
    mid = (contract.bid + contract.ask) / 2
    if mid <= 0 or contract.strike <= 0:
        return None
    dte = (contract.expiration - today).days
    if dte <= 0:
        return None
    return (mid / contract.strike) * (365.0 / dte)
