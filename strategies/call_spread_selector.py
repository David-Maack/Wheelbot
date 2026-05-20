"""Bear call credit spread selector.

Mirror of `strategies/spread_selector.py` but on the call side:

  1. Run the same chain filter on calls to find candidate short legs
     (abs-delta band, OI/volume, spread, DTE band).
  2. Rank short candidates by annualized premium yield (highest first).
  3. For each, look for a long call at strike = short.strike + wing_width
     in the same expiry. If exact strike isn't listed, take the nearest
     strike at or above the target.
  4. Compute net credit = short.mid - long.mid; max_loss = (width - credit) * 100.
  5. Reject if net_credit / width < min_credit_pct (defined-risk economics).
  6. Return the first surviving candidate (best yield that also has a usable
     long leg and acceptable credit-to-width).

`ChainFilters.delta_*` is interpreted as an absolute-value band in
`data.chain`, so the same `short_delta_min=0.20, short_delta_max=0.30`
configuration applies to calls (positive delta) and puts (negative delta).

Returns a `SpreadCandidate` from `spread_selector` — fields are
direction-agnostic, just `short`/`long`/credit/width/max_loss/yield.

Out of scope (handled elsewhere):
  - IVR gate (selector caller can add it later if needed)
  - Earnings blackout, BP floor, concurrent caps (risk/limits.py)
  - Regime gate (risk/limits.py dispatches on direction → bear_calls_allowed)
  - Per-ticker overrides via universe.yaml (not yet wired for spreads)
"""

from __future__ import annotations

from datetime import date
from typing import Any, Awaitable, Callable

from core.broker import Broker
from core.checkpoint import log_checkpoint
from core.models import OptionContract, OptionType
from data.chain import (
    ChainFilters,
    annualized_yield,
    fetch_filtered_chain,
)
from data.ivr import IVRProvider
from strategies.spread_selector import SpreadCandidate

ChainRecorder = Callable[[str, str, list[OptionContract]], Awaitable[None]]


def _mid(c: OptionContract) -> float | None:
    if c.bid is None or c.ask is None:
        return None
    mid = (c.bid + c.ask) / 2
    return mid if mid > 0 else None


def _find_long_leg(
    chain: list[OptionContract],
    short: OptionContract,
    wing_width: float,
) -> OptionContract | None:
    """Pick the long leg for a bear call spread: strike = short + wing_width.

    Falls back to the nearest available strike at or above the target so
    a missing exact-width strike doesn't block the trade. Returns None
    when nothing in the chain satisfies the constraint.
    """
    target = short.strike + wing_width
    same_expiry = [
        c for c in chain
        if c.expiration == short.expiration
        and c.option_type == OptionType.CALL
        and c.strike > short.strike
        and c.bid is not None
        and c.ask is not None
    ]
    if not same_expiry:
        return None
    # Prefer exact match, otherwise the nearest strike at or above target.
    same_expiry.sort(key=lambda c: c.strike)  # lowest strike first
    at_or_above = [c for c in same_expiry if c.strike >= target]
    if at_or_above:
        # Closest from above = lowest strike that is ≥ target.
        return at_or_above[0]
    # All available strikes are below target — take the highest available
    # (smallest possible width) so we still have *some* long protection.
    return same_expiry[-1]


async def select_bear_call_spread(
    broker: Broker,
    symbol: str,
    params: dict[str, Any],
    *,
    today: date | None = None,
    record_chain: ChainRecorder | None = None,
    ivr: IVRProvider | None = None,
) -> SpreadCandidate | None:
    """Return the best bear-call spread, or None if nothing passes."""
    today = today or date.today()

    # IVR soft filter (Sprint 12 sub-sprint 5). Same shape as bull-put.
    if ivr is not None:
        rank = await ivr.iv_rank(symbol)
        ivr_min = float(params.get("ivr_min", 0))
        if rank is not None and rank < ivr_min:
            log_checkpoint(
                "call_spread_skip_ivr",
                status="ok",
                symbol=symbol,
                ivr=rank,
                ivr_min=ivr_min,
            )
            return None

    filters = ChainFilters(
        dte_min=int(params.get("dte_min", 30)),
        dte_max=int(params.get("dte_max", 45)),
        delta_min=float(params.get("short_delta_min", 0.20)),
        delta_max=float(params.get("short_delta_max", 0.30)),
        open_interest_min=int(params.get("open_interest_min", 0)),
        volume_min=int(params.get("volume_min", 0)),
        bid_ask_spread_max_pct=float(params.get("bid_ask_spread_max_pct", 100.0)),
    )
    wing_width = float(params.get("spread_width_dollars", 1.0))
    min_credit_pct = float(params.get("min_credit_pct_of_width", 25.0))

    short_candidates = await fetch_filtered_chain(
        broker, symbol, "call", filters, today=today
    )
    if record_chain is not None:
        await record_chain(symbol, "call", short_candidates)
    if not short_candidates:
        log_checkpoint("call_spread_no_short_candidates", status="ok", symbol=symbol)
        return None

    # Pull the wider call chain (no delta filter) once so we can find long
    # legs that would otherwise be filtered out by the short-leg delta band.
    wide_filters = ChainFilters(
        dte_min=filters.dte_min,
        dte_max=filters.dte_max,
        delta_min=0.0,
        delta_max=1.0,
        open_interest_min=0,
        volume_min=0,
        bid_ask_spread_max_pct=100.0,
    )
    full_chain = await fetch_filtered_chain(
        broker, symbol, "call", wide_filters, today=today
    )

    scored = []
    for short in short_candidates:
        y = annualized_yield(short, today)
        if y is None:
            continue
        scored.append((y, short))
    if not scored:
        log_checkpoint("call_spread_no_short_scored", status="ok", symbol=symbol)
        return None
    scored.sort(key=lambda pair: pair[0], reverse=True)

    for short_yield, short in scored:
        long = _find_long_leg(full_chain, short, wing_width)
        if long is None:
            continue
        short_mid = _mid(short)
        long_mid = _mid(long)
        if short_mid is None or long_mid is None:
            continue
        net_credit = short_mid - long_mid
        if net_credit <= 0:
            continue
        actual_width = long.strike - short.strike
        if actual_width <= 0:
            continue
        credit_pct = (net_credit / actual_width) * 100.0
        if credit_pct < min_credit_pct:
            log_checkpoint(
                "call_spread_skip_low_credit",
                status="ok",
                symbol=symbol,
                short_strike=short.strike,
                long_strike=long.strike,
                credit=net_credit,
                width=actual_width,
                credit_pct=credit_pct,
                min_credit_pct=min_credit_pct,
            )
            continue
        max_loss = (actual_width - net_credit) * 100.0
        log_checkpoint(
            "call_spread_selected",
            status="ok",
            symbol=symbol,
            short_occ=short.occ_symbol,
            long_occ=long.occ_symbol,
            short_strike=short.strike,
            long_strike=long.strike,
            credit=net_credit,
            width=actual_width,
            max_loss=max_loss,
            short_delta=short.delta,
        )
        return SpreadCandidate(
            short=short,
            long=long,
            net_credit_per_spread=net_credit,
            max_loss_per_spread=max_loss,
            width_dollars=actual_width,
            short_yield=short_yield,
        )

    log_checkpoint("call_spread_no_viable_pair", status="ok", symbol=symbol)
    return None
