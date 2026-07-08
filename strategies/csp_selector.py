"""CSP selector — pick the best cash-secured put for one symbol.

Scoring: highest annualized premium yield = (mid / strike) * (365 / DTE).
Spec §5 says "premium yield"; the annualized form makes 30-DTE and 45-DTE
candidates comparable.

Filters applied (spec §5):
  - DTE band [dte_min, dte_max]
  - |delta| in [csp_delta_min, csp_delta_max]
  - open_interest >= open_interest_min
  - daily volume >= volume_min
  - bid-ask spread <= bid_ask_spread_max_pct of mid
  - IVR >= ivr_min when computable (None passes — see data/ivr.py)

Per-ticker overrides from universe.yaml apply via core.config.effective_wheel_params.

Out of scope here (lives in risk/limits.py — Sprint 4):
  - Earnings blackout
  - Buying-power floor
  - Per-position notional cap
  - Concurrent position cap
  - Regime gate
"""

from __future__ import annotations

from datetime import date
from typing import Any, Awaitable, Callable

from core.broker import Broker
from core.checkpoint import log_checkpoint
from core.config import effective_wheel_params
from core.models import OptionContract
from data.chain import ChainFilters, annualized_yield, fetch_filtered_chain
from data.ivr import IVRProvider, effective_ivr_min

# Optional callback that persists the chain we evaluated so the cycle backtester
# can replay decisions. Caller is responsible for the storage details; selector
# just hands off the (symbol, side, contracts) tuple.
ChainRecorder = Callable[[str, str, list[OptionContract]], Awaitable[None]]


async def select_csp(
    broker: Broker,
    symbol: str,
    config: dict[str, Any],
    universe: dict[str, Any],
    ivr: IVRProvider,
    *,
    today: date | None = None,
    record_chain: ChainRecorder | None = None,
) -> OptionContract | None:
    """Return the best CSP candidate, or None if nothing passes the filters."""
    today = today or date.today()
    params = effective_wheel_params(symbol, config, universe)

    rank = await ivr.iv_rank(symbol)
    ivr_min = effective_ivr_min(ivr, params)  # regime-aware (relaxes when VIX elevated)
    if rank is not None and rank < ivr_min:
        log_checkpoint(
            "csp_skip_ivr",
            status="ok",
            symbol=symbol,
            ivr=rank,
            ivr_min=ivr_min,
        )
        return None

    filters = ChainFilters(
        dte_min=int(params.get("dte_min", 30)),
        dte_max=int(params.get("dte_max", 45)),
        delta_min=float(params.get("csp_delta_min", 0.20)),
        delta_max=float(params.get("csp_delta_max", 0.30)),
        open_interest_min=int(params.get("open_interest_min", 0)),
        volume_min=int(params.get("volume_min", 0)),
        bid_ask_spread_max_pct=float(params.get("bid_ask_spread_max_pct", 100.0)),
    )

    candidates = await fetch_filtered_chain(broker, symbol, "put", filters, today=today)
    if record_chain is not None:
        await record_chain(symbol, "put", candidates)
    if not candidates:
        log_checkpoint("csp_no_candidates", status="ok", symbol=symbol)
        return None

    scored = [
        (annualized_yield(c, today), c)
        for c in candidates
        if annualized_yield(c, today) is not None
    ]
    if not scored:
        log_checkpoint("csp_no_scored", status="ok", symbol=symbol)
        return None

    scored.sort(key=lambda pair: pair[0], reverse=True)
    best_yield, best = scored[0]
    log_checkpoint(
        "csp_selected",
        status="ok",
        symbol=symbol,
        occ=best.occ_symbol,
        strike=best.strike,
        delta=best.delta,
        annualized_yield=best_yield,
    )
    return best
