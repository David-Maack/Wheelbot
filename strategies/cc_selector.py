"""CC selector — pick the best covered call for one symbol given cost basis.

Same shape as csp_selector with one extra rule from spec §8 #5:

    Strike floor (CC only): strike >= cost basis. No exceptions in v1.

That rule is non-negotiable and is enforced *here*, not in risk/limits.py, because
it's strike-selection, not a portfolio gate. Even if a sub-cost-basis call has the
best yield, we won't sell it.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from core.broker import Broker
from core.checkpoint import log_checkpoint
from core.config import effective_wheel_params
from core.models import OptionContract
from data.chain import ChainFilters, annualized_yield, fetch_filtered_chain


async def select_cc(
    broker: Broker,
    symbol: str,
    cost_basis: float,
    config: dict[str, Any],
    universe: dict[str, Any],
    *,
    today: date | None = None,
) -> OptionContract | None:
    """Return the best CC candidate at or above cost basis, or None."""
    today = today or date.today()
    params = effective_wheel_params(symbol, config, universe)

    filters = ChainFilters(
        dte_min=int(params.get("dte_min", 30)),
        dte_max=int(params.get("dte_max", 45)),
        delta_min=float(params.get("cc_delta_min", 0.20)),
        delta_max=float(params.get("cc_delta_max", 0.30)),
        open_interest_min=int(params.get("open_interest_min", 0)),
        volume_min=int(params.get("volume_min", 0)),
        bid_ask_spread_max_pct=float(params.get("bid_ask_spread_max_pct", 100.0)),
    )

    candidates = await fetch_filtered_chain(broker, symbol, "call", filters, today=today)
    if not candidates:
        log_checkpoint("cc_no_candidates", status="ok", symbol=symbol)
        return None

    above_basis = [c for c in candidates if c.strike >= cost_basis]
    if not above_basis:
        log_checkpoint(
            "cc_skip_below_basis",
            status="ok",
            symbol=symbol,
            cost_basis=cost_basis,
            n_filtered=len(candidates),
        )
        return None

    scored = [
        (annualized_yield(c, today), c)
        for c in above_basis
        if annualized_yield(c, today) is not None
    ]
    if not scored:
        log_checkpoint("cc_no_scored", status="ok", symbol=symbol)
        return None

    scored.sort(key=lambda pair: pair[0], reverse=True)
    best_yield, best = scored[0]
    log_checkpoint(
        "cc_selected",
        status="ok",
        symbol=symbol,
        occ=best.occ_symbol,
        strike=best.strike,
        delta=best.delta,
        cost_basis=cost_basis,
        annualized_yield=best_yield,
    )
    return best
