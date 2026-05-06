"""Rule-based roll advisor (spec §9.3).

Triggered when a short option's |delta| reaches `roll_trigger_delta` (default
0.50, ITM-ish). Evaluates three actions:

    ROLL        — find a new strike same-or-further OTM at later DTE such that
                  the new credit > the cost to close the existing leg
                  (`roll_only_for_credit`). Returns the new contract.
    LET_ASSIGN  — for short puts: take the shares (cost basis = strike − premium).
                  Sensible when stock is one we'd own anyway and roll wouldn't
                  net credit.
    CLOSE       — buy to close at a loss; cycle ends with negative P&L. Used
                  when the position is deep-ITM enough that neither rolling
                  for credit nor accepting assignment is acceptable.

This module is **deterministic** — no LLM here. The LLM second opinion lives
in `intelligence/roll_advisor_llm.py`. The roll orchestrator combines them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Any

from core.broker import Broker
from core.checkpoint import log_checkpoint
from core.config import effective_wheel_params
from core.models import OptionContract, OptionType
from data.chain import ChainFilters, annualized_yield, fetch_filtered_chain


class RollAction(StrEnum):
    ROLL = "ROLL"
    LET_ASSIGN = "LET_ASSIGN"
    CLOSE = "CLOSE"


@dataclass(frozen=True, slots=True)
class RollContext:
    """What the advisor needs to decide. Caller (reconciler) assembles this."""

    symbol: str
    short_contract: OptionContract  # the open short leg
    short_quantity: int
    short_premium_collected_per_share: float  # what we sold the original for
    current_short_mid: float                  # cost to BTC right now (per share)
    underlying_price: float
    cycle_id: int | None = None


@dataclass(frozen=True, slots=True)
class RollDecision:
    action: RollAction
    rationale: str
    new_contract: OptionContract | None = None  # populated only on ROLL
    expected_credit_per_share: float | None = None  # net credit if we roll


def _is_triggered(ctx: RollContext, trigger: float) -> bool:
    delta = ctx.short_contract.delta
    if delta is None:
        return False
    return abs(delta) >= trigger


async def evaluate_roll(
    *,
    broker: Broker,
    ctx: RollContext,
    config: dict[str, Any],
    universe: dict[str, Any],
    today: date | None = None,
) -> RollDecision | None:
    """Return a RollDecision when the trigger is met, else None.

    None means "below trigger — leave the position alone." A non-None result
    is one of ROLL/LET_ASSIGN/CLOSE.
    """
    today = today or date.today()
    params = effective_wheel_params(ctx.symbol, config, universe)
    trigger = float(params.get("roll_trigger_delta", 0.50))
    only_credit = bool(params.get("roll_only_for_credit", True))

    if not _is_triggered(ctx, trigger):
        return None

    side = "put" if ctx.short_contract.option_type == OptionType.PUT else "call"
    candidate = await _find_credit_roll(
        broker=broker,
        ctx=ctx,
        params=params,
        only_credit=only_credit,
        side=side,
        today=today,
    )

    if candidate is not None:
        new_credit, new_contract = candidate
        log_checkpoint(
            "roll_advisor_decide",
            status="ok",
            symbol=ctx.symbol,
            action="ROLL",
            new_strike=new_contract.strike,
            net_credit=round(new_credit, 4),
        )
        return RollDecision(
            action=RollAction.ROLL,
            rationale=(
                f"credit roll {ctx.short_contract.strike}→{new_contract.strike} "
                f"net {new_credit:.2f}/share"
            ),
            new_contract=new_contract,
            expected_credit_per_share=new_credit,
        )

    # No credit roll available. Decide between LET_ASSIGN and CLOSE.
    if ctx.short_contract.option_type == OptionType.PUT:
        # Per spec §6: "tier 1 = stocks we'd own anyway" — taking assignment is
        # the standard outcome of a wheel CSP that goes ITM and can't be rolled.
        log_checkpoint(
            "roll_advisor_decide",
            status="ok",
            symbol=ctx.symbol,
            action="LET_ASSIGN",
        )
        return RollDecision(
            action=RollAction.LET_ASSIGN,
            rationale="no credit roll available; take assignment (CSP wheel rule)",
        )
    # Covered calls: if we can't roll for credit, closing is better than letting
    # shares go for less than current price. Spec §5: cc_strike_floor = cost_basis,
    # so the existing CC was sold at >= cost basis — a CLOSE here may book a
    # mark-to-market loss on the option but preserves the shares.
    log_checkpoint(
        "roll_advisor_decide",
        status="ok",
        symbol=ctx.symbol,
        action="CLOSE",
    )
    return RollDecision(
        action=RollAction.CLOSE,
        rationale="no credit roll; close CC to retain shares",
    )


async def _find_credit_roll(
    *,
    broker: Broker,
    ctx: RollContext,
    params: dict[str, Any],
    only_credit: bool,
    side: str,
    today: date,
) -> tuple[float, OptionContract] | None:
    """Search the chain for a strike further OTM at later DTE that pays >= the
    cost to BTC the existing leg. Returns (net_credit_per_share, contract) or
    None when no credit roll exists."""
    filters = ChainFilters(
        # Push DTE further out than current (typical wheel roll: 30-60 days
        # past the existing expiry's DTE).
        dte_min=int(params.get("dte_min", 30)),
        dte_max=int(params.get("dte_max", 45)),
        delta_min=float(params.get("csp_delta_min" if side == "put" else "cc_delta_min", 0.20)),
        delta_max=float(params.get("csp_delta_max" if side == "put" else "cc_delta_max", 0.30)),
        open_interest_min=int(params.get("open_interest_min", 0)),
        volume_min=int(params.get("volume_min", 0)),
        bid_ask_spread_max_pct=float(params.get("bid_ask_spread_max_pct", 100.0)),
    )

    candidates = await fetch_filtered_chain(
        broker, ctx.symbol, side, filters, today=today
    )
    if not candidates:
        return None

    # Only roll to expiries strictly after the current short.
    current_exp = ctx.short_contract.expiration
    candidates = [c for c in candidates if c.expiration > current_exp]
    if not candidates:
        return None

    # For puts: same or lower strike (further OTM = lower for puts) keeps the
    # roll defensive. For calls: same or higher strike.
    if side == "put":
        candidates = [c for c in candidates if c.strike <= ctx.short_contract.strike]
    else:
        candidates = [c for c in candidates if c.strike >= ctx.short_contract.strike]

    if not candidates:
        return None

    best: tuple[float, OptionContract] | None = None
    best_score: float | None = None
    for c in candidates:
        if c.bid is None or c.ask is None:
            continue
        new_premium = (c.bid + c.ask) / 2
        net_credit = new_premium - ctx.current_short_mid
        if only_credit and net_credit <= 0:
            continue
        # Net credit dominates; annualized yield breaks ties between same-credit legs.
        score = annualized_yield(c, today) or 0.0
        ranked_score = net_credit + score / 1000.0
        if best is None or best_score is None or ranked_score > best_score:
            best = (net_credit, c)
            best_score = ranked_score
    return best
