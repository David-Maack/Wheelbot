"""Shared multi-leg (vertical spread) types and helpers.

Since the 2026-10-27 two-strategy cleanup the only multi-leg strategy is
zero_dte (a same-day bull put spread); the 30-45 DTE put_spread /
narrow_put_spread / bear_call_spread orchestrators that used to live here
were removed. What stays is what zero_dte, the router, the risk gate and the
reconciler share: the MultiLegProposal package, the selected-spread
candidate, leg building, chain recording, and the close-path helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from core.broker import Broker
from core.models import ChainSnapshot, OptionContract, OrderLeg, OrderType, Quote
from db.repo import Repos

DIRECTION_BULL_PUT = "bull_put"


@dataclass(frozen=True, slots=True)
class MultiLegProposal:
    """A defined-risk multi-leg trade the router submits as a single package.

    Sign convention matches Broker.place_multi_leg_order: positive
    `net_credit_per_spread` = we receive premium. `max_loss_per_spread` is
    always positive ($ at risk per package, before commissions).
    """

    symbol: str
    legs: list[OrderLeg]
    net_credit_per_spread: float    # NET credit per single package, in dollars per share
    max_loss_per_spread: float      # defined max loss per single package, dollars total
    width_dollars: float            # |short - long| strike, in dollars (always positive)
    quantity: int
    rationale: str
    strategy_id: str = "zero_dte"
    # Treated as MULTI_LEG_OPEN at submission. Closes are a separate proposal.
    order_type: OrderType = OrderType.MULTI_LEG_OPEN
    direction: str = DIRECTION_BULL_PUT


@dataclass(frozen=True, slots=True)
class SpreadCandidate:
    short: OptionContract
    long: OptionContract
    net_credit_per_spread: float    # dollars per share, positive = credit
    max_loss_per_spread: float      # dollars total per package
    width_dollars: float            # short.strike - long.strike, positive
    short_yield: float              # annualized yield of the short leg (0 at DTE 0)


def _make_chain_recorder(
    repos: Repos,
    *,
    cycle_id: int | None = None,
    strategy_id: str | None = None,
) -> Any:
    chain_repo = getattr(repos, "chain_snapshots", None)
    if chain_repo is None:
        async def _noop(symbol: str, side: str, contracts: list[OptionContract]) -> None:
            return None
        return _noop

    async def _record(symbol: str, side: str, contracts: list[OptionContract]) -> None:
        if not contracts:
            return
        underlying_price = next(
            (c.underlying_price for c in contracts if c.underlying_price is not None),
            None,
        )
        await chain_repo.insert(
            ChainSnapshot(
                captured_at=datetime.now(UTC).replace(tzinfo=None),
                symbol=symbol,
                strategy_id=strategy_id,
                side=side,
                underlying_price=underlying_price,
                contracts=[c.model_dump(mode="json") for c in contracts],
                cycle_id=cycle_id,
            )
        )

    return _record


def _build_legs(candidate: SpreadCandidate) -> list[OrderLeg]:
    """Short leg sells to open, long leg buys to open."""
    return [
        OrderLeg(
            contract_symbol=candidate.short.occ_symbol,
            underlying=candidate.short.underlying,
            option_type=candidate.short.option_type,
            strike=candidate.short.strike,
            expiration=candidate.short.expiration,
            action=OrderType.SELL_TO_OPEN,
            ratio_qty=1,
        ),
        OrderLeg(
            contract_symbol=candidate.long.occ_symbol,
            underlying=candidate.long.underlying,
            option_type=candidate.long.option_type,
            strike=candidate.long.strike,
            expiration=candidate.long.expiration,
            action=OrderType.BUY_TO_OPEN,
            ratio_qty=1,
        ),
    ]


# -- close-path helpers ------------------------------------------------------


async def _open_order_for_position(repos: Repos, position_id: int) -> Any:
    """Most recent FILLED MULTI_LEG_OPEN order for this position's cycle.

    Used to recover the legs (and original credit) we need to construct the
    matching MULTI_LEG_CLOSE proposal.
    """
    c = await repos.db.connect()
    async with c.execute(
        "SELECT * FROM orders WHERE cycle_id = ("
        "SELECT current_cycle_id FROM positions WHERE id = ?"
        ") AND order_type = ? AND status = ? "
        "ORDER BY filled_at DESC LIMIT 1",
        (position_id, OrderType.MULTI_LEG_OPEN.value, "FILLED"),
    ) as cur:
        row = await cur.fetchone()
    if row is None:
        return None
    from core.models import Order
    from db.repo import JSON_FIELDS_BY_TABLE, _row_to_dict

    return Order(**_row_to_dict(row, JSON_FIELDS_BY_TABLE["orders"]))


def _flip_action(action: OrderType) -> OrderType:
    """OPEN → CLOSE for the same side."""
    if action == OrderType.SELL_TO_OPEN:
        return OrderType.BUY_TO_CLOSE
    if action == OrderType.BUY_TO_OPEN:
        return OrderType.SELL_TO_CLOSE
    return action


async def _leg_quote(broker: Broker, symbol: str) -> Quote | None:
    """Full quote for a close leg. The MID drives the close DECISION (fair
    value); the BID/ASK drive the ORDER PRICE so the close is marketable and
    actually fills rather than sitting behind a moving spread."""
    try:
        return await broker.get_quote(symbol)
    except Exception:
        return None
