"""TICKET-016 — Calendar close orchestrator.

A calendar is a net-DEBIT structure, so CLOSING it nets a CREDIT (sell the
back, buy the front). Triggers:

  profit: close when the current close-value ≥ (1 + profit_close_pct/100) ×
          original debit  (i.e. the position has gained profit_close_pct of
          the debit).
  time:   front-leg DTE ≤ time_close_short_dte (default 2). This is the
          force-close that keeps the partial-expiry path (front expires while
          back lives) unreachable — a reconciler guard flags
          MANUAL_INTERVENTION if it ever doesn't fire.

v1 closes-and-reenters (does NOT roll the front). Builds a MULTI_LEG_CLOSE
with direction=calendar; the reconciler tags the cycle CALENDAR_CLOSED_PROFIT
(relabelled to _LOSS if realized P&L is negative).
"""

from __future__ import annotations

from datetime import date
from typing import Any

from core.broker import Broker
from core.checkpoint import log_checkpoint
from core.models import OrderLeg, OrderType, PositionState
from core.strategies import StrategyDefinition
from db.repo import Repos
from strategies.spreads import (
    DIRECTION_CALENDAR,
    MultiLegProposal,
    _close_pricing,
    _flip_action,
    _open_order_for_position,
)


async def propose_close_for_symbol(
    broker: Broker,
    repos: Repos,
    symbol: str,
    config: dict[str, Any],
    *,
    today: date | None = None,
    strategy: StrategyDefinition | None = None,
) -> MultiLegProposal | None:
    today = today or date.today()
    if strategy is None:
        return None
    account_id = config.get("account", {}).get("id", "primary")
    position = await repos.positions.get_by_symbol(account_id, symbol, strategy_id=strategy.id)
    if position is None or position.state != PositionState.SPREAD_OPEN or position.id is None:
        return None

    open_order = await _open_order_for_position(repos, position.id)
    if open_order is None or not open_order.raw_request:
        return None
    legs_raw = open_order.raw_request.get("legs") or []
    if not legs_raw:
        return None

    close_legs: list[OrderLeg] = []
    front_leg: OrderLeg | None = None
    for leg in legs_raw:
        ol = OrderLeg(
            contract_symbol=leg["contract_symbol"], underlying=leg["underlying"],
            option_type=leg["option_type"], strike=leg["strike"],
            expiration=leg["expiration"],
            action=_flip_action(OrderType(leg["action"])),
            ratio_qty=leg.get("ratio_qty", 1),
        )
        close_legs.append(ol)
        # The front (short) leg was SELL_TO_OPEN; its DTE drives the time close.
        if str(leg["action"]) == OrderType.SELL_TO_OPEN.value:
            front_leg = ol

    # Dual pricing via the shared spreads helper. Calendars quote WIDE (config
    # allows 20% bid/ask spreads), so a mid-priced close limit never crosses
    # and the router cancel-loops (CCL 2026-07-20/28). The MID value keeps
    # driving the close DECISION exactly as before; the MARKETABLE value —
    # sell the back at the bid, buy the front at the ask — drives the ORDER
    # limit so the close actually fills. Helper returns debits (positive = we
    # pay); a calendar close nets a credit, so negate.
    pricing = await _close_pricing(broker, close_legs)
    if pricing is None:
        log_checkpoint("calendar_close_skip_no_quote", status="skip",
                       symbol=symbol, strategy=strategy.id)
        return None
    mid_debit, marketable_debit = pricing
    # Close value (credit received on closing) per share = sell_back − buy_front.
    close_value = -mid_debit
    marketable_credit = -marketable_debit

    debit = abs(open_order.fill_price or 0.0)
    profit_close_pct = float(strategy.params.get("profit_close_pct", 25))
    profit_trigger = debit > 0 and close_value >= (1 + profit_close_pct / 100.0) * debit

    time_close_dte = int(strategy.params.get("time_close_short_dte", 2))
    front_dte = (front_leg.expiration - today).days if front_leg is not None else None
    time_trigger = front_dte is not None and front_dte <= time_close_dte

    if not (profit_trigger or time_trigger):
        return None
    reason = "calendar_profit" if profit_trigger else "calendar_time"
    rationale = (
        f"calendar_close[{symbol}] {reason} debit={debit:.2f} "
        f"value={close_value:.2f} marketable={marketable_credit:.2f} "
        f"front_dte={front_dte}"
    )
    log_checkpoint(
        "calendar_close_triggered", status="ok", symbol=symbol,
        trigger=reason, debit=debit, close_value=close_value, front_dte=front_dte,
    )
    return MultiLegProposal(
        symbol=symbol,
        legs=close_legs,
        # Closing a debit nets a credit. The ORDER price is the MARKETABLE
        # credit (sell at bid / buy at ask) so the router's limit crosses the
        # wide calendar quotes; the mid-based close_value drove the trigger.
        net_credit_per_spread=marketable_credit,
        max_loss_per_spread=debit * 100,
        width_dollars=0.0,
        quantity=open_order.quantity or 1,
        rationale=rationale,
        strategy_id=strategy.id,
        order_type=OrderType.MULTI_LEG_CLOSE,
        direction=DIRECTION_CALENDAR,
    )


async def propose_all_closes(
    broker: Broker,
    repos: Repos,
    config: dict[str, Any],
    *,
    today: date | None = None,
    strategy: StrategyDefinition | None = None,
) -> list[MultiLegProposal]:
    if strategy is None:
        return []
    account_id = config.get("account", {}).get("id", "primary")
    active = await repos.positions.list_active(account_id, strategy_id=strategy.id)
    out: list[MultiLegProposal] = []
    for pos in active:
        if pos.state != PositionState.SPREAD_OPEN:
            continue
        p = await propose_close_for_symbol(
            broker, repos, pos.symbol, config, today=today, strategy=strategy,
        )
        if p is not None:
            out.append(p)
    log_checkpoint("calendar_propose_all_closes", status="ok",
                   strategy=strategy.id, n_proposals=len(out))
    return out
