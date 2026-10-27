"""Resting profit-take orders (2026-10-27).

A strategy with `resting_profit_take: true` keeps a BUY_TO_CLOSE limit
resting at its profit target (e.g. 50% of the premium received) for as long
as the short is open. It fills only when the market comes to it, so the
profit-take pays no spread, where the old path crossed to the ask once the
mid reached the target. Everything else (stops, time / earnings / delta
closes, rolls) stays urgent and pays the ask.

Resting orders are recognised by their client_order_id prefix, so no schema
change is needed and every existing "wb-" consumer still sees them as ours:

  * router: places them under their own id family, never stale-replaces them,
    and cancels any live one before placing an urgent close on the same
    contract;
  * wheel_close: does not treat them as "in flight", so stops and time
    closes keep being evaluated while one is working;
  * the orphan sweep cancels one whose position is no longer open.
"""
from __future__ import annotations

import math

from core.models import Order, OrderStatus

RESTING_CLIENT_PREFIX = "wb-rest-"

_WORKING = (OrderStatus.PENDING, OrderStatus.PARTIAL)


def is_resting(order: Order) -> bool:
    return bool(order.client_order_id) and order.client_order_id.startswith(RESTING_CLIENT_PREFIX)


def is_working(order: Order) -> bool:
    return order.status in _WORKING


def profit_target(original_premium: float, profit_close_pct: float) -> float:
    """Mid at which the profit trigger fires: (1 - pct/100) x premium."""
    return (1 - float(profit_close_pct) / 100.0) * float(original_premium)


def resting_buy_limit(target: float) -> float:
    """A buy limit at or below `target` on a valid options tick (penny below
    $3, nickel at/above), floored at $0.01. Rounding DOWN keeps the take at
    least `profit_close_pct`; Alpaca rejects off-tick prices."""
    if target >= 3.0:
        snapped = math.floor(target * 20 + 1e-9) / 20
    else:
        snapped = math.floor(target * 100 + 1e-9) / 100
    return max(round(snapped, 2), 0.01)
