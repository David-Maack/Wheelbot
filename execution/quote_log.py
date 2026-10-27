"""Quote tape for the mid-fill test (docs/MID_FILL_TEST.md).

Every tick during market hours, record the live quote of each working
single-leg order (phase "working") and of each order filled within the last
`post_fill_minutes` (phase "post_fill"), for the strategies listed under
`execution.quote_log`. scripts/fill_report.py reads it back:

  * touch: did the quote ever reach our limit while the order worked
    (bid >= our sell limit, ask <= our buy limit)? A real-market fill at a
    resting price needs the market to come to it; a paper fill without any
    touch is a simulator artefact.
  * post-fill drift: how far the mid moved against us in the 30 minutes
    after a fill (adverse selection — patient orders tend to fill just
    before the price runs through them).

Observability only: failures are logged by the caller and never block
trading.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from core.broker import Broker
from core.models import OrderType
from db.repo import Repos
from execution.market_hours import is_market_hours

_SINGLE_LEG = (OrderType.SELL_TO_OPEN, OrderType.BUY_TO_CLOSE)


async def log_order_quotes(
    broker: Broker, repos: Repos, config: dict[str, Any], *, now: datetime | None = None,
) -> int:
    cfg = (config.get("execution") or {}).get("quote_log") or {}
    if not cfg.get("enabled", False):
        return 0
    if not is_market_hours(now):
        return 0
    strategies = set(cfg.get("strategies") or [])
    post_fill = timedelta(minutes=int(cfg.get("post_fill_minutes", 30)))
    account_id = (config.get("account") or {}).get("id", "primary")
    stamp = (now or datetime.now(UTC)).replace(tzinfo=None)

    working = await repos.orders.list_pending(account_id)
    filled = await repos.orders.list_filled_since(account_id, stamp - post_fill)
    rows = [(o, "working") for o in working] + [(o, "post_fill") for o in filled]
    written = 0
    for order, phase in rows:
        if order.id is None or order.order_type not in _SINGLE_LEG or not order.contract_symbol:
            continue
        if strategies and order.strategy_id not in strategies:
            continue
        try:
            q = await broker.get_quote(order.contract_symbol)
        except Exception:  # noqa: BLE001 — a missing quote is just a gap in the tape
            continue
        await repos.order_quote_log.insert(
            order_id=order.id, logged_at=stamp, phase=phase,
            bid=q.bid, ask=q.ask, limit_price=order.limit_price,
        )
        written += 1
    return written
