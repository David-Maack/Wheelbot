"""Simulate what tighter close thresholds would have done to live spread cycles.

For each open put_spread / bear_call_spread cycle:
  1. Find the MULTI_LEG_OPEN fill order, extract original credit per share
  2. Quote both legs at the broker
  3. Compute current debit-to-close per share
  4. Show profit-capture % vs the original credit
  5. Show which thresholds (25/35/50%) would have triggered
  6. Show DTE on short leg vs 7/14/21 time-close

Read-only. Does not modify state. Safe to run during market hours.

Run:
    docker exec wheelbot python -m scripts.simulate_close_thresholds
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from core.broker import Broker
from core.broker_factory import make_broker
from core.config import load_config
from core.models import OrderType, PositionState
from db.repo import Database, Repos


THRESHOLDS = [25, 35, 50]
TIME_CLOSE_DTES = [21, 14, 7]


async def _quote_mid(broker: Broker, symbol: str) -> float | None:
    try:
        q = await broker.get_quote(symbol)
    except Exception:
        return None
    if q.mid is not None:
        return q.mid
    if q.bid is not None and q.ask is not None:
        return (q.bid + q.ask) / 2
    return q.last or q.bid or q.ask


async def _open_order_for_position(repos: Repos, position_id: int) -> Any:
    c = await repos.db.connect()
    async with c.execute(
        "SELECT * FROM orders WHERE cycle_id = ("
        " SELECT current_cycle_id FROM positions WHERE id = ?"
        ") AND order_type = ? AND status = 'FILLED'"
        " ORDER BY filled_at DESC LIMIT 1",
        (position_id, OrderType.MULTI_LEG_OPEN.value),
    ) as cur:
        row = await cur.fetchone()
    if row is None:
        return None
    from db.repo import _row_to_dict, JSON_FIELDS_BY_TABLE
    from core.models import Order

    return Order(**_row_to_dict(row, JSON_FIELDS_BY_TABLE["orders"]))


async def _analyze_position(broker: Broker, repos: Repos, position) -> dict | None:
    if position.id is None:
        return None
    open_order = await _open_order_for_position(repos, position.id)
    if open_order is None or not open_order.raw_request:
        return None
    legs = open_order.raw_request.get("legs") or []
    if len(legs) < 2:
        return None
    original_credit = open_order.fill_price or 0.0
    if original_credit <= 0:
        return None

    # Quote each leg and compute net debit-to-close per share.
    leg_quotes: list[tuple[str, str, float, float | None]] = []
    debit_to_close = 0.0
    for leg in legs:
        action = leg["action"]
        contract = leg["contract_symbol"]
        mid = await _quote_mid(broker, contract)
        leg_quotes.append((contract, action, leg.get("strike", 0.0), mid))
        if mid is None:
            return None
        # Closing means BUY back what we sold, SELL what we bought.
        if str(action) in ("SELL_TO_OPEN", "OrderType.SELL_TO_OPEN"):
            debit_to_close += mid  # buy back the short
        else:
            debit_to_close -= mid  # sell out of the long

    pct_captured = (
        (1.0 - (debit_to_close / original_credit)) * 100.0
        if original_credit > 0
        else 0.0
    )
    # Short leg DTE
    short_leg = next(
        (l for l in legs if str(l["action"]) in ("SELL_TO_OPEN", "OrderType.SELL_TO_OPEN")),
        legs[0],
    )
    short_expiry = date.fromisoformat(short_leg["expiration"]) if isinstance(short_leg["expiration"], str) else short_leg["expiration"]
    today = datetime.now(UTC).date()
    dte = (short_expiry - today).days

    quantity = open_order.quantity or 1
    triggers_profit = {pct: pct_captured >= pct for pct in THRESHOLDS}
    triggers_time = {tdte: dte <= tdte for tdte in TIME_CLOSE_DTES}

    return {
        "symbol": position.symbol,
        "strategy": position.strategy_id,
        "state": position.state,
        "qty": quantity,
        "original_credit_per_share": original_credit,
        "current_debit_per_share": debit_to_close,
        "pct_captured": pct_captured,
        "short_strike": short_leg.get("strike"),
        "short_dte": dte,
        "triggers_profit": triggers_profit,
        "triggers_time": triggers_time,
        # Realized P&L if we closed right now, per package + total
        "pnl_per_package": (original_credit - debit_to_close) * 100.0,
        "pnl_total": (original_credit - debit_to_close) * 100.0 * quantity,
    }


def _format_row(r: dict) -> str:
    profit_flags = "".join(
        "✓" if r["triggers_profit"][p] else "·" for p in THRESHOLDS
    )
    time_flags = "".join(
        "✓" if r["triggers_time"][t] else "·" for t in TIME_CLOSE_DTES
    )
    state_val = r["state"].value if hasattr(r["state"], "value") else str(r["state"])
    return (
        f"  {r['symbol']:<6} {r['strategy']:<18} {state_val:<14} "
        f"qty={r['qty']:>2}  "
        f"orig=${r['original_credit_per_share']:>5.2f}  "
        f"debit=${r['current_debit_per_share']:>5.2f}  "
        f"captured={r['pct_captured']:>6.1f}%  "
        f"DTE={r['short_dte']:>3}  "
        f"profit[25/35/50]={profit_flags}  "
        f"time[21/14/7]={time_flags}  "
        f"PnL=${r['pnl_total']:>8.2f}"
    )


async def main() -> int:
    config = load_config()
    db_path = Path(config.get("database", {}).get("path", "wheelbot.db")).expanduser()
    async with Database(db_path) as db:
        repos = Repos(db)
        broker = make_broker(config)
        account_id = config.get("account", {}).get("id", "primary")

        # All spread positions (open + pending) — we want the full picture.
        active = await repos.positions.list_active(account_id)
        spread_positions = [
            p for p in active
            if p.state in (PositionState.SPREAD_OPEN, PositionState.SPREAD_PENDING)
        ]
        if not spread_positions:
            print("No active spread positions found.")
            return 0

        rows: list[dict] = []
        for pos in spread_positions:
            r = await _analyze_position(broker, repos, pos)
            if r is not None:
                rows.append(r)

        if not rows:
            print("No analyzable spreads (missing quotes or open orders).")
            return 0

        # Sort by captured % descending so the highest-profit candidates first.
        rows.sort(key=lambda r: r["pct_captured"], reverse=True)

        print("\nLive spread positions — close-threshold simulation")
        print("=" * 130)
        for r in rows:
            print(_format_row(r))
        print("=" * 130)

        # Summary counts: how many would have closed at each threshold?
        n = len(rows)
        print(f"\nSimulation results across {n} spread{'s' if n != 1 else ''}:")
        for pct in THRESHOLDS:
            count = sum(1 for r in rows if r["triggers_profit"][pct])
            print(f"  At {pct}% profit-close → {count}/{n} would close NOW")
        for tdte in TIME_CLOSE_DTES:
            count = sum(1 for r in rows if r["triggers_time"][tdte])
            print(f"  At DTE-{tdte} time-close  → {count}/{n} would close NOW")

        # Total realized P&L if we closed every winner at each threshold right now.
        print("\nProspective total P&L if all winners closed NOW (sum of pnl_total > 0):")
        for pct in THRESHOLDS:
            total = sum(
                r["pnl_total"] for r in rows
                if r["triggers_profit"][pct] and r["pnl_total"] > 0
            )
            print(f"  {pct}% profit-close: ${total:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
