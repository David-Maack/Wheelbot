"""One-shot backfill for the missing-cycle_id P&L bug.

Before the fix, single-leg BUY_TO_CLOSE orders were never tagged with their
cycle_id, so _compute_cycle_pnl (which filters on cycle_id) ignored the
buyback debit and booked the FULL premium as profit. Every CSP profit-close,
time-close, and stop-loss exit was overstated by its buyback cost, and loss
exits were mislabeled as wins.

This script:
  1. Links orphan FILLED BUY_TO_CLOSE orders (cycle_id IS NULL) to the cycle
     whose SELL_TO_OPEN traded the same contract_symbol.
  2. Recomputes final_pnl for every closed cycle from the (now complete) set
     of linked orders.
  3. Relabels *_CLOSED_PROFIT outcomes to *_CLOSED_LOSS where realized P&L
     is negative.

    docker exec wheelbot python -m scripts.backfill_cycle_pnl --dry-run
    docker exec wheelbot python -m scripts.backfill_cycle_pnl --apply

Idempotent: safe to run repeatedly. --dry-run prints the diff without writing.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from core.config import load_config
from core.models import CycleOutcome, OrderType
from db.repo import Database, Repos

_PROFIT_TO_LOSS = {
    "CSP_CLOSED_PROFIT": "CSP_CLOSED_LOSS",
    "CC_CLOSED_PROFIT": "CC_CLOSED_LOSS",
    "SPREAD_CLOSED_PROFIT": "SPREAD_CLOSED_LOSS",
}


def _cycle_pnl_from_rows(rows: list) -> float:
    """Same math as reconciler._compute_cycle_pnl."""
    pnl = 0.0
    for row in rows:
        qty = row["quantity"] or 0
        price = row["fill_price"] or 0
        ot = row["order_type"]
        if ot in (OrderType.MULTI_LEG_OPEN.value, OrderType.MULTI_LEG_CLOSE.value):
            pnl += price * qty * 100
            continue
        sign = 1 if ot in (OrderType.SELL_TO_OPEN.value, OrderType.SELL_TO_CLOSE.value) else -1
        multiplier = 100 if ot in (OrderType.SELL_TO_OPEN.value, OrderType.BUY_TO_CLOSE.value) else 1
        pnl += sign * price * qty * multiplier
    return pnl


async def main_async(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="Show the diff, write nothing")
    g.add_argument("--apply", action="store_true", help="Apply the corrections")
    args = parser.parse_args(argv)

    config = load_config()
    db_path = Path(config.get("database", {}).get("path", "wheelbot.db")).expanduser()
    async with Database(db_path) as db:
        repos = Repos(db)
        conn = await db.connect()
        conn.row_factory = __import__("sqlite3").Row

        # 1. Link orphan BUY_TO_CLOSE orders to their cycle (match by contract).
        orphans = list(await (await conn.execute(
            "SELECT id, contract_symbol FROM orders "
            "WHERE order_type = 'BUY_TO_CLOSE' AND status = 'FILLED' "
            "AND cycle_id IS NULL AND contract_symbol IS NOT NULL"
        )).fetchall())
        linked = 0
        for o in orphans:
            match = await (await conn.execute(
                "SELECT cycle_id FROM orders WHERE contract_symbol = ? "
                "AND order_type = 'SELL_TO_OPEN' AND cycle_id IS NOT NULL LIMIT 1",
                (o["contract_symbol"],),
            )).fetchone()
            if match is None:
                print(f"  [skip] BUY_TO_CLOSE id={o['id']} {o['contract_symbol']} — no parent cycle")
                continue
            print(f"  [link] BUY_TO_CLOSE id={o['id']} {o['contract_symbol']} → cycle {match['cycle_id']}")
            if args.apply:
                await conn.execute(
                    "UPDATE orders SET cycle_id = ? WHERE id = ?",
                    (match["cycle_id"], o["id"]),
                )
            linked += 1
        if args.apply:
            await conn.commit()

        # 2. Recompute final_pnl for every closed cycle.
        cycles = list(await (await conn.execute(
            "SELECT id, symbol, cycle_outcome, final_pnl, initial_capital_at_risk "
            "FROM wheel_cycles WHERE ended_at IS NOT NULL"
        )).fetchall())
        print("\n=== cycle P&L recompute ===")
        old_total = 0.0
        new_total = 0.0
        for cyc in cycles:
            rows = list(await (await conn.execute(
                "SELECT order_type, quantity, fill_price FROM orders "
                "WHERE cycle_id = ? AND status = 'FILLED' AND fill_price IS NOT NULL",
                (cyc["id"],),
            )).fetchall())
            new_pnl = round(_cycle_pnl_from_rows(rows), 2)
            old_pnl = cyc["final_pnl"] or 0.0
            old_total += old_pnl
            new_total += new_pnl
            outcome = cyc["cycle_outcome"]
            new_outcome = outcome
            if new_pnl < 0 and outcome in _PROFIT_TO_LOSS:
                new_outcome = _PROFIT_TO_LOSS[outcome]
            flag = "" if abs(new_pnl - old_pnl) < 0.005 else "  <-- CHANGED"
            print(f"  {cyc['symbol']:<6} {outcome:<22} {old_pnl:>9.2f} → {new_pnl:>9.2f}  "
                  f"{new_outcome if new_outcome != outcome else ''}{flag}")
            if args.apply:
                cap = cyc["initial_capital_at_risk"] or 0
                pct = (new_pnl / cap * 100.0) if cap else None
                await conn.execute(
                    "UPDATE wheel_cycles SET final_pnl = ?, final_pnl_pct = ?, cycle_outcome = ? "
                    "WHERE id = ?",
                    (new_pnl, pct, new_outcome, cyc["id"]),
                )
        if args.apply:
            await conn.commit()

        print(f"\norphan BUY_TO_CLOSE orders linked: {linked}")
        print(f"realized P&L total:  {old_total:.2f} → {new_total:.2f}  "
              f"(delta {new_total - old_total:+.2f})")
        print("APPLIED." if args.apply else "DRY RUN — nothing written. Re-run with --apply.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async()))
