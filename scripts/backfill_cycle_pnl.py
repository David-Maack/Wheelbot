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
from core.models import CycleOutcome
from db.repo import Database, Repos

_PROFIT_TO_LOSS = {
    "CSP_CLOSED_PROFIT": "CSP_CLOSED_LOSS",
    "CC_CLOSED_PROFIT": "CC_CLOSED_LOSS",
    "SPREAD_CLOSED_PROFIT": "SPREAD_CLOSED_LOSS",
}


async def main_async(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="Show the diff, write nothing")
    g.add_argument("--apply", action="store_true", help="Apply the corrections")
    args = parser.parse_args(argv)

    config = load_config()
    db_path = Path(config.get("database", {}).get("path", "wheelbot.db")).expanduser()
    async with Database(db_path) as db:
        repos = Repos(db)  # noqa: F841 — kept for parity / future use
        conn = await db.connect()
        conn.row_factory = __import__("sqlite3").Row

        # Find orphan FILLED BUY_TO_CLOSE orders (the buybacks that were never
        # counted). Match each to its cycle via the SELL_TO_OPEN that traded
        # the same contract. Accumulate a per-cycle correction = the buyback
        # debit that was wrongly omitted: -(fill_price * qty * 100).
        orphans = list(await (await conn.execute(
            "SELECT id, contract_symbol, quantity, fill_price FROM orders "
            "WHERE order_type = 'BUY_TO_CLOSE' AND status = 'FILLED' "
            "AND cycle_id IS NULL AND contract_symbol IS NOT NULL AND fill_price IS NOT NULL"
        )).fetchall())

        # cycle_id -> {"correction": float, "order_ids": [...]}
        corrections: dict[int, dict] = {}
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
            cyc_id = match["cycle_id"]
            contribution = -(o["fill_price"] or 0) * (o["quantity"] or 0) * 100
            entry = corrections.setdefault(cyc_id, {"correction": 0.0, "order_ids": []})
            entry["correction"] += contribution
            entry["order_ids"].append(o["id"])
            print(f"  [link] BUY_TO_CLOSE id={o['id']} {o['contract_symbol']} → cycle {cyc_id} "
                  f"(buyback debit {contribution:+.2f})")
            linked += 1

        print("\n=== affected cycle P&L corrections ===")
        old_total = 0.0
        new_total = 0.0
        for cyc_id, info in sorted(corrections.items()):
            cyc = await (await conn.execute(
                "SELECT id, symbol, cycle_outcome, final_pnl, initial_capital_at_risk "
                "FROM wheel_cycles WHERE id = ?",
                (cyc_id,),
            )).fetchone()
            if cyc is None:
                continue
            outcome = cyc["cycle_outcome"]
            old_pnl = cyc["final_pnl"] or 0.0
            # Never touch a manual close — its P&L was set deliberately.
            if outcome == CycleOutcome.MANUAL_CLOSE.value:
                print(f"  {cyc['symbol']:<6} {outcome:<22} {old_pnl:>9.2f} (manual close — left as-is)")
                continue
            new_pnl = round(old_pnl + info["correction"], 2)
            new_outcome = outcome
            if new_pnl < 0 and outcome in _PROFIT_TO_LOSS:
                new_outcome = _PROFIT_TO_LOSS[outcome]
            label = f"  → {new_outcome}" if new_outcome != outcome else ""
            print(f"  {cyc['symbol']:<6} {outcome:<22} {old_pnl:>9.2f} → {new_pnl:>9.2f}{label}")
            old_total += old_pnl
            new_total += new_pnl
            if args.apply:
                cap = cyc["initial_capital_at_risk"] or 0
                pct = (new_pnl / cap * 100.0) if cap else None
                for oid in info["order_ids"]:
                    await conn.execute("UPDATE orders SET cycle_id = ? WHERE id = ?", (cyc_id, oid))
                await conn.execute(
                    "UPDATE wheel_cycles SET final_pnl = ?, final_pnl_pct = ?, cycle_outcome = ? "
                    "WHERE id = ?",
                    (new_pnl, pct, new_outcome, cyc_id),
                )
        if args.apply:
            await conn.commit()

        print(f"\norphan buybacks matched: {linked}")
        print(f"affected-cycle P&L:  {old_total:.2f} → {new_total:.2f}  (delta {new_total - old_total:+.2f})")
        print("This delta is the correction to total realized P&L.")
        print("APPLIED." if args.apply else "DRY RUN — nothing written. Re-run with --apply.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async()))
