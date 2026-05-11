"""Diagnostic for BROKER_DOWN / MANUAL_INTERVENTION recovery.

Run with:
    docker exec wheelbot python -m scripts.inspect_recovery

Read-only. Prints two reports:

  1. Every position in BROKER_DOWN or MANUAL_INTERVENTION, with the state
     it was in immediately before being flagged (recovered from state_log).

  2. Every recent order for symbols that appear in those positions, so we
     can see which orders are still PENDING / FILLED / etc. and which
     cycle_id they belong to.

The output is the input for the recovery script. Paste the output to the
operator (Claude) and they'll generate targeted UPDATE statements.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from core.config import load_config


def main() -> int:
    config = load_config()
    db_path = Path(config.get("database", {}).get("path", "wheelbot.db"))
    if not db_path.exists():
        print(f"DB not found: {db_path}")
        return 1

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    print("=" * 78)
    print("STUCK POSITIONS (BROKER_DOWN or MANUAL_INTERVENTION)")
    print("=" * 78)
    rows = list(
        conn.execute(
            """
            SELECT
                p.id, p.symbol, p.strategy_id, p.state AS current_state,
                p.current_cycle_id, p.shares, p.cost_basis,
                (SELECT from_state FROM state_log
                 WHERE position_id = p.id
                   AND to_state IN ('BROKER_DOWN', 'MANUAL_INTERVENTION')
                 ORDER BY id DESC LIMIT 1) AS prior_state,
                (SELECT reason FROM state_log
                 WHERE position_id = p.id
                   AND to_state IN ('BROKER_DOWN', 'MANUAL_INTERVENTION')
                 ORDER BY id DESC LIMIT 1) AS flag_reason
            FROM positions p
            WHERE p.state IN ('BROKER_DOWN', 'MANUAL_INTERVENTION')
            ORDER BY p.strategy_id, p.symbol
            """
        )
    )
    if not rows:
        print("(none)")
    else:
        for r in rows:
            print(
                f"id={r['id']:>4} | {r['symbol']:>6} | "
                f"strategy={r['strategy_id'] or 'NULL':<15} | "
                f"current={r['current_state']:<22} | "
                f"prior={(r['prior_state'] or 'NULL'):<18} | "
                f"cycle={r['current_cycle_id']!s:<5} | "
                f"shares={r['shares']:<5} | "
                f"basis={r['cost_basis']}"
            )
            print(f"     reason: {r['flag_reason']}")

    print()
    print("=" * 78)
    print("RECENT ORDERS for the affected symbols")
    print("=" * 78)
    symbols = sorted({r["symbol"] for r in rows})
    if not symbols:
        print("(no symbols to look up)")
    else:
        placeholders = ",".join("?" * len(symbols))
        orders = list(
            conn.execute(
                f"""
                SELECT id, symbol, strategy_id, order_type, status, cycle_id,
                       broker_order_id, client_order_id, fill_price, quantity,
                       placed_at, filled_at
                FROM orders
                WHERE symbol IN ({placeholders})
                ORDER BY placed_at DESC
                LIMIT 60
                """,
                symbols,
            )
        )
        for o in orders:
            print(
                f"id={o['id']:>4} | {o['symbol']:>6} | "
                f"strat={(o['strategy_id'] or 'NULL'):<15} | "
                f"{o['order_type']:<18} | "
                f"status={o['status']:<10} | "
                f"cycle={o['cycle_id']!s:<5} | "
                f"qty={o['quantity']:<3} | "
                f"fill={o['fill_price']!s:<8} | "
                f"placed={o['placed_at']}"
            )

    print()
    print("=" * 78)
    print("OPEN WHEEL CYCLES")
    print("=" * 78)
    cycles = list(
        conn.execute(
            """
            SELECT id, symbol, strategy_id, started_at,
                   initial_csp_strike, initial_csp_premium,
                   initial_capital_at_risk
            FROM wheel_cycles
            WHERE ended_at IS NULL
            ORDER BY strategy_id, symbol
            """
        )
    )
    if not cycles:
        print("(none)")
    else:
        for c in cycles:
            print(
                f"id={c['id']:>4} | {c['symbol']:>6} | "
                f"strat={(c['strategy_id'] or 'NULL'):<15} | "
                f"started={c['started_at']} | "
                f"strike={c['initial_csp_strike']} | "
                f"premium={c['initial_csp_premium']} | "
                f"risk={c['initial_capital_at_risk']}"
            )

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
