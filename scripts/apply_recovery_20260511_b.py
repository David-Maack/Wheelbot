"""Second recovery pass after the leg-order MANUAL_INTERVENTION fix.

Between the first recovery and the deploy of `nested=True` + leg-skip in
alpaca_broker, every reconcile tick re-flagged 5 positions as
MANUAL_INTERVENTION because Alpaca was returning multi-leg child orders
as flat top-level results. This script unflags any position currently
in MANUAL_INTERVENTION that has an associated FILLED order tagged with
a strategy_id — i.e. positions we placed and that have an open cycle.

The targets are determined dynamically (read from DB) rather than
hardcoded, because we don't know exactly which positions got re-flagged.

Rule applied per position:
  - If position.state == MANUAL_INTERVENTION
    AND position.current_cycle_id IS NOT NULL
    AND the cycle is still open (ended_at IS NULL)
    AND there exists a FILLED order for this position's symbol+strategy
        with order_type in (SELL_TO_OPEN, MULTI_LEG_OPEN)
    THEN restore to the appropriate *_OPEN state based on order type:
      - SELL_TO_OPEN + option_type=PUT → CSP_OPEN
      - SELL_TO_OPEN + option_type=CALL → CC_OPEN
      - MULTI_LEG_OPEN → SPREAD_OPEN

Anything else in MANUAL_INTERVENTION is left alone for human review.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

from core.config import load_config


def _utcnow_iso() -> str:
    return datetime.now(UTC).replace(tzinfo=None).isoformat()


def _resolve_target_state(conn: sqlite3.Connection, position: sqlite3.Row) -> tuple[str, str] | None:
    """Return (target_state, rationale) for a position to be recovered, or None
    if it should be left alone."""
    if position["current_cycle_id"] is None:
        return None
    cycle = conn.execute(
        "SELECT ended_at FROM wheel_cycles WHERE id = ?",
        (position["current_cycle_id"],),
    ).fetchone()
    if cycle is None or cycle["ended_at"] is not None:
        return None

    order = conn.execute(
        """
        SELECT order_type, option_type FROM orders
        WHERE symbol = ?
          AND strategy_id IS ?
          AND status = 'FILLED'
          AND order_type IN ('SELL_TO_OPEN', 'MULTI_LEG_OPEN')
        ORDER BY filled_at DESC
        LIMIT 1
        """,
        (position["symbol"], position["strategy_id"]),
    ).fetchone()
    if order is None:
        return None

    if order["order_type"] == "MULTI_LEG_OPEN":
        return ("SPREAD_OPEN", f"matching MLEG order found, cycle {position['current_cycle_id']} still open")
    if order["order_type"] == "SELL_TO_OPEN":
        if order["option_type"] == "PUT":
            return ("CSP_OPEN", f"matching CSP order found, cycle {position['current_cycle_id']} still open")
        if order["option_type"] == "CALL":
            return ("CC_OPEN", f"matching CC order found, cycle {position['current_cycle_id']} still open")
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    config = load_config()
    db_path = Path(config.get("database", {}).get("path", "wheelbot.db"))
    if not db_path.exists():
        print(f"DB not found: {db_path}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    flagged = list(
        conn.execute(
            "SELECT id, symbol, strategy_id, state, current_cycle_id "
            "FROM positions WHERE state = 'MANUAL_INTERVENTION'"
        )
    )
    if not flagged:
        print("No MANUAL_INTERVENTION positions. Nothing to do.")
        return 0

    plan: list[tuple[int, str, str, str, str]] = []
    skipped: list[tuple[str, str]] = []
    for pos in flagged:
        resolution = _resolve_target_state(conn, pos)
        if resolution is None:
            skipped.append(
                (f"id={pos['id']} {pos['symbol']} ({pos['strategy_id']})",
                 "no matching open cycle + FILLED order — leaving for human review")
            )
            continue
        target, rationale = resolution
        plan.append((pos["id"], pos["symbol"], pos["strategy_id"], target, rationale))

    print()
    print(f"Plan ({len(plan)} restore{'s' if len(plan) != 1 else ''}):")
    for pos_id, sym, strat, target, rationale in plan:
        print(f"  id={pos_id:>3} {sym:>6} ({strat}): MANUAL_INTERVENTION → {target}")
        print(f"      ↳ {rationale}")

    if skipped:
        print()
        print(f"Skipped ({len(skipped)}):")
        for label, reason in skipped:
            print(f"  {label}: {reason}")

    if not plan:
        return 0

    if args.dry_run:
        print()
        print("Dry run — no changes applied. Re-run with --apply to execute.")
        return 0

    now = _utcnow_iso()
    reason = "manual recovery 2026-05-11 pass 2 (leg-order filter deployed)"
    try:
        with conn:
            for pos_id, sym, strat, target, _ in plan:
                conn.execute(
                    "UPDATE positions SET state = ?, state_changed_at = ?, "
                    "state_change_reason = ? WHERE id = ?",
                    (target, now, reason, pos_id),
                )
                conn.execute(
                    "INSERT INTO state_log (position_id, from_state, to_state, "
                    "reason, triggered_by, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (pos_id, "MANUAL_INTERVENTION", target, reason, "MANUAL", now),
                )
        print()
        print("Applied. Reconciler will manage these on the next tick.")
        return 0
    except sqlite3.Error as exc:
        print(f"FAILED — transaction rolled back: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
