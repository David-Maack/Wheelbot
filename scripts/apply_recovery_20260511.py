"""One-shot recovery of stuck positions after the MLEG read-back bug.

Targets the 8 stuck positions from `scripts.inspect_recovery` output on
2026-05-11. Idempotent — safe to run multiple times; positions already in
the target state are skipped.

Run with:
    docker exec wheelbot python -m scripts.apply_recovery_20260511 --dry-run
    docker exec wheelbot python -m scripts.apply_recovery_20260511 --apply

Targets:

  | id | symbol | strategy       | current             | → target    | cycle |
  | --:| ------ | -------------- | ------------------- | ----------- | ----- |
  |  5 | BAC    | monthly_wheel  | BROKER_DOWN         | CSP_OPEN    | 4     |
  |  6 | NOK    | monthly_wheel  | BROKER_DOWN         | CSP_OPEN    | 5     |
  |  4 | RIVN   | monthly_wheel  | BROKER_DOWN         | CSP_OPEN    | 2     |
  |  3 | SOFI   | monthly_wheel  | BROKER_DOWN         | CSP_OPEN    | 3     |
  |  2 | HOOD   | monthly_wheel  | MANUAL_INTERVENTION | IDLE        | clear |
  |  1 | PLTR   | monthly_wheel  | MANUAL_INTERVENTION | IDLE        | clear |
  | 10 | GOOGL  | put_spread     | BROKER_DOWN         | IDLE        | clear |
  |  7 | AMD    | weekly_wheel   | BROKER_DOWN         | IDLE        | clear |

Each transition writes a state_log row tagged triggered_by=MANUAL.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

from core.config import load_config


# (position_id, expected_symbol, target_state, keep_cycle, target_cycle_id)
#   keep_cycle = True  → leave current_cycle_id alone
#   keep_cycle = False → clear current_cycle_id (set to NULL)
TARGETS: list[tuple[int, str, str, bool, int | None]] = [
    (5,  "BAC",   "CSP_OPEN", True,  None),
    (6,  "NOK",   "CSP_OPEN", True,  None),
    (4,  "RIVN",  "CSP_OPEN", True,  None),
    (3,  "SOFI",  "CSP_OPEN", True,  None),
    (2,  "HOOD",  "IDLE",     False, None),
    (1,  "PLTR",  "IDLE",     False, None),
    (10, "GOOGL", "IDLE",     False, None),
    (7,  "AMD",   "IDLE",     False, None),
]


def _utcnow_iso() -> str:
    return datetime.now(UTC).replace(tzinfo=None).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="Print plan, no writes")
    group.add_argument("--apply", action="store_true", help="Apply the recovery")
    args = parser.parse_args()

    config = load_config()
    db_path = Path(config.get("database", {}).get("path", "wheelbot.db"))
    if not db_path.exists():
        print(f"DB not found: {db_path}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    actions: list[tuple[int, str, str, str, bool]] = []
    for pos_id, expected_symbol, target_state, keep_cycle, _target_cycle in TARGETS:
        row = conn.execute(
            "SELECT id, symbol, strategy_id, state, current_cycle_id "
            "FROM positions WHERE id = ?",
            (pos_id,),
        ).fetchone()
        if row is None:
            print(f"SKIP id={pos_id}: position not found")
            continue
        if row["symbol"] != expected_symbol:
            print(
                f"SKIP id={pos_id}: expected symbol {expected_symbol} "
                f"got {row['symbol']} — bailing rather than guess"
            )
            continue
        if row["state"] == target_state:
            print(
                f"SKIP id={pos_id} {row['symbol']}: already in {target_state}"
            )
            continue
        actions.append(
            (pos_id, row["symbol"], row["state"], target_state, keep_cycle)
        )

    if not actions:
        print("Nothing to do.")
        return 0

    print()
    print(f"Plan ({len(actions)} change{'s' if len(actions) != 1 else ''}):")
    for pos_id, sym, current, target, keep in actions:
        cycle_note = "(keep cycle)" if keep else "(clear current_cycle_id)"
        print(f"  id={pos_id:>3} {sym:>6}: {current} → {target} {cycle_note}")

    if args.dry_run:
        print()
        print("Dry run — no changes applied. Re-run with --apply to execute.")
        return 0

    now = _utcnow_iso()
    reason = "manual recovery 2026-05-11 (MLEG read-back fix)"
    try:
        with conn:
            for pos_id, sym, current, target, keep in actions:
                if keep:
                    conn.execute(
                        "UPDATE positions SET state = ?, state_changed_at = ?, "
                        "state_change_reason = ? WHERE id = ?",
                        (target, now, reason, pos_id),
                    )
                else:
                    conn.execute(
                        "UPDATE positions SET state = ?, current_cycle_id = NULL, "
                        "state_changed_at = ?, state_change_reason = ? WHERE id = ?",
                        (target, now, reason, pos_id),
                    )
                conn.execute(
                    "INSERT INTO state_log (position_id, from_state, to_state, "
                    "reason, triggered_by, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (pos_id, current, target, reason, "MANUAL", now),
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
