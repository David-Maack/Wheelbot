"""Manually re-enable a strategy whose runtime state was set by the
drawdown circuit breaker (TICKET-008) or the win-rate floor (TICKET-009).

    docker exec wheelbot python -m scripts.reenable_strategy --list
    docker exec wheelbot python -m scripts.reenable_strategy --strategy put_spread
    docker exec wheelbot python -m scripts.reenable_strategy --strategy put_spread \\
        --reason "reviewed last 10 cycles; 3 losses on KMI during a sector rotation; tightened universe"

This script DELETES the entire `strategy_runtime_state` row for the named
strategy, clearing BOTH the drawdown gate (DISABLED / WARNING) AND the
pause gate (LOW_WIN_RATE) in one shot. If you only meant to clear one but
the row had both set, you'll see both in the prior-state printout BEFORE
the clear happens — Ctrl-C if surprised.

The static `enabled` flag in config.yaml is NOT touched — this only
clears runtime state.

`--reason` is optional but encouraged for win-rate pause clears (TICKET-009).
It gets emitted as a `strategy_manual_reenable` checkpoint so the
operator's audit trail captures the *why* alongside the *when*. Drawdown
auto-resets are mechanical and don't need this; win-rate reenables are
human judgment calls and benefit from a written-down rationale.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from core.checkpoint import log_checkpoint
from core.config import load_config
from db.repo import Database, Repos


def _format_row(row: dict) -> str:
    """Render a single strategy_runtime_state row for the --list output and
    the prior-state printout. Shows both gates explicitly so the operator
    can't miss one."""
    parts = []
    if row.get("disabled_until"):
        parts.append(
            f"drawdown_state={row.get('drawdown_state') or 'DISABLED'} "
            f"until={row['disabled_until']} reason={row.get('disabled_reason') or '—'}"
        )
    elif row.get("drawdown_state"):
        parts.append(
            f"drawdown_state={row['drawdown_state']} reason={row.get('disabled_reason') or '—'}"
        )
    if row.get("pause_state"):
        eligible = row.get("paused_reenable_eligible_at") or "—"
        parts.append(
            f"pause_state={row['pause_state']} paused_at={row.get('paused_at') or '—'} "
            f"eligible={eligible} reason={row.get('paused_reason') or '—'}"
        )
    return " | ".join(parts) if parts else "(no active state)"


async def main_async(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--list", action="store_true",
        help="Show currently disabled OR paused strategies",
    )
    group.add_argument(
        "--strategy",
        help="Strategy id to re-enable (e.g. put_spread). Clears BOTH drawdown and pause state.",
    )
    parser.add_argument(
        "--reason",
        help="Optional rationale for the reenable. Logged via checkpoint for the audit trail "
             "(encouraged for win-rate pauses; redundant for drawdown auto-resets).",
    )
    args = parser.parse_args(argv)

    config = load_config()
    db_path = Path(config.get("database", {}).get("path", "wheelbot.db")).expanduser()
    async with Database(db_path) as db:
        repos = Repos(db)

        if args.list:
            disabled = await repos.strategy_runtime.list_disabled()
            paused = await repos.strategy_runtime.list_paused()
            # Merge by strategy_id so a strategy in BOTH gates appears once
            # with both flags visible.
            by_id: dict[str, dict] = {}
            for r in disabled:
                by_id[r["strategy_id"]] = r
            for r in paused:
                by_id.setdefault(r["strategy_id"], r).update(
                    {k: r[k] for k in (
                        "pause_state", "paused_at", "paused_reason",
                        "paused_reenable_eligible_at",
                    )}
                )
            if not by_id:
                print("No strategies are currently auto-disabled or paused.")
                return 0
            print(f"{'strategy':<24} state")
            print("-" * 110)
            for sid in sorted(by_id):
                print(f"{sid:<24} {_format_row(by_id[sid])}")
            return 0

        # --strategy path
        prior_row = await repos.strategy_runtime.get(args.strategy)
        if prior_row is None or _format_row(prior_row) == "(no active state)":
            print(f"Strategy {args.strategy!r} has no active runtime state — no-op.")
            return 0
        # Prior-state printout BEFORE clearing. The operator may have
        # intended to clear only one gate; surfacing both prevents the
        # "I meant to clear X, didn't realize I also cleared Y" surprise.
        print(f"Clearing {args.strategy!r}:")
        print(f"  prior: {_format_row(prior_row)}")
        if args.reason:
            print(f"  reason: {args.reason}")
        else:
            print("  reason: (none — consider passing --reason for win-rate pauses)")
        await repos.strategy_runtime.enable(args.strategy)
        # Grep-able audit trail. State_log is per-position so doesn't fit;
        # checkpoint logger is the right surface for strategy-level events.
        log_checkpoint(
            "strategy_manual_reenable",
            status="ok",
            strategy=args.strategy,
            prior_drawdown_state=prior_row.get("drawdown_state"),
            prior_pause_state=prior_row.get("pause_state"),
            prior_disabled_until=prior_row.get("disabled_until"),
            prior_disabled_reason=prior_row.get("disabled_reason"),
            prior_paused_reason=prior_row.get("paused_reason"),
            reason=args.reason,
        )
        print(f"Re-enabled {args.strategy!r}. Both drawdown and pause state cleared.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async()))
