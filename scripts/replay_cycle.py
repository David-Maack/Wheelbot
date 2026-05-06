"""Replay a wheel cycle from the DB — narrative form.

    python -m scripts.replay_cycle --cycle-id 42
    python -m scripts.replay_cycle --symbol F --latest

Prints, in order:
  1. Cycle metadata (symbol, started/ended, outcome, days held).
  2. Order timeline (every fill: type, contract, qty, price, cycle-relative day).
  3. State transitions for the underlying position.
  4. P&L decomposition: premium credits, debits, share P&L, total.

This is debugging / audit, not a counterfactual rerun. The "would current
strategy have done the same?" backtester is the Sprint 8 ticket.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from core.config import load_config
from core.models import OrderStatus, OrderType
from db.repo import Database, Repos


def _row_to_dict(row: Any) -> dict[str, Any]:
    return dict(row)


async def _resolve_cycle_id(
    repos: Repos, cycle_id: int | None, symbol: str | None, latest: bool, account_id: str
) -> int | None:
    if cycle_id is not None:
        return cycle_id
    c = await repos.db.connect()
    if symbol and latest:
        async with c.execute(
            "SELECT id FROM wheel_cycles WHERE account_id = ? AND symbol = ? "
            "ORDER BY COALESCE(ended_at, started_at) DESC LIMIT 1",
            (account_id, symbol.upper()),
        ) as cur:
            row = await cur.fetchone()
        return row["id"] if row else None
    if latest:
        async with c.execute(
            "SELECT id FROM wheel_cycles WHERE account_id = ? "
            "ORDER BY COALESCE(ended_at, started_at) DESC LIMIT 1",
            (account_id,),
        ) as cur:
            row = await cur.fetchone()
        return row["id"] if row else None
    return None


async def _fetch_orders(repos: Repos, cycle_id: int) -> list[dict[str, Any]]:
    c = await repos.db.connect()
    async with c.execute(
        "SELECT * FROM orders WHERE cycle_id = ? ORDER BY placed_at",
        (cycle_id,),
    ) as cur:
        rows = await cur.fetchall()
    return [_row_to_dict(r) for r in rows]


async def _fetch_state_log(repos: Repos, symbol: str, account_id: str) -> list[dict[str, Any]]:
    c = await repos.db.connect()
    async with c.execute(
        "SELECT sl.* FROM state_log sl "
        "JOIN positions p ON sl.position_id = p.id "
        "WHERE p.account_id = ? AND p.symbol = ? "
        "ORDER BY sl.created_at",
        (account_id, symbol),
    ) as cur:
        rows = await cur.fetchall()
    return [_row_to_dict(r) for r in rows]


def _decompose_pnl(orders: list[dict[str, Any]]) -> dict[str, float]:
    """Sum filled orders with sign + multiplier conventions matching reconciler."""
    premium_credits = 0.0
    premium_debits = 0.0
    share_legs = 0.0
    for o in orders:
        if o["status"] != OrderStatus.FILLED.value or o.get("fill_price") is None:
            continue
        qty = o["quantity"] or 0
        price = o["fill_price"]
        ot = o["order_type"]
        if ot == OrderType.SELL_TO_OPEN.value:
            premium_credits += price * qty * 100
        elif ot == OrderType.BUY_TO_CLOSE.value:
            premium_debits += price * qty * 100
        elif ot == OrderType.BUY_TO_OPEN.value:
            share_legs -= price * qty
        elif ot == OrderType.SELL_TO_CLOSE.value:
            share_legs += price * qty
    return {
        "premium_credits": premium_credits,
        "premium_debits": premium_debits,
        "share_legs": share_legs,
        "total": premium_credits - premium_debits + share_legs,
    }


def _print_cycle(cycle: dict[str, Any]) -> None:
    print("=" * 72)
    print(f"Cycle #{cycle['id']} — {cycle['symbol']}")
    print(f"  started_at:   {cycle['started_at']}")
    print(f"  ended_at:     {cycle['ended_at'] or '— still open —'}")
    print(f"  outcome:      {cycle['cycle_outcome'] or '—'}")
    print(f"  days_held:    {cycle['days_held'] or '—'}")
    if cycle.get("initial_csp_strike") is not None:
        print(f"  initial CSP:  strike={cycle['initial_csp_strike']} premium={cycle['initial_csp_premium']}")
    print(f"  recorded P&L: {cycle['final_pnl'] if cycle.get('final_pnl') is not None else '—'}")


def _print_orders(orders: list[dict[str, Any]]) -> None:
    print("\nOrder timeline:")
    if not orders:
        print("  (none)")
        return
    for o in orders:
        print(
            f"  {o['placed_at']}  {o['order_type']:<14}  "
            f"{o.get('contract_symbol') or o['symbol']}  qty={o['quantity']}  "
            f"limit={o.get('limit_price')}  fill={o.get('fill_price')}  status={o['status']}"
        )


def _print_state_log(rows: list[dict[str, Any]]) -> None:
    print("\nState transitions:")
    if not rows:
        print("  (none recorded)")
        return
    for r in rows:
        print(
            f"  {r['created_at']}  {r['from_state'] or '∅'} → {r['to_state']}  "
            f"({r.get('triggered_by') or '—'}: {r.get('reason') or ''})"
        )


def _print_pnl(parts: dict[str, float]) -> None:
    print("\nP&L decomposition:")
    print(f"  premium credits  +{parts['premium_credits']:>12.2f}")
    print(f"  premium debits   -{parts['premium_debits']:>12.2f}")
    sign = "+" if parts["share_legs"] >= 0 else "-"
    print(f"  share legs       {sign}{abs(parts['share_legs']):>12.2f}")
    print(f"  ──────────────")
    print(f"  total            {parts['total']:+13.2f}")


async def run(cycle_id: int | None, symbol: str | None, latest: bool) -> int:
    config = load_config()
    db_path = Path(config.get("database", {}).get("path", "wheelbot.db")).expanduser()
    account_id = config.get("account", {}).get("id", "primary")
    async with Database(db_path) as db:
        repos = Repos(db)
        resolved = await _resolve_cycle_id(repos, cycle_id, symbol, latest, account_id)
        if resolved is None:
            print("No matching cycle found.", file=sys.stderr)
            return 1
        cycle = await repos.cycles.get(resolved)
        if cycle is None:
            print(f"Cycle {resolved} not found.", file=sys.stderr)
            return 1
        cycle_dict = cycle.model_dump(mode="json")
        cycle_dict["id"] = resolved
        orders = await _fetch_orders(repos, resolved)
        state_log = await _fetch_state_log(repos, cycle.symbol, account_id)

        _print_cycle(cycle_dict)
        _print_orders(orders)
        _print_state_log(state_log)
        _print_pnl(_decompose_pnl(orders))
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle-id", type=int)
    parser.add_argument("--symbol")
    parser.add_argument("--latest", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.cycle_id is None and not args.latest:
        print("Specify --cycle-id N or --latest [--symbol SYM].", file=sys.stderr)
        return 2
    return asyncio.run(run(args.cycle_id, args.symbol, args.latest))


if __name__ == "__main__":
    sys.exit(main())
