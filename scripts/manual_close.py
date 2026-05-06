"""Manual flatten CLI.

    python -m scripts.manual_close --symbol F             # close one ticker
    python -m scripts.manual_close --all                  # close everything
    python -m scripts.manual_close --symbol F --dry-run   # preview only
    python -m scripts.manual_close --symbol F --force     # bypass risk gates

For each affected position:

  CSP_OPEN / CSP_PENDING  → buy-to-close the open short put.
  CC_OPEN  / CC_PENDING   → buy-to-close the open short call.
  SHARES_HELD             → sell-to-close the shares.

All orders go through OrderRouter so client_order_id idempotency, retry, and
DB persistence stay consistent. `--force` short-circuits the risk gate (e.g.
when BP-floor would block a close-out under stress).

Every action writes a `state_log` row with `triggered_by=MANUAL`.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from core.broker import Broker
from core.broker_factory import make_broker
from core.checkpoint import configure_logging, log_checkpoint
from core.config import load_config, load_universe
from core.models import (
    OptionContract,
    OptionType,
    Order,
    OrderStatus,
    OrderType,
    Position,
    PositionState,
    StateLog,
    StateLogTrigger,
)
from db.repo import Database, Repos
from execution.router import OrderRouter
from risk.limits import RiskCheckFailed
from strategies.wheel import Proposal


CLOSE_STATES = {
    PositionState.CSP_OPEN,
    PositionState.CSP_PENDING,
    PositionState.CC_OPEN,
    PositionState.CC_PENDING,
    PositionState.SHARES_HELD,
}


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


async def _open_short_for(repos: Repos, position: Position) -> Order | None:
    """Most recent FILLED short option order for the position's current cycle."""
    if position.current_cycle_id is None:
        return None
    c = await repos.db.connect()
    async with c.execute(
        "SELECT * FROM orders WHERE cycle_id = ? AND order_type = ? "
        "AND status = ? ORDER BY placed_at DESC LIMIT 1",
        (
            position.current_cycle_id,
            OrderType.SELL_TO_OPEN.value,
            OrderStatus.FILLED.value,
        ),
    ) as cur:
        row = await cur.fetchone()
    if row is None:
        return None
    from db.repo import _row_to_dict, JSON_FIELDS_BY_TABLE

    return Order(**_row_to_dict(row, JSON_FIELDS_BY_TABLE["orders"]))


def _proposal_for_position(
    position: Position,
    open_short: Order | None,
) -> Proposal | None:
    """Construct the close-out Proposal that the router will place."""
    if position.state in (
        PositionState.CSP_OPEN,
        PositionState.CSP_PENDING,
        PositionState.CC_OPEN,
        PositionState.CC_PENDING,
    ):
        if open_short is None or open_short.contract_symbol is None:
            return None
        contract = OptionContract(
            underlying=position.symbol,
            occ_symbol=open_short.contract_symbol,
            strike=open_short.strike or 0.0,
            expiration=open_short.expiration or _utcnow().date(),
            option_type=open_short.option_type or OptionType.PUT,
            bid=None,
            ask=None,
        )
        return Proposal(
            symbol=position.symbol,
            contract=contract,
            order_type=OrderType.BUY_TO_CLOSE,
            quantity=open_short.quantity or 1,
            rationale="manual_close",
        )
    if position.state == PositionState.SHARES_HELD:
        contract = OptionContract(
            underlying=position.symbol,
            occ_symbol=position.symbol,  # stock — OCC slot reused for the underlying
            strike=0.0,
            expiration=_utcnow().date(),
            option_type=OptionType.CALL,  # placeholder; not used for stock orders
        )
        return Proposal(
            symbol=position.symbol,
            contract=contract,
            order_type=OrderType.SELL_TO_CLOSE,
            quantity=max(position.shares, 0),
            rationale="manual_close_shares",
        )
    return None


async def _select_targets(
    repos: Repos,
    account_id: str,
    symbol: str | None,
) -> list[Position]:
    if symbol:
        pos = await repos.positions.get_by_symbol(account_id, symbol)
        return [pos] if pos and pos.state in CLOSE_STATES else []
    active = await repos.positions.list_active(account_id)
    return [p for p in active if p.state in CLOSE_STATES]


async def _log_manual(repos: Repos, position: Position, reason: str) -> None:
    if position.id is None:
        return
    await repos.state_log.insert(
        StateLog(
            position_id=position.id,
            from_state=position.state,
            to_state=position.state,  # state will move when reconciler sees the fill
            reason=reason,
            triggered_by=StateLogTrigger.MANUAL,
            created_at=_utcnow(),
        )
    )


async def run(symbol: str | None, *, all_flag: bool, dry_run: bool, force: bool) -> int:
    config = load_config()
    universe = load_universe()
    db_path = Path(config.get("database", {}).get("path", "wheelbot.db")).expanduser()
    account_id = config.get("account", {}).get("id", "primary")

    if force:
        # Bypass the risk gates by toggling generous overrides for this run.
        config = {**config, "wheel": {**config.get("wheel", {}), "buying_power_floor_pct": 0.0, "max_position_pct_of_account": 100.0, "max_concurrent_positions": 1000}}
    if dry_run:
        config = {**config, "execution": {**config.get("execution", {}), "dry_run": True}}

    async with Database(db_path) as db:
        repos = Repos(db)
        broker = make_broker(config)
        target = symbol if not all_flag else None
        positions = await _select_targets(repos, account_id, target)
        if not positions:
            log_checkpoint(
                "manual_close_no_targets",
                status="ok",
                symbol=target,
                all=all_flag,
            )
            print("No closable positions found.")
            return 0
        router = OrderRouter(broker, repos, config, universe)
        for pos in positions:
            short = await _open_short_for(repos, pos)
            proposal = _proposal_for_position(pos, short)
            if proposal is None:
                print(f"[skip] {pos.symbol}: state={pos.state} not closable from data")
                continue
            print(
                f"[close] {pos.symbol} state={pos.state} qty={proposal.quantity} "
                f"type={proposal.order_type.value} dry_run={dry_run}"
            )
            try:
                result = await router.place(proposal)
            except RiskCheckFailed as exc:
                print(f"  risk check failed: {exc}")
                continue
            if result.dry_run:
                print("  dry-run: no order submitted, no DB write")
            elif result.placed is not None:
                print(f"  placed broker_order_id={result.placed.broker_order_id}")
            await _log_manual(repos, pos, f"manual_close --{'all' if all_flag else f'symbol={target}'}")
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--symbol", help="Close one ticker")
    group.add_argument("--all", action="store_true", help="Close everything")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually submit")
    parser.add_argument("--force", action="store_true", help="Bypass risk gates")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    configure_logging()
    return asyncio.run(
        run(args.symbol, all_flag=args.all, dry_run=args.dry_run, force=args.force)
    )


if __name__ == "__main__":
    sys.exit(main())
