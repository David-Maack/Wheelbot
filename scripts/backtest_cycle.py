"""Per-cycle decision-point backtester (spec §13 #35).

    python -m scripts.backtest_cycle --cycle-id 42

For each decision in the closed cycle (CSP entry, any roll, CC entry, etc.),
re-run the *current* strategy code against the chain that was captured at
the time and report whether the decision would have been the same today.

Data sources, in priority order:
    1. chain_snapshots table (Sprint 8+ cycles).
    2. The chain stashed inside Order.raw_request (older paper-only cycles —
       mostly used by the alpaca paper adapter).

Output:
    - Per-decision row: original contract vs current selector's pick.
    - Summary line counting decisions, agreements, divergences.

Important caveat: this is decision-replay, NOT market replay. We can't
re-fetch live broker quotes to check whether the original order would have
filled at a different price; we just check whether the strategy code would
have *picked* the same OCC symbol given the same chain.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from core.config import load_config, load_universe
from core.models import (
    ChainSnapshot,
    OptionContract,
    OptionType,
    Order,
    OrderType,
)
from data.chain import annualized_yield
from db.repo import Database, Repos


@dataclass(slots=True)
class DecisionRow:
    placed_at: str
    side: str  # "put" | "call"
    original_occ: str
    original_strike: float | None
    original_score: float | None
    current_pick_occ: str | None
    current_pick_strike: float | None
    current_pick_score: float | None
    diverged: bool
    reason: str = ""


@dataclass(slots=True)
class CycleReport:
    cycle_id: int
    symbol: str
    started_at: str | None
    ended_at: str | None
    decisions: list[DecisionRow] = field(default_factory=list)

    @property
    def n_diverged(self) -> int:
        return sum(1 for d in self.decisions if d.diverged)


def _ot_for(order: Order) -> str | None:
    if order.option_type == OptionType.PUT:
        return "put"
    if order.option_type == OptionType.CALL:
        return "call"
    return None


async def _fetch_orders(repos: Repos, cycle_id: int) -> list[Order]:
    c = await repos.db.connect()
    async with c.execute(
        "SELECT * FROM orders WHERE cycle_id = ? AND order_type = ? "
        "ORDER BY placed_at",
        (cycle_id, OrderType.SELL_TO_OPEN.value),
    ) as cur:
        rows = await cur.fetchall()
    from db.repo import _row_to_dict, JSON_FIELDS_BY_TABLE

    return [Order(**_row_to_dict(r, JSON_FIELDS_BY_TABLE["orders"])) for r in rows]


def _chain_from_snapshot(snap: ChainSnapshot) -> list[OptionContract]:
    return [OptionContract(**c) for c in snap.contracts]


def _chain_from_raw_request(order: Order) -> list[OptionContract]:
    """Some legacy adapters stashed the candidate chain into raw_request.
    If so, parse it back. Returns [] when not present."""
    raw = order.raw_request or {}
    candidates = raw.get("candidates") if isinstance(raw, dict) else None
    if not isinstance(candidates, list):
        return []
    out: list[OptionContract] = []
    for item in candidates:
        try:
            out.append(OptionContract(**item))
        except Exception:
            continue
    return out


def _pick_top(chain: list[OptionContract], today: date) -> OptionContract | None:
    """Apply the same scoring rule as csp/cc selectors: highest annualized yield."""
    scored: list[tuple[float, OptionContract]] = []
    for c in chain:
        score = annualized_yield(c, today)
        if score is None:
            continue
        scored.append((score, c))
    if not scored:
        return None
    scored.sort(key=lambda p: p[0], reverse=True)
    return scored[0][1]


async def _resolve_chain(
    repos: Repos, order: Order, side: str
) -> tuple[list[OptionContract], str]:
    """Return (chain, source_label). Snapshot first, raw_request fallback."""
    placed_at = order.placed_at
    if order.cycle_id is not None:
        snaps = await repos.chain_snapshots.for_cycle(order.cycle_id)
        # Closest captured_at to placed_at + matching side.
        relevant = [s for s in snaps if s.side == side]
        if relevant:
            relevant.sort(key=lambda s: abs((s.captured_at - placed_at).total_seconds()))
            return _chain_from_snapshot(relevant[0]), "chain_snapshots"
    fallback = _chain_from_raw_request(order)
    return fallback, "raw_request" if fallback else "none"


async def backtest(repos: Repos, cycle_id: int) -> CycleReport | None:
    cycle = await repos.cycles.get(cycle_id)
    if cycle is None:
        return None
    report = CycleReport(
        cycle_id=cycle_id,
        symbol=cycle.symbol,
        started_at=str(cycle.started_at) if cycle.started_at else None,
        ended_at=str(cycle.ended_at) if cycle.ended_at else None,
    )
    orders = await _fetch_orders(repos, cycle_id)
    today = date.today()

    for order in orders:
        side = _ot_for(order)
        if side is None or order.contract_symbol is None:
            continue
        chain, source = await _resolve_chain(repos, order, side)
        if not chain:
            report.decisions.append(
                DecisionRow(
                    placed_at=str(order.placed_at),
                    side=side,
                    original_occ=order.contract_symbol,
                    original_strike=order.strike,
                    original_score=None,
                    current_pick_occ=None,
                    current_pick_strike=None,
                    current_pick_score=None,
                    diverged=False,
                    reason=f"no chain stored ({source}); skipping",
                )
            )
            continue

        original = next((c for c in chain if c.occ_symbol == order.contract_symbol), None)
        original_score = (
            annualized_yield(original, today) if original else None
        )
        pick = _pick_top(chain, today)
        pick_score = annualized_yield(pick, today) if pick else None

        diverged = pick is not None and pick.occ_symbol != order.contract_symbol
        report.decisions.append(
            DecisionRow(
                placed_at=str(order.placed_at),
                side=side,
                original_occ=order.contract_symbol,
                original_strike=order.strike,
                original_score=original_score,
                current_pick_occ=pick.occ_symbol if pick else None,
                current_pick_strike=pick.strike if pick else None,
                current_pick_score=pick_score,
                diverged=diverged,
                reason=f"chain source: {source}",
            )
        )
    return report


def _print(report: CycleReport) -> None:
    print(f"Cycle #{report.cycle_id} — {report.symbol}")
    print(f"  started_at: {report.started_at}")
    print(f"  ended_at:   {report.ended_at}")
    print(f"  decisions:  {len(report.decisions)}  diverged: {report.n_diverged}")
    for d in report.decisions:
        flag = "DIVERGED" if d.diverged else "agreed"
        score_orig = f"{d.original_score:.3f}" if d.original_score is not None else "—"
        score_pick = f"{d.current_pick_score:.3f}" if d.current_pick_score is not None else "—"
        print(
            f"  [{flag}] {d.placed_at}  side={d.side}  "
            f"orig={d.original_occ} ({d.original_strike}, yield={score_orig}) → "
            f"pick={d.current_pick_occ or 'NONE'} (yield={score_pick})  "
            f"{d.reason}"
        )


async def run(cycle_id: int, *, json_out: bool) -> int:
    config = load_config()
    db_path = Path(config.get("database", {}).get("path", "wheelbot.db")).expanduser()
    async with Database(db_path) as db:
        repos = Repos(db)
        report = await backtest(repos, cycle_id)
    if report is None:
        print(f"Cycle {cycle_id} not found.", file=sys.stderr)
        return 1
    if json_out:
        print(json.dumps(
            {
                "cycle_id": report.cycle_id,
                "symbol": report.symbol,
                "n_decisions": len(report.decisions),
                "n_diverged": report.n_diverged,
                "decisions": [d.__dict__ for d in report.decisions],
            },
            indent=2,
            default=str,
        ))
    else:
        _print(report)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle-id", type=int, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    return asyncio.run(run(args.cycle_id, json_out=args.json))


if __name__ == "__main__":
    sys.exit(main())
