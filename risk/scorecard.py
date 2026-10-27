"""Per-strategy scorecard — the go-live / retire rule (audit decision D9).

One honest number set per strategy, computed from what actually filled:

  n            closed trades since `scorecard_since` (strategy param; default
               all history)
  per_trade    mean $/trade — zero_dte reads its PENALIZED ledger (half the leg
               spreads conceded each way + the flatten adder), every other
               strategy its realized cycle P&L
  max_drawdown worst peak-to-trough of cumulative P&L, in closing order
  fill_rate    share of entry ATTEMPTS that filled: an attempt is one (symbol,
               day) with at least one open order, so the 15-minute cancel /
               re-price churn doesn't count as many failures

Verdicts (thresholds: risk.scorecard.min_trades / min_fill_rate):
  COLLECTING   n < min_trades
  RETIRE       n >= min_trades and per_trade <= 0
  FILL_LOW     n >= min_trades, per_trade > 0, fill_rate < min_fill_rate
  GO_LIVE_BAR  n >= min_trades, per_trade > 0, fill_rate >= min_fill_rate

Read-only; shown on /risk and in the daily Discord summary. It never pauses
anything — the drawdown breaker is the automatic circuit; this is the
human's decision rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.models import OrderStatus, OrderType
from core.strategies import StrategyDefinition
from db.repo import Repos

DEFAULT_MIN_TRADES = 30
DEFAULT_MIN_FILL_RATE = 0.70

_OPEN_TYPES = (OrderType.SELL_TO_OPEN.value, OrderType.MULTI_LEG_OPEN.value)


@dataclass(frozen=True, slots=True)
class Scorecard:
    strategy_id: str
    since: str | None
    penalized: bool
    n: int
    total: float
    per_trade: float | None
    max_drawdown: float
    opens_placed: int
    opens_filled: int
    fill_rate: float | None
    verdict: str
    min_trades: int


def max_drawdown(pnls: list[float]) -> float:
    """Largest peak-to-trough drop of the running total (>= 0)."""
    peak = running = worst = 0.0
    for p in pnls:
        running += p
        peak = max(peak, running)
        worst = max(worst, peak - running)
    return worst


def verdict(
    n: int, per_trade: float | None, fill_rate: float | None,
    *, min_trades: int, min_fill_rate: float,
) -> str:
    if n < min_trades or per_trade is None:
        return f"COLLECTING ({n}/{min_trades})"
    if per_trade <= 0:
        return "RETIRE"
    if fill_rate is None or fill_rate < min_fill_rate:
        return "FILL_LOW"
    return "GO_LIVE_BAR"


async def _closed_pnls(
    repos: Repos, strategy: StrategyDefinition, since: str | None,
) -> list[float]:
    c = await repos.db.connect()
    if strategy.type == "zero_dte":
        sql = ("SELECT pnl_penalized AS p FROM zero_dte_ledger "
               "WHERE pnl_penalized IS NOT NULL")
        args: tuple[Any, ...] = ()
        if since:
            sql += " AND entry_ts >= ?"
            args = (since,)
        sql += " ORDER BY exit_ts, id"
    else:
        sql = ("SELECT final_pnl AS p FROM wheel_cycles WHERE strategy_id = ? "
               "AND ended_at IS NOT NULL AND final_pnl IS NOT NULL")
        args = (strategy.id,)
        if since:
            sql += " AND ended_at >= ?"
            args += (since,)
        sql += " ORDER BY ended_at, id"
    async with c.execute(sql, args) as cur:
        return [float(r["p"]) for r in await cur.fetchall()]


async def _entry_attempts(
    repos: Repos, strategy_id: str, since: str | None,
) -> tuple[int, int]:
    """(attempts, filled_attempts) over distinct (symbol, placed day)."""
    c = await repos.db.connect()
    sql = (
        "SELECT symbol, date(placed_at) AS d, "
        "MAX(CASE WHEN status = ? THEN 1 ELSE 0 END) AS filled "
        "FROM orders WHERE strategy_id = ? AND order_type IN (?, ?)"
    )
    args: tuple[Any, ...] = (OrderStatus.FILLED.value, strategy_id, *_OPEN_TYPES)
    if since:
        sql += " AND placed_at >= ?"
        args += (since,)
    sql += " GROUP BY symbol, date(placed_at)"
    async with c.execute(sql, args) as cur:
        rows = await cur.fetchall()
    return len(rows), sum(int(r["filled"]) for r in rows)


async def compute_scorecards(
    repos: Repos,
    strategies: list[StrategyDefinition],
    config: dict[str, Any],
) -> list[Scorecard]:
    cfg = (config.get("risk", {}) or {}).get("scorecard", {}) or {}
    min_trades = int(cfg.get("min_trades", DEFAULT_MIN_TRADES))
    min_fill = float(cfg.get("min_fill_rate", DEFAULT_MIN_FILL_RATE))
    out: list[Scorecard] = []
    for strategy in strategies:
        since_raw = strategy.params.get("scorecard_since")
        since = str(since_raw) if since_raw else None
        pnls = await _closed_pnls(repos, strategy, since)
        attempts, filled = await _entry_attempts(repos, strategy.id, since)
        n = len(pnls)
        per_trade = sum(pnls) / n if n else None
        fill_rate = filled / attempts if attempts else None
        out.append(Scorecard(
            strategy_id=strategy.id,
            since=since,
            penalized=strategy.type == "zero_dte",
            n=n,
            total=sum(pnls),
            per_trade=per_trade,
            max_drawdown=max_drawdown(pnls),
            opens_placed=attempts,
            opens_filled=filled,
            fill_rate=fill_rate,
            verdict=verdict(n, per_trade, fill_rate,
                            min_trades=min_trades, min_fill_rate=min_fill),
            min_trades=min_trades,
        ))
    return out


def format_scorecard_line(card: Scorecard) -> str:
    """One Discord line, e.g.
    `zero_dte 12 trades $+3.10/trade (pen.) dd $140 fill 80% — COLLECTING (12/30)`."""
    per = f"${card.per_trade:+.2f}/trade" if card.per_trade is not None else "no closed trades"
    fill = f"fill {card.fill_rate * 100:.0f}%" if card.fill_rate is not None else "fill —"
    tag = " (pen.)" if card.penalized else ""
    return (f"{card.strategy_id} {card.n} trades {per}{tag} "
            f"dd ${card.max_drawdown:.0f} {fill} — {card.verdict}")
