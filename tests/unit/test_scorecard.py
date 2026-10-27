"""risk/scorecard — the D9 go-live / retire rule (2026-10-27)."""

from __future__ import annotations

from datetime import datetime

import pytest

from core.models import Order, OrderStatus, OrderType, WheelCycle
from core.strategies import StrategyDefinition
from risk.scorecard import (
    compute_scorecards,
    format_scorecard_line,
    max_drawdown,
    verdict,
)


def _strategy(id_: str, type_: str, **params) -> StrategyDefinition:
    return StrategyDefinition(
        id=id_, display_name=id_, type=type_, enabled=True, max_concurrent=2,
        params=params,
    )


def test_max_drawdown_is_worst_peak_to_trough():
    assert max_drawdown([]) == 0.0
    assert max_drawdown([10, 20, 5]) == 0.0
    # peak +30 after two wins, then -50 → trough -20: drawdown 50.
    assert max_drawdown([10, 20, -50, 5]) == pytest.approx(50.0)


@pytest.mark.parametrize(("n", "per", "fill", "expected"), [
    (12, 5.0, 0.9, "COLLECTING (12/30)"),
    (30, -0.5, 0.9, "RETIRE"),
    (30, 0.0, 0.9, "RETIRE"),
    (30, 2.0, 0.5, "FILL_LOW"),
    (30, 2.0, None, "FILL_LOW"),
    (31, 2.0, 0.7, "GO_LIVE_BAR"),
])
def test_verdict_rules(n, per, fill, expected):
    assert verdict(n, per, fill, min_trades=30, min_fill_rate=0.70) == expected


async def _cycle(repos, strategy_id: str, pnl: float, ended: str, symbol: str = "F") -> None:
    await repos.cycles.insert(WheelCycle(
        account_id="test", symbol=symbol, strategy_id=strategy_id,
        started_at=datetime.fromisoformat(ended), ended_at=datetime.fromisoformat(ended),
        final_pnl=pnl,
    ))


async def _open_order(repos, strategy_id: str, symbol: str, placed: str, status: OrderStatus,
                      n: int) -> None:
    await repos.orders.insert(Order(
        account_id="test", symbol=symbol, strategy_id=strategy_id,
        order_type=OrderType.SELL_TO_OPEN, quantity=1, status=status,
        placed_at=datetime.fromisoformat(placed),
        client_order_id=f"{strategy_id}-{symbol}-{placed}-{n}",
    ))


@pytest.mark.asyncio
async def test_wheel_scorecard_reads_realized_cycles_and_attempts(db_repos):
    await _cycle(db_repos, "monthly_wheel", 30.0, "2026-09-01T15:00:00")
    await _cycle(db_repos, "monthly_wheel", -50.0, "2026-09-10T15:00:00")
    await _cycle(db_repos, "monthly_wheel", 20.0, "2026-09-20T15:00:00")
    await _cycle(db_repos, "other", -999.0, "2026-09-20T15:00:00")
    # Two attempts: F on 09-01 re-priced 3x then filled; NOK on 09-02 never filled.
    for i in range(3):
        await _open_order(db_repos, "monthly_wheel", "F", "2026-09-01T14:00:00",
                          OrderStatus.CANCELLED, i)
    await _open_order(db_repos, "monthly_wheel", "F", "2026-09-01T15:00:00", OrderStatus.FILLED, 9)
    await _open_order(db_repos, "monthly_wheel", "NOK", "2026-09-02T14:00:00",
                      OrderStatus.CANCELLED, 0)

    [card] = await compute_scorecards(db_repos, [_strategy("monthly_wheel", "wheel")], {})
    assert card.n == 3 and card.penalized is False
    assert card.total == pytest.approx(0.0)
    assert card.per_trade == pytest.approx(0.0)
    assert card.max_drawdown == pytest.approx(50.0)
    assert (card.opens_placed, card.opens_filled) == (2, 1)   # per attempt, not per order
    assert card.fill_rate == pytest.approx(0.5)
    assert card.verdict == "COLLECTING (3/30)"
    assert "monthly_wheel 3 trades" in format_scorecard_line(card)


@pytest.mark.asyncio
async def test_scorecard_since_filters_history(db_repos):
    await _cycle(db_repos, "monthly_wheel", -500.0, "2026-06-01T15:00:00")
    await _cycle(db_repos, "monthly_wheel", 40.0, "2026-11-02T15:00:00")
    [card] = await compute_scorecards(
        db_repos, [_strategy("monthly_wheel", "wheel", scorecard_since="2026-10-28")], {},
    )
    assert card.n == 1 and card.total == pytest.approx(40.0)


@pytest.mark.asyncio
async def test_zero_dte_scorecard_reads_the_penalized_ledger(db_repos):
    """Closed ledger rows count at their PENALIZED P&L; proposal-only rows
    (no exit) don't count."""
    for i, (raw, pen) in enumerate([(12.0, 5.0), (-90.0, -110.0), (15.0, 8.0)]):
        row_id = await db_repos.zero_dte_ledger.insert_entry({
            "symbol": "SPY", "structure": "narrow_vertical", "quantity": 1,
            "entry_ts": f"2026-11-0{i + 2}T14:05:00", "entry_credit_raw": 0.12,
            "entry_spread_width_quotes": 0.02, "entry_credit_penalized": 0.11,
        })
        await db_repos.zero_dte_ledger.record_exit(
            row_id, exit_ts=f"2026-11-0{i + 2}T19:00:00", exit_debit_raw=0.0,
            exit_debit_penalized=0.0, pnl_raw=raw, pnl_penalized=pen, exit_reason="flatten",
        )
    await db_repos.zero_dte_ledger.insert_entry({
        "symbol": "QQQ", "structure": "narrow_vertical", "quantity": 1,
        "entry_ts": "2026-11-05T14:05:00", "entry_credit_raw": 0.10,
        "entry_spread_width_quotes": 0.02, "entry_credit_penalized": 0.09,
    })
    [card] = await compute_scorecards(
        db_repos, [_strategy("zero_dte", "zero_dte")],
        {"risk": {"scorecard": {"min_trades": 3, "min_fill_rate": 0.5}}},
    )
    assert card.penalized is True
    assert card.n == 3
    assert card.total == pytest.approx(-97.0)
    assert card.verdict == "RETIRE"
    assert "(pen.)" in format_scorecard_line(card)
