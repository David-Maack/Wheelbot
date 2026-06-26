"""Metrics + formatting for the swing backtest.

Summarizes a list of option `Trade`s into the numbers that decide the go/no-go:
win rate, expectancy (% on premium and $), profit factor, max drawdown on the
cumulative P&L curve, and average hold. Reported per structure (ITM/OTM) and per
timeframe-count so the ITM-vs-OTM and 2-vs-3-vs-4 questions are answered side by
side.
"""

from __future__ import annotations

from dataclasses import dataclass

from backtest.engine import Trade


@dataclass(frozen=True, slots=True)
class Summary:
    n_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    expectancy_pct: float  # mean return on premium
    expectancy_usd: float  # mean $ per trade
    profit_factor: float
    total_pnl: float
    max_drawdown: float
    avg_hold_days: float


def summarize(trades: list[Trade]) -> Summary:
    if not trades:
        return Summary(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    pnls = [t.pnl for t in trades]
    pcts = [t.pnl_pct for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_pcts = [t.pnl_pct for t in trades if t.pnl > 0]
    loss_pcts = [t.pnl_pct for t in trades if t.pnl <= 0]

    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

    # Max drawdown on the cumulative P&L curve.
    cum, peak, max_dd = 0.0, 0.0, 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)

    return Summary(
        n_trades=len(trades),
        win_rate=len(wins) / len(trades),
        avg_win_pct=(sum(win_pcts) / len(win_pcts)) if win_pcts else 0.0,
        avg_loss_pct=(sum(loss_pcts) / len(loss_pcts)) if loss_pcts else 0.0,
        expectancy_pct=sum(pcts) / len(pcts),
        expectancy_usd=sum(pnls) / len(pnls),
        profit_factor=profit_factor,
        total_pnl=sum(pnls),
        max_drawdown=max_dd,
        avg_hold_days=sum(t.hold_days for t in trades) / len(trades),
    )


def format_table(rows: list[tuple[str, str, Summary]]) -> str:
    """rows = list of (config_label, structure, Summary). Returns a printable table."""
    header = (
        f"{'config':>12} {'struct':>6} {'n':>5} {'win%':>6} {'exp%':>7} "
        f"{'exp$':>8} {'PF':>6} {'maxDD$':>9} {'hold_d':>7} {'totP&L$':>10}"
    )
    lines = [header, "-" * len(header)]
    for tf_label, struct, s in rows:
        pf = "inf" if s.profit_factor == float("inf") else f"{s.profit_factor:.2f}"
        lines.append(
            f"{tf_label:>12} {struct:>6} {s.n_trades:>5} {s.win_rate * 100:>5.1f} "
            f"{s.expectancy_pct * 100:>6.1f} {s.expectancy_usd:>8.0f} {pf:>6} "
            f"{s.max_drawdown:>9.0f} {s.avg_hold_days:>7.2f} {s.total_pnl:>10.0f}"
        )
    return "\n".join(lines)
