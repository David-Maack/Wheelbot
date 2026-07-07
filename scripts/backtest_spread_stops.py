"""Put-spread stop-loss policy backtest — 2x stop vs width-as-stop.

Runs the paired exit-policy A/B in backtest/spread_policy.py over real daily
bars for the put_spread universe. READ-ONLY: no DB, no broker, no state. The
output gates any change to `stop_loss_mult` in config — do not touch the live
config without this evidence.

Policies compared on identical entries + identical modeled price paths:
    live_2x      50% profit / DTE-21 / 2x-credit stop   (current config)
    stop_3x      50% profit / DTE-21 / 3x-credit stop   (looser middle ground)
    no_stop      50% profit / DTE-21 / width is the stop
    hold_to_exp  50% profit only, ride to expiration    (shows DTE-21's value)

Run (local dev box needs no keys — data is yfinance):
    py -3.14 -m scripts.backtest_spread_stops
    py -3.14 -m scripts.backtest_spread_stops --symbols AAPL,TSLA --years 5
    docker exec wheelbot python -m scripts.backtest_spread_stops

Interpretation guide: `expectancy` is the per-trade edge; the paired-stop
section is the direct answer — of the trades the live stop closed, how many
would the no-stop policy have recovered (stop hurt) vs how many got worse
(stop helped), and the net dollars between them.
"""

from __future__ import annotations

import argparse

from backtest.data import load_daily_yf
from backtest.spread_policy import (
    PairedTrade,
    PolicySpec,
    aggregate,
    paired_stop_analysis,
    run_symbol,
)

# put_spread universe (universe.yaml put_spread tags as of 2026-07). Override
# with --symbols; the watchlist overlay can drift from this default.
DEFAULT_SYMBOLS = "AAPL,MSFT,META,GOOGL,TSLA,NVDA,SPY,QQQ"

POLICIES = [
    PolicySpec(name="live_2x", profit_close_pct=50, time_close_dte=21, stop_loss_mult=2.0),
    PolicySpec(name="stop_3x", profit_close_pct=50, time_close_dte=21, stop_loss_mult=3.0),
    PolicySpec(name="no_stop", profit_close_pct=50, time_close_dte=21, stop_loss_mult=None),
    PolicySpec(name="hold_to_exp", profit_close_pct=50, time_close_dte=None, stop_loss_mult=None),
]


def _fmt_row(stats: dict) -> str:
    reasons = stats["reasons"]
    rtxt = " ".join(f"{k}:{v}" for k, v in sorted(reasons.items()))
    return (
        f"{stats['policy']:<12} n={stats['n']:<5} "
        f"win={stats['win_rate'] * 100:5.1f}%  "
        f"avgW=${stats['avg_win']:7.2f}  avgL=${stats['avg_loss']:8.2f}  "
        f"exp=${stats['expectancy']:7.2f}  total=${stats['total_pnl']:10.2f}  "
        f"worst=${stats['worst']:8.2f}  [{rtxt}]"
    )


def _report(trades: list[PairedTrade], label: str, lines: list[str]) -> None:
    lines.append(f"\n== {label} ({len(trades)} paired trades) " + "=" * 30)
    for p in POLICIES:
        lines.append(_fmt_row(aggregate(trades, p.name)))
    for alt in ("no_stop", "stop_3x"):
        pa = paired_stop_analysis(trades, "live_2x", alt)
        lines.append(
            f"paired: of {pa['n_stopped']} live_2x stop-outs, {alt} did better on "
            f"{pa['stop_hurt']} (stop hurt) / worse on {pa['stop_helped']} (stop helped); "
            f"net {alt}-vs-stop ${pa['alt_minus_stop_total']:+.2f} "
            f"(${pa['alt_minus_stop_avg']:+.2f}/stopped trade)"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--dte", type=int, default=38, help="entry DTE (config band 30-45)")
    ap.add_argument("--delta", type=float, default=0.25, help="short-put target delta")
    ap.add_argument("--width", type=float, default=5.0)
    ap.add_argument("--entry-every", type=int, default=5, help="trading days between entries")
    ap.add_argument("--iv-premium", type=float, default=1.15,
                    help="IV proxy = realized vol x this (VRP); sensitivity-test it")
    ap.add_argument("--slippage", type=float, default=0.05,
                    help="per-share concession each way (matches router open/close_slippage)")
    ap.add_argument("--min-credit-pct", type=float, default=0.0,
                    help="skip entries with credit < pct of width (live gate is 25, but "
                    "no-skew BS understates put credit — default off here)")
    ap.add_argument("--out", default=None, help="also write the report to this file")
    args = ap.parse_args()

    all_trades: list[PairedTrade] = []
    lines: list[str] = [
        f"put_spread stop-policy backtest — dte={args.dte} delta={args.delta} "
        f"width=${args.width} entry_every={args.entry_every}d iv_premium={args.iv_premium} "
        f"slippage={args.slippage} years={args.years}",
        "NOTE: BS-modeled legs (no skew/term structure), daily closes only. Absolute "
        "P&L is modeled; the POLICY DELTAS are the deliverable.",
    ]
    for symbol in [s.strip().upper() for s in args.symbols.split(",") if s.strip()]:
        bars = load_daily_yf(symbol, period=f"{args.years}y")
        if bars.empty:
            lines.append(f"\n== {symbol}: no data — skipped")
            continue
        trades = run_symbol(
            symbol, bars, POLICIES,
            dte_days=args.dte, short_delta=args.delta, width=args.width,
            entry_every=args.entry_every, iv_premium=args.iv_premium,
            min_credit_pct_of_width=args.min_credit_pct,
            slippage_per_share=args.slippage,
        )
        _report(trades, symbol, lines)
        all_trades.extend(trades)

    if all_trades:
        _report(all_trades, "ALL SYMBOLS", lines)
    report = "\n".join(lines)
    print(report)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report + "\n")


if __name__ == "__main__":
    main()
