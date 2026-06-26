"""SPY swing backtest — go/no-go gate before any live wiring (sub-sprint 1).

Pulls 5-min SPY bars from Alpaca + daily/weekly SPY and daily VIX from yfinance,
runs the multi-timeframe VWAP/EMA crossover signal, prices ITM (~0.67 delta) and
OTM (~0.30 delta) option structures along the same SPY path, and reports win rate /
expectancy / profit factor / drawdown per structure across a 2/3/4-timeframe
sweep.

    python -m scripts.backtest_swing --lookback-days 730
    python -m scripts.backtest_swing --start 2024-01-01 --end 2026-06-01 --json

Needs ALPACA_API_KEY/SECRET (config/secrets.env). Modeled option leg uses a
single ATM IV (VIX proxy) — no skew/term-structure; see backtest/option_model.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime, timedelta

from backtest.data import load_daily_yf, load_intraday_alpaca, load_vix_daily
from backtest.engine import EngineConfig, run_backtest
from backtest.report import format_table, summarize
from strategies.swing_signal import SwingParams, TimeframeSpec

# Timeframe stacks for the sweep. The trigger is always 5m; higher TFs gate.
_STACKS = {
    2: SwingParams(timeframes=(
        TimeframeSpec("1D", "direction", vwap_mode="rolling", vwap_window=20),
        TimeframeSpec("5m", "trigger", vwap_mode="session"),
    )),
    3: SwingParams(timeframes=(
        TimeframeSpec("1W", "direction", vwap_mode="rolling", vwap_window=20),
        TimeframeSpec("1D", "direction", vwap_mode="rolling", vwap_window=20),
        TimeframeSpec("5m", "trigger", vwap_mode="session"),
    )),
}


def run(start, end, sweep: list[int], yf_period: str, feed: str, json_out: bool) -> int:
    print(f"Loading SPY 5-min bars from Alpaca ({start.date()} -> {end.date()}, feed={feed})...",
          file=sys.stderr)
    bars5m = load_intraday_alpaca("SPY", start, end, feed=feed)
    if bars5m.empty:
        print("No intraday bars returned — check keys / date range / feed.", file=sys.stderr)
        return 1
    daily = load_daily_yf("SPY", period=yf_period)
    weekly = load_daily_yf("SPY", period=yf_period, weekly=True)
    vix = load_vix_daily(period=yf_period)
    print(f"  5m bars: {len(bars5m)}  daily: {len(daily)}  weekly: {len(weekly)}  vix: {len(vix)}",
          file=sys.stderr)

    cfg = EngineConfig()
    rows, payload = [], {}
    for k in sweep:
        params = _STACKS.get(k)
        if params is None:
            print(f"  (timeframe-count {k} not wired yet — skipping)", file=sys.stderr)
            continue
        results = run_backtest(bars5m, daily, vix, params, cfg, weekly=weekly)
        payload[k] = {}
        for struct, trades in results.items():
            s = summarize(trades)
            rows.append((f"{k}", struct, s))
            payload[k][struct] = asdict(s)

    if json_out:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(format_table(rows))
        print("\nGate: only advance to a live-wired strategy if expectancy is clearly "
              "positive on recent data. Modeled option leg (single ATM IV) — treat "
              "magnitudes as indicative, the ITM-vs-OTM and 2-vs-3 *ordering* as the signal.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lookback-days", type=int, default=730)
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--sweep", type=str, default="2,3", help="timeframe counts, e.g. '2,3'")
    p.add_argument("--yf-period", type=str, default="3y")
    p.add_argument("--feed", type=str, default="iex")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    end = datetime.fromisoformat(args.end) if args.end else datetime.now(UTC).replace(tzinfo=None)
    start = (datetime.fromisoformat(args.start) if args.start
             else end - timedelta(days=args.lookback_days))
    sweep = [int(x) for x in args.sweep.split(",") if x.strip()]
    return run(start, end, sweep, args.yf_period, args.feed, args.json)


if __name__ == "__main__":
    sys.exit(main())
