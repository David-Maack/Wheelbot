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

import pandas as pd

from backtest.data import load_daily_yf, load_intraday_alpaca, load_vix_daily
from backtest.engine import (
    EngineConfig,
    StructureSpec,
    generate_signals,
    price_shares,
    run_backtest,
    simulate_spy_trades,
)
from backtest.report import format_exit_breakdown, format_table, summarize
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


# Exit-mechanism grid (3-TF stack). The prior round showed the 5-min opposite
# -cross was flushing every trade in ~0.5 day regardless of stop width, so this
# round varies the EXIT MECHANISM: keep it / drop it / replace with a daily flip.
# Base = prior-day-level stop + 1-day min-hold + 7-day max-hold (the prior best).
_BASE = dict(stop_mode="prior_day_level", min_hold_days=1.0, max_hold_days=7)
_EXIT_CONFIGS = [
    ("pdl/opp", dict(**_BASE, opposite_cross_exit=True)),  # prior best, for reference
    ("pdl/noopp", dict(**_BASE, opposite_cross_exit=False)),  # stop/target/time only
    ("pdl/flip", dict(**_BASE, opposite_cross_exit=False, exit_on_daily_flip=True)),
    ("atr2/noopp", dict(stop_atr=2.0, min_hold_days=1.0, max_hold_days=7, opposite_cross_exit=False)),
]


def _tune(bars5m, daily, weekly, vix, json_out: bool) -> int:
    """Sweep exit configs x {no-regime, +200SMA} on the 3-TF stack."""
    params = _STACKS[3]
    rows, payload = [], {}
    for exit_label, kw in _EXIT_CONFIGS:
        for use_regime in (False, True):
            cfg = EngineConfig(use_regime=use_regime, **kw)
            label = f"{exit_label}{'+sma' if use_regime else ''}"
            results = run_backtest(bars5m, daily, vix, params, cfg, weekly=weekly)
            payload[label] = {}
            for struct, trades in results.items():
                s = summarize(trades)
                rows.append((label, struct, s))
                payload[label][struct] = asdict(s)
    if json_out:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(format_table(rows))
        print()
        print(format_exit_breakdown(rows))
        print("\nAll variants shown (no cherry-picking). If a config turns clearly "
              "positive, validate it on a held-out window before trusting it — else "
              "it's curve-fit to 2024-26.")
    return 0


# The winning config from tune round 2 (pdl/noopp+sma ITM). Locked for cost +
# out-of-sample validation — we do NOT re-tune here.
_WINNER = dict(
    stop_mode="prior_day_level", min_hold_days=1.0, max_hold_days=7,
    opposite_cross_exit=False, use_regime=True,
)
_WINNER_STRUCTS = (StructureSpec("ITM", 0.67),)
# Slippage levels to bracket reality ($/contract/side). Research: spread, not
# commission, dominates for ITM SPY, and PF~1.10 is where costs bite — so we
# find the breakeven slippage rather than trust one number.
_SLIPPAGE_GRID = [0.0, 2.0, 5.0, 10.0, 15.0, 20.0]
_COMMISSION = 0.65  # per contract per side (typical retail)
_OOS_SLIPPAGE = 5.0  # the cost level used for the per-period robustness check


def _winner_cfg(delta: float = 0.67, dte: float = 25.0, **over) -> EngineConfig:
    return EngineConfig(
        **_WINNER, dte=float(dte), structures=(StructureSpec("ITM", delta),), **over
    )


def _costs(bars5m, daily, weekly, vix, json_out: bool, delta: float, dte: float) -> int:
    """Run the chosen option config across a slippage grid → breakeven cost."""
    params = _STACKS[3]
    rows, payload = [], {}
    for slip in _SLIPPAGE_GRID:
        cfg = _winner_cfg(delta, dte, commission_per_contract=_COMMISSION,
                          slippage_per_contract=slip)
        trades = run_backtest(bars5m, daily, vix, params, cfg, weekly=weekly)["ITM"]
        s = summarize(trades)
        rt = 2 * (_COMMISSION + slip)
        rows.append((f"slip${slip:.0f}/rt${rt:.0f}", "ITM", s))
        payload[f"slip_{slip}"] = asdict(s)
    if json_out:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(f"{delta:.2f}-delta / {dte:.0f}-DTE option, slippage sweep:\n")
        print(format_table(rows))
        print("\nrt$ = round-trip cost/contract (commission+slippage, both sides). The "
              "edge survives only up to the slippage where exp$/PF cross <=0. A deeper-ITM "
              "(pricier) contract may have a wider absolute spread, so push the grid high.")
    return 0


def _oos(bars5m, daily, weekly, vix, json_out: bool, delta: float, dte: float) -> int:
    """Run the chosen option config (realistic cost) once, then bucket trades by
    year to check the edge isn't driven by a single regime."""
    params = _STACKS[3]
    cfg = _winner_cfg(delta, dte, commission_per_contract=_COMMISSION,
                      slippage_per_contract=_OOS_SLIPPAGE)
    trades = run_backtest(bars5m, daily, vix, params, cfg, weekly=weekly)["ITM"]
    rows = [(lbl, "ITM", s) for lbl, s in _by_year(trades, summarize)]
    if json_out:
        print(json.dumps({lbl: asdict(s) for lbl, _st, s in rows}, indent=2, default=str))
    else:
        print(f"{delta:.2f}-delta / {dte:.0f}-DTE option, cost ${_COMMISSION} commission + "
              f"${_OOS_SLIPPAGE}/side slippage (rt ${2*(_COMMISSION+_OOS_SLIPPAGE):.0f}):\n")
        print(format_table(rows))
        print("\nThe edge must hold in EACH period, not just FULL. One good year "
              "carrying a flat/negative one = fragile. Note: config was chosen on the "
              "whole window, so this is regime-robustness, not a pure holdout.")
    return 0


def _by_year(trades, summarize_fn):
    """(label, Summary) rows: FULL + each calendar year, by entry date."""
    buckets: dict[str, list] = {"FULL": list(trades)}
    for t in trades:
        buckets.setdefault(str(pd.Timestamp(t.entry_ts).year), []).append(t)
    out = []
    for label in ["FULL"] + sorted(k for k in buckets if k != "FULL"):
        out.append((label, summarize_fn(buckets[label])))
    return out


def _shares(bars5m, daily, weekly, vix, json_out: bool) -> int:
    """P&L the locked winner's SPY-level trades as the UNDERLYING (shares/MES):
    no spread, no theta. Per-year, so we test signal robustness directly."""
    cfg = _winner_cfg()  # option-cost fields irrelevant; we re-P&L as shares
    sig = generate_signals(bars5m, daily, _STACKS[3], weekly=weekly, cfg=cfg)
    spy = simulate_spy_trades(sig, daily, cfg)
    # ~$2/round-trip is generous for 100 SPY shares or 1 MES (commission + ~1 tick).
    trades = price_shares(spy, shares=100, cost_per_trade=2.0)
    rows = [(lbl, "SHARES", s) for lbl, s in _by_year(trades, summarize)]
    if json_out:
        print(json.dumps({lbl: asdict(s) for lbl, _st, s in rows}, indent=2, default=str))
    else:
        print("Locked signal, P&L as 100 SPY shares / 1 MES (cost $2/round-trip, "
              "NO spread, NO theta):\n")
        print(format_table(rows))
        print("\nCompare to the options OOS: if SHARES is positive in EACH of "
              "2024/2025/2026 (esp. 2025, which lost on options), the signal is real "
              "and options theta/spread were the killer. If 2025 still loses here, the "
              "signal itself lacks edge.")
    return 0


# Be choosier via Greeks: deeper ITM (higher delta) + longer DTE cut theta. The
# delta->1 / long-DTE corner approaches the shares trade.
_DELTA_DTE_GRID = [(0.67, 25), (0.80, 25), (0.90, 25), (0.80, 45), (0.90, 45), (0.90, 60)]


def _greeks(bars5m, daily, weekly, vix, json_out: bool) -> int:
    """Sweep option delta x DTE on the locked winner (realistic cost) to see if
    deeper-ITM / longer-DTE (less theta) survives costs better."""
    rows, payload = [], {}
    for delta, dte in _DELTA_DTE_GRID:
        cfg = EngineConfig(
            **_WINNER, dte=float(dte), structures=(StructureSpec(f"d{delta:.2f}/{dte}", delta),),
            commission_per_contract=_COMMISSION, slippage_per_contract=_OOS_SLIPPAGE,
        )
        trades = run_backtest(bars5m, daily, vix, _STACKS[3], cfg, weekly=weekly)[f"d{delta:.2f}/{dte}"]
        s = summarize(trades)
        rows.append((f"d{delta:.2f}/{dte}d", "OPT", s))
        payload[f"d{delta:.2f}/{dte}"] = asdict(s)
    if json_out:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(f"Option delta x DTE sweep, cost ${_COMMISSION}+${_OOS_SLIPPAGE}/side "
              f"(rt ${2*(_COMMISSION+_OOS_SLIPPAGE):.0f}):\n")
        print(format_table(rows))
        print("\nDeeper ITM (higher delta) + longer DTE = less theta drag. If a corner "
              "turns clearly positive, theta was the killer; the limit (delta->1) is the "
              "shares trade (--mode shares). Watch n: deeper/longer also costs more capital.")
    return 0


def run(start, end, sweep: list[int], yf_period: str, feed: str, json_out: bool,
        mode: str, delta: float = 0.67, dte: float = 25.0) -> int:
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

    if mode == "tune":
        return _tune(bars5m, daily, weekly, vix, json_out)
    if mode == "costs":
        return _costs(bars5m, daily, weekly, vix, json_out, delta, dte)
    if mode == "oos":
        return _oos(bars5m, daily, weekly, vix, json_out, delta, dte)
    if mode == "shares":
        return _shares(bars5m, daily, weekly, vix, json_out)
    if mode == "greeks":
        return _greeks(bars5m, daily, weekly, vix, json_out)

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
    p.add_argument("--mode", choices=["sweep", "tune", "costs", "oos", "shares", "greeks"],
                   default="sweep",
                   help="'sweep'=TF-count; 'tune'=exit x 200-SMA; 'costs'=slippage sweep; "
                        "'oos'=per-year robustness; 'shares'=P&L signal as shares/MES (no "
                        "theta/spread); 'greeks'=option delta x DTE sweep")
    p.add_argument("--yf-period", type=str, default="3y")
    p.add_argument("--feed", type=str, default="iex")
    p.add_argument("--delta", type=float, default=0.67,
                   help="target option delta for costs/oos modes (e.g. 0.90 for deep-ITM)")
    p.add_argument("--dte", type=float, default=25.0,
                   help="option DTE for costs/oos modes (e.g. 60)")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    end = datetime.fromisoformat(args.end) if args.end else datetime.now(UTC).replace(tzinfo=None)
    start = (datetime.fromisoformat(args.start) if args.start
             else end - timedelta(days=args.lookback_days))
    sweep = [int(x) for x in args.sweep.split(",") if x.strip()]
    return run(start, end, sweep, args.yf_period, args.feed, args.json, args.mode,
               args.delta, args.dte)


if __name__ == "__main__":
    sys.exit(main())
