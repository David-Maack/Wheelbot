"""Bull-put-spread exit-policy A/B engine (Sprint: stop-loss backtest).

Question under test: does the live 2x-credit stop (`stop_loss_mult: 2.0`) help
or hurt put_spread expectancy vs letting the spread's width be the stop?
Live cycles show avg win ~$54 vs -$150 stop-outs — that ratio needs ~74% win
rate to break even and the strategy runs ~57%. Research (tastytrade 50%/DTE-21
canon) treats the width as the stop on defined-risk structures; hard-stop
support is thin. This engine produces the evidence either way.

Method: model entries on real historical daily bars (BS-priced legs, realized
vol x premium as the IV proxy — we have no historical chains), then walk the
SAME modeled price path under multiple exit policies. The comparison is a
paired A/B per entry: absolute P&L is modeled, but the POLICY DELTA is robust
to IV-model bias because every policy sees identical prices.

Trigger semantics mirror strategies/spreads.py propose_close_for_symbol:
    profit: debit_to_close <= (1 - profit_close_pct/100) x credit
    time:   short DTE <= time_close_dte
    stop:   debit_to_close >= stop_loss_mult x credit
with credit = the entry fill (slippage already conceded), any trigger closes,
checked once per trading day at the close (live checks every ~5-min tick —
daily is the resolution we have; it slightly DELAYS stops vs live, noted in
the report).

Limitations (flag in any writeup):
- No vol skew/smile: BS off a single ATM-proxy sigma understates real put
  credit. Entry economics are conservative; the A/B delta is the deliverable.
- European exercise, no dividends, no early assignment.
- Daily closes only — intraday stop touches that recovered by the close are
  invisible (favors ALL policies equally, but delays stops most).

Pure functions over already-loaded frames — no I/O — unit-testable on
synthetic bars. The networked CLI lives in scripts/backtest_spread_stops.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from backtest.option_model import select_strike
from core.models import OptionType
from data.greeks import bs_price

CONTRACT_MULTIPLIER = 100
_R = 0.045  # matches data.greeks.DEFAULT_RISK_FREE_RATE


# --- policy + result types ---------------------------------------------------


@dataclass(frozen=True, slots=True)
class PolicySpec:
    """One exit-rule set. `stop_loss_mult` None/0 = no stop; `time_close_dte`
    None = ride to expiration (no DTE close)."""

    name: str
    profit_close_pct: float = 50.0
    time_close_dte: int | None = 21
    stop_loss_mult: float | None = 2.0


@dataclass(frozen=True, slots=True)
class SpreadEntry:
    symbol: str
    entry_date: date
    expiration: date
    entry_spot: float
    entry_iv: float
    short_strike: float
    long_strike: float
    width: float
    credit: float  # per share, modeled mid at entry (pre-slippage)
    credit_fill: float  # per share, after entry slippage concession


@dataclass(frozen=True, slots=True)
class PolicyOutcome:
    policy: str
    exit_date: date
    exit_reason: str  # "profit" | "stop" | "time" | "expiry"
    exit_debit: float  # per share, what we paid to close (incl. slippage)
    pnl: float  # dollars per 1-lot spread
    days_held: int


@dataclass(frozen=True, slots=True)
class PairedTrade:
    entry: SpreadEntry
    outcomes: dict[str, PolicyOutcome]


# --- pricing helpers ----------------------------------------------------------


def realized_vol(closes: pd.Series, window: int = 21) -> pd.Series:
    """Annualized close-to-close realized vol, rolling `window` days."""
    rets = np.log(closes / closes.shift(1))
    return rets.rolling(window).std() * math.sqrt(252)


def spread_debit(
    spot: float, iv: float, short_k: float, long_k: float, dte_days: float,
    *, r: float = _R,
) -> float:
    """Per-share cost to close the bull put spread now (positive = we pay).
    At dte<=0 bs_price returns intrinsic, so this also settles expiration."""
    t = max(dte_days, 0.0) / 365.0
    return (
        bs_price(spot, short_k, t, iv, OptionType.PUT, r)
        - bs_price(spot, long_k, t, iv, OptionType.PUT, r)
    )


def model_entry(
    symbol: str,
    entry_date: date,
    spot: float,
    iv: float,
    *,
    dte_days: int,
    short_delta: float,
    width: float,
    strike_step: float,
    slippage_per_share: float,
) -> SpreadEntry | None:
    """Pick strikes and price the entry credit. None when the structure is
    degenerate (long strike <= 0 or non-positive credit)."""
    if spot <= 0 or iv <= 0:
        return None
    short_k = select_strike(spot, dte_days, short_delta, OptionType.PUT, iv, step=strike_step)
    long_k = short_k - width
    if long_k <= 0:
        return None
    credit = spread_debit(spot, iv, short_k, long_k, dte_days)
    credit_fill = credit - slippage_per_share
    if credit <= 0 or credit_fill <= 0:
        return None
    return SpreadEntry(
        symbol=symbol,
        entry_date=entry_date,
        expiration=entry_date + timedelta(days=dte_days),
        entry_spot=spot,
        entry_iv=iv,
        short_strike=short_k,
        long_strike=long_k,
        width=width,
        credit=credit,
        credit_fill=credit_fill,
    )


# --- the walk ------------------------------------------------------------------


def walk_policy(
    entry: SpreadEntry,
    bars: pd.DataFrame,
    iv_series: pd.Series,
    policy: PolicySpec,
    *,
    slippage_per_share: float,
) -> PolicyOutcome | None:
    """Walk trading days after entry until a trigger fires or expiration.

    `bars` is the symbol's full daily frame (DatetimeIndex, `close` column);
    `iv_series` is aligned to the same index. Returns None when the data ends
    before the trade resolves (incomplete tail trade — caller drops it so
    every policy sees the same trade set).
    """
    credit = entry.credit_fill
    target_max_debit = (1.0 - policy.profit_close_pct / 100.0) * credit
    stop_mult = policy.stop_loss_mult or 0.0
    stop_threshold = stop_mult * credit

    path = bars.loc[bars.index.normalize() > pd.Timestamp(entry.entry_date)]
    last_iv = entry.entry_iv
    for ts, row in path.iterrows():
        day = ts.date()
        dte = (entry.expiration - day).days
        spot = float(row["close"])
        iv_raw = iv_series.get(ts)
        if iv_raw is not None and not math.isnan(iv_raw) and iv_raw > 0:
            last_iv = float(iv_raw)

        if dte <= 0:
            # Expiration: settle at intrinsic, no closing spread to cross.
            intrinsic = spread_debit(spot, last_iv, entry.short_strike, entry.long_strike, 0.0)
            pnl = (credit - intrinsic) * CONTRACT_MULTIPLIER
            return PolicyOutcome(
                policy=policy.name, exit_date=day, exit_reason="expiry",
                exit_debit=intrinsic, pnl=pnl,
                days_held=(day - entry.entry_date).days,
            )

        debit = spread_debit(spot, last_iv, entry.short_strike, entry.long_strike, dte)
        profit_trigger = debit <= target_max_debit
        time_trigger = policy.time_close_dte is not None and dte <= policy.time_close_dte
        stop_trigger = stop_mult > 0 and debit >= stop_threshold
        if profit_trigger or time_trigger or stop_trigger:
            # Reporting precedence mirrors the economics: a profit close is a
            # profit close even if DTE also tripped the same day.
            reason = "profit" if profit_trigger else ("stop" if stop_trigger else "time")
            exit_debit = debit + slippage_per_share
            pnl = (credit - exit_debit) * CONTRACT_MULTIPLIER
            return PolicyOutcome(
                policy=policy.name, exit_date=day, exit_reason=reason,
                exit_debit=exit_debit, pnl=pnl,
                days_held=(day - entry.entry_date).days,
            )
    return None  # ran out of data before resolution


def run_symbol(
    symbol: str,
    bars: pd.DataFrame,
    policies: list[PolicySpec],
    *,
    dte_days: int = 38,
    short_delta: float = 0.25,
    width: float = 5.0,
    strike_step: float = 1.0,
    entry_every: int = 5,
    iv_window: int = 21,
    iv_premium: float = 1.15,
    iv_floor: float = 0.10,
    min_credit_pct_of_width: float = 0.0,
    slippage_per_share: float = 0.05,
) -> list[PairedTrade]:
    """Fixed-cadence entries (every `entry_every` trading days) walked under
    every policy on the same path.

    The cadence is deliberately independent of exits so the entry set is
    identical across policies — that's what makes the comparison paired. Trades
    whose expiration runs past the data (or that any policy can't resolve) are
    dropped for all policies.
    """
    if bars.empty or "close" not in bars.columns:
        return []
    iv_series = (realized_vol(bars["close"], window=iv_window) * iv_premium).clip(lower=iv_floor)
    last_day = bars.index[-1].date()

    trades: list[PairedTrade] = []
    for i in range(iv_window, len(bars), entry_every):
        ts = bars.index[i]
        entry_day = ts.date()
        if entry_day + timedelta(days=dte_days) > last_day:
            break  # expiration beyond data — tail trades can't resolve
        iv = iv_series.iloc[i]
        if math.isnan(iv) or iv <= 0:
            continue
        entry = model_entry(
            symbol, entry_day, float(bars["close"].iloc[i]), float(iv),
            dte_days=dte_days, short_delta=short_delta, width=width,
            strike_step=strike_step, slippage_per_share=slippage_per_share,
        )
        if entry is None:
            continue
        credit_pct = (entry.credit / entry.width) * 100.0
        if min_credit_pct_of_width > 0 and credit_pct < min_credit_pct_of_width:
            continue
        outcomes: dict[str, PolicyOutcome] = {}
        for policy in policies:
            out = walk_policy(entry, bars, iv_series, policy, slippage_per_share=slippage_per_share)
            if out is None:
                outcomes = {}
                break
            outcomes[policy.name] = out
        if outcomes:
            trades.append(PairedTrade(entry=entry, outcomes=outcomes))
    return trades


# --- aggregation ----------------------------------------------------------------


def aggregate(trades: list[PairedTrade], policy_name: str) -> dict[str, float | int | dict]:
    """Per-policy stats over the paired trade set."""
    pnls = [t.outcomes[policy_name].pnl for t in trades]
    reasons: dict[str, int] = {}
    for t in trades:
        r = t.outcomes[policy_name].exit_reason
        reasons[r] = reasons.get(r, 0) + 1
    n = len(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    return {
        "policy": policy_name,
        "n": n,
        "win_rate": (len(wins) / n) if n else 0.0,
        "avg_win": (sum(wins) / len(wins)) if wins else 0.0,
        "avg_loss": (sum(losses) / len(losses)) if losses else 0.0,
        "expectancy": (sum(pnls) / n) if n else 0.0,
        "total_pnl": sum(pnls),
        "worst": min(pnls) if pnls else 0.0,
        "reasons": reasons,
    }


def paired_stop_analysis(
    trades: list[PairedTrade], stopped_policy: str, alt_policy: str,
) -> dict[str, float | int]:
    """For trades where `stopped_policy` exited on the stop: did `alt_policy`
    (same entry, same path) do better or worse? This is the direct answer to
    "is the stop converting recoverable drawdowns into realized losses?"."""
    stopped = [
        t for t in trades if t.outcomes[stopped_policy].exit_reason == "stop"
    ]
    helped = hurt = 0
    delta_total = 0.0
    for t in stopped:
        delta = t.outcomes[alt_policy].pnl - t.outcomes[stopped_policy].pnl
        delta_total += delta
        if delta > 0:
            hurt += 1  # the stop realized a loss the alt policy recovered from
        else:
            helped += 1  # the stop cut a loss that got worse
    return {
        "n_stopped": len(stopped),
        "stop_hurt": hurt,
        "stop_helped": helped,
        "alt_minus_stop_total": delta_total,
        "alt_minus_stop_avg": (delta_total / len(stopped)) if stopped else 0.0,
    }
