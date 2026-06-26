"""Swing backtest engine: signal generation + market-replay trade walk.

Flow (all no-lookahead):
  1. Per timeframe, compute indicators (EMA + anchored VWAP) and reduce to a
     direction series (higher TFs) or a cross series (the 5-min trigger).
  2. As-of merge: at each 5-min bar, each higher TF contributes the direction of
     its **last completed** bar (daily/weekly bar for the PRIOR day/week — never
     the in-progress one). `combine_signal` requires all of them to agree.
  3. Walk the 5-min bars: on a confirmed signal open a SPY-level trade with an
     ATR stop and an R-multiple target; manage to stop / target / opposite-cross
     / day-N time stop. One position at a time.
  4. Price the ITM and OTM option structures along the SAME SPY path, so the A/B
     differs only by structure.

The SPY-level trade is computed once; both option structures are priced on it.
Everything is a pure function over already-loaded frames — no I/O — so it runs
in unit tests on synthetic bars.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from backtest.data import atr
from backtest.option_model import OptionLeg, leg_pnl, open_leg, price_leg, pnl_pct
from core.models import OptionType
from strategies.swing_signal import (
    SwingParams,
    combine_signal,
    compute_indicators,
    entry_crosses,
    trend_direction,
)


@dataclass(frozen=True, slots=True)
class StructureSpec:
    name: str  # "ITM" | "OTM"
    target_delta: float


@dataclass(frozen=True, slots=True)
class EngineConfig:
    dte: float = 25.0
    stop_atr: float = 1.0  # stop distance = stop_atr * ATR(entry)  (stop_mode="atr")
    stop_mode: str = "atr"  # "atr" | "prior_day_level" (anchor stop to prior day's low/high)
    reward_risk: float = 1.5  # target distance = reward_risk * stop distance
    max_hold_days: int = 4  # time stop (calendar days)
    min_hold_days: float = 0.0  # suppress the STOP until this many days elapse (let it breathe)
    atr_period: int = 14
    use_regime: bool = False  # require signal to agree with the daily 200-SMA trend
    regime_sma: int = 200
    iv_floor: float = 0.08
    contracts: int = 1
    opposite_cross_exit: bool = True  # exit on a 5-min opposite EMA/VWAP cross (noisy)
    exit_on_daily_flip: bool = False  # exit when the daily direction flips against us
    structures: tuple[StructureSpec, ...] = (
        StructureSpec("ITM", 0.67),
        StructureSpec("OTM", 0.30),
    )


@dataclass(frozen=True, slots=True)
class SpyTrade:
    direction: int
    entry_ts: pd.Timestamp
    entry_spot: float
    exit_ts: pd.Timestamp
    exit_spot: float
    exit_reason: str
    hold_days: float


@dataclass(slots=True)
class Trade:
    structure: str
    direction: int
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry_spot: float
    exit_spot: float
    strike: float
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    hold_days: float
    exit_reason: str


# --- no-lookahead as-of direction ------------------------------------------
def _direction_by_date(daily_ind: pd.DataFrame) -> pd.Series:
    """Trend direction per daily bar, indexed by normalized date."""
    d = trend_direction(daily_ind)
    d.index = pd.to_datetime(d.index).normalize()
    return d


def asof_daily_direction(intraday_index: pd.DatetimeIndex, daily_dir: pd.Series) -> pd.Series:
    """Map each intraday bar to the PRIOR completed daily bar's direction.

    Shift the daily direction by one bar so date D sees D-1 (no lookahead), then
    broadcast onto the intraday timeline by calendar date. Unknown → 0 (neutral).
    """
    prior = daily_dir.shift(1).fillna(0)
    dates = pd.to_datetime(intraday_index).normalize()
    mapped = prior.reindex(dates.unique())
    lookup = mapped.to_dict()
    return pd.Series([lookup.get(d, 0) for d in dates], index=intraday_index)


def asof_weekly_direction(intraday_index: pd.DatetimeIndex, weekly_dir: pd.Series) -> pd.Series:
    """Like the daily version but keyed by ISO week, prior completed week."""
    wk = weekly_dir.copy()
    wk.index = pd.to_datetime(wk.index).to_period("W")
    prior = wk.shift(1).fillna(0)
    lookup = prior.to_dict()
    periods = pd.to_datetime(intraday_index).to_period("W")
    return pd.Series([lookup.get(p, 0) for p in periods], index=intraday_index)


# --- signal generation ------------------------------------------------------
def _regime_by_date(daily: pd.DataFrame, sma_period: int) -> pd.Series:
    """+1 / -1 / 0 per daily bar: sign(close - SMA). Warm-up (NaN SMA) → 0."""
    sma = daily["close"].rolling(sma_period, min_periods=sma_period).mean()
    reg = pd.Series(0, index=daily.index)
    reg[daily["close"] > sma] = 1
    reg[daily["close"] < sma] = -1
    reg.index = pd.to_datetime(reg.index).normalize()
    return reg


def generate_signals(
    bars5m: pd.DataFrame,
    daily: pd.DataFrame,
    params: SwingParams,
    *,
    weekly: pd.DataFrame | None = None,
    cfg: EngineConfig | None = None,
) -> pd.DataFrame:
    """Return the 5-min frame with `cross` and `signal` columns.

    `signal` is +1/-1/0 — a fresh 5-min EMA/VWAP cross confirmed by every higher
    timeframe in `params` agreeing on the same direction, and (when
    `cfg.use_regime`) by the daily 200-SMA trend agreeing too.
    """
    tf_by_role = {tf.role: tf for tf in params.timeframes}
    trigger_tf = tf_by_role["trigger"]
    out = compute_indicators(bars5m, trigger_tf, params)
    out["cross"] = entry_crosses(out)

    # Build the per-bar higher-TF direction columns (as-of, no lookahead).
    htf_cols: list[str] = []
    daily_tf = next((tf for tf in params.timeframes if tf.name == "1D"), None)
    if daily_tf is not None:
        daily_ind = compute_indicators(daily, daily_tf, params)
        out["dir_1D"] = asof_daily_direction(out.index, _direction_by_date(daily_ind)).values
        htf_cols.append("dir_1D")
    weekly_tf = next((tf for tf in params.timeframes if tf.name == "1W"), None)
    if weekly_tf is not None and weekly is not None:
        weekly_ind = compute_indicators(weekly, weekly_tf, params)
        wdir = trend_direction(weekly_ind)
        out["dir_1W"] = asof_weekly_direction(out.index, wdir).values
        htf_cols.append("dir_1W")

    # Optional 200-SMA regime gate (as-of prior completed day, no lookahead).
    regime = None
    if cfg is not None and cfg.use_regime:
        regime = asof_daily_direction(out.index, _regime_by_date(daily, cfg.regime_sma)).values

    signals = []
    for i, (_, row) in enumerate(out.iterrows()):
        htf = [int(row[c]) for c in htf_cols]
        sig = combine_signal(int(row["cross"]), htf)
        if regime is not None and sig != int(regime[i]):
            sig = 0  # counter-trend vs the 200-SMA → skip
        signals.append(sig)
    out["signal"] = signals
    return out


# --- SPY-level trade walk ---------------------------------------------------
def simulate_spy_trades(signal_df: pd.DataFrame, daily: pd.DataFrame, cfg: EngineConfig) -> list[SpyTrade]:
    """Walk the 5-min timeline, one position at a time, exiting on SPY-level
    stop / target / opposite-cross / time stop."""
    atr_by_date = atr(daily, cfg.atr_period)
    atr_by_date.index = pd.to_datetime(atr_by_date.index).normalize()
    atr_lookup = atr_by_date.shift(1).bfill().to_dict()  # prior-day ATR
    # Prior-day low/high for the "prior_day_level" stop anchor.
    low_prior = daily["low"].copy(); low_prior.index = pd.to_datetime(low_prior.index).normalize()
    high_prior = daily["high"].copy(); high_prior.index = pd.to_datetime(high_prior.index).normalize()
    low_lookup = low_prior.shift(1).to_dict()
    high_lookup = high_prior.shift(1).to_dict()

    rows = list(signal_df.itertuples())
    trades: list[SpyTrade] = []
    i, n = 0, len(rows)
    while i < n:
        r = rows[i]
        sig = int(getattr(r, "signal", 0))
        if sig == 0:
            i += 1
            continue
        entry_ts = r.Index
        entry_spot = float(r.close)
        entry_date = pd.Timestamp(entry_ts).normalize()
        a = atr_lookup.get(entry_date, None)
        if a is None or a <= 0:
            i += 1
            continue

        # Stop distance: ATR multiple, or anchored to the prior day's low/high.
        stop_dist = cfg.stop_atr * a
        if cfg.stop_mode == "prior_day_level":
            lvl = low_lookup.get(entry_date) if sig > 0 else high_lookup.get(entry_date)
            if lvl is not None:
                d = (entry_spot - lvl) if sig > 0 else (lvl - entry_spot)
                if d > 0:
                    stop_dist = d  # fall back to ATR distance if the level is on the wrong side
        if sig > 0:
            stop_px, target_px = entry_spot - stop_dist, entry_spot + cfg.reward_risk * stop_dist
        else:
            stop_px, target_px = entry_spot + stop_dist, entry_spot - cfg.reward_risk * stop_dist

        exit_ts, exit_spot, reason = None, None, None
        j = i + 1
        while j < n:
            b = rows[j]
            held_days = (pd.Timestamp(b.Index) - pd.Timestamp(entry_ts)).total_seconds() / 86400.0
            hi, lo, close = float(b.high), float(b.low), float(b.close)
            opp_cross = cfg.opposite_cross_exit and int(getattr(b, "cross", 0)) == -sig
            daily_flip = cfg.exit_on_daily_flip and int(getattr(b, "dir_1D", 0)) == -sig
            allow_stop = held_days >= cfg.min_hold_days  # let the trade breathe first
            if sig > 0:
                if allow_stop and lo <= stop_px:  # stop checked first (conservative)
                    exit_ts, exit_spot, reason = b.Index, stop_px, "stop"
                elif hi >= target_px:
                    exit_ts, exit_spot, reason = b.Index, target_px, "target"
            else:
                if allow_stop and hi >= stop_px:
                    exit_ts, exit_spot, reason = b.Index, stop_px, "stop"
                elif lo <= target_px:
                    exit_ts, exit_spot, reason = b.Index, target_px, "target"
            if exit_ts is None and opp_cross:
                exit_ts, exit_spot, reason = b.Index, close, "opposite_cross"
            if exit_ts is None and daily_flip:
                exit_ts, exit_spot, reason = b.Index, close, "daily_flip"
            if exit_ts is None and held_days >= cfg.max_hold_days:
                exit_ts, exit_spot, reason = b.Index, close, "time_stop"
            if exit_ts is not None:
                break
            j += 1
        if exit_ts is None:  # ran out of data
            last = rows[-1]
            exit_ts, exit_spot, reason = last.Index, float(last.close), "eod_data"
            j = n - 1
        trades.append(
            SpyTrade(
                direction=sig,
                entry_ts=entry_ts,
                entry_spot=entry_spot,
                exit_ts=exit_ts,
                exit_spot=float(exit_spot),
                exit_reason=reason,
                hold_days=(pd.Timestamp(exit_ts) - pd.Timestamp(entry_ts)).total_seconds() / 86400.0,
            )
        )
        i = j + 1  # one position at a time: resume after the exit bar
    return trades


# --- option pricing along the SPY path -------------------------------------
def _iv_asof(ts: pd.Timestamp, vix_daily: pd.Series, floor: float) -> float:
    prior = vix_daily[vix_daily.index < pd.Timestamp(ts).normalize()]
    iv = float(prior.iloc[-1]) if not prior.empty else (float(vix_daily.iloc[0]) if not vix_daily.empty else floor)
    return max(iv, floor)


def price_structure(
    spy_trades: list[SpyTrade], spec: StructureSpec, vix_daily: pd.Series, cfg: EngineConfig
) -> list[Trade]:
    """Price one option structure (ITM or OTM) along the shared SPY trades."""
    out: list[Trade] = []
    for st in spy_trades:
        ot = OptionType.CALL if st.direction > 0 else OptionType.PUT
        iv_in = _iv_asof(st.entry_ts, vix_daily, cfg.iv_floor)
        leg: OptionLeg = open_leg(st.entry_spot, cfg.dte, spec.target_delta, ot, iv_in, cfg.contracts)
        iv_out = _iv_asof(st.exit_ts, vix_daily, cfg.iv_floor)
        exit_price = price_leg(leg, st.exit_spot, st.hold_days, iv_out)
        out.append(
            Trade(
                structure=spec.name,
                direction=st.direction,
                entry_ts=st.entry_ts,
                exit_ts=st.exit_ts,
                entry_spot=st.entry_spot,
                exit_spot=st.exit_spot,
                strike=leg.strike,
                entry_price=leg.entry_price,
                exit_price=exit_price,
                pnl=leg_pnl(leg, exit_price),
                pnl_pct=pnl_pct(leg, exit_price),
                hold_days=st.hold_days,
                exit_reason=st.exit_reason,
            )
        )
    return out


def run_backtest(
    bars5m: pd.DataFrame,
    daily: pd.DataFrame,
    vix_daily: pd.Series,
    params: SwingParams,
    cfg: EngineConfig,
    *,
    weekly: pd.DataFrame | None = None,
) -> dict[str, list[Trade]]:
    """Full run: signals → SPY trades → ITM & OTM option trades."""
    sig_df = generate_signals(bars5m, daily, params, weekly=weekly, cfg=cfg)
    spy_trades = simulate_spy_trades(sig_df, daily, cfg)
    return {spec.name: price_structure(spy_trades, spec, vix_daily, cfg) for spec in cfg.structures}
