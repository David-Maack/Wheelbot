"""Live directional SPY swing strategy.

Sub-sprint 2.1: the live SIGNAL EVALUATOR, in DRY-RUN. It reuses the backtest's
signal logic verbatim so the live signal == the validated backtest:

  fetch recent SPY 5-min bars (Alpaca) + daily/weekly + VIX
    -> backtest.engine.generate_signals (same MTF VWAP/EMA + 200-SMA gate)
    -> read the latest bar's signal

In dry-run, `propose_all_swings` LOGS what it would do and places NOTHING
(returns []). Once paper confirms the live signal matches the backtest, sub-
sprint 2.2 turns the signal into a deep-ITM long-option entry, and 2.3 adds the
shares variant. Validated config (see the backtest): prior-day-level stop,
1-day min-hold, 7-day time stop, no opposite-cross exit, ITM ~0.90 delta / ~60
DTE for the option leg.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

from backtest.data import load_daily_yf, load_intraday_alpaca, load_vix_daily
from backtest.engine import EngineConfig, generate_signals
from core.broker import Broker
from core.checkpoint import log_checkpoint
from core.models import OptionContract, OptionType
from core.strategies import StrategyDefinition
from data.chain import ChainFilters, fetch_filtered_chain
from strategies.swing_signal import Signal, SwingParams, TimeframeSpec

# Default 3-timeframe stack (weekly + daily direction, 5-min trigger), matching
# the backtest. A strategy's `params.timeframes` can override (2 vs 3 TF).
_DEFAULT_PARAMS = SwingParams(timeframes=(
    TimeframeSpec("1W", "direction", vwap_mode="rolling", vwap_window=20),
    TimeframeSpec("1D", "direction", vwap_mode="rolling", vwap_window=20),
    TimeframeSpec("5m", "trigger", vwap_mode="session"),
))


def evaluate_swing_signal(
    bars5m,
    daily,
    *,
    weekly=None,
    params: SwingParams = _DEFAULT_PARAMS,
    cfg: EngineConfig | None = None,
) -> Signal | None:
    """Return the signal on the LATEST bar (a fresh aligned cross + 200-SMA gate),
    or None. Pure — reuses backtest.engine.generate_signals so live == backtest."""
    cfg = cfg or EngineConfig(use_regime=True)
    if bars5m is None or len(bars5m) == 0 or daily is None or len(daily) == 0:
        return None
    sig_df = generate_signals(bars5m, daily, params, weekly=weekly, cfg=cfg)
    if sig_df.empty:
        return None
    last = sig_df.iloc[-1]
    direction = int(last["signal"])
    if direction == 0:
        return None
    return Signal(ts=sig_df.index[-1], direction=direction, spot=float(last["close"]))


# Daily/weekly/VIX only change once a day, but the loop ticks every 5 min — so
# cache them per (symbol, date) and only re-pull intraday 5-min bars each tick.
_DAILY_CACHE: dict[tuple[str, date], tuple] = {}


def _cached_daily(symbol: str, today: date) -> tuple:
    key = (symbol, today)
    if key not in _DAILY_CACHE:
        _DAILY_CACHE.clear()  # drop yesterday's
        _DAILY_CACHE[key] = (
            load_daily_yf(symbol, period="2y"),
            load_daily_yf(symbol, period="2y", weekly=True),
            load_vix_daily(period="2y"),
        )
    return _DAILY_CACHE[key]


def fetch_swing_frames(symbol: str = "SPY", *, lookback_days: int = 10, feed: str = "iex"):
    """Live data: fresh 5-min bars from Alpaca each tick; daily/weekly/VIX from
    yfinance, cached per day. Returns (bars5m, daily, weekly, vix)."""
    end = datetime.now(UTC).replace(tzinfo=None)
    start = end - timedelta(days=lookback_days)
    bars5m = load_intraday_alpaca(symbol, start, end, feed=feed)
    daily, weekly, vix = _cached_daily(symbol, end.date())
    return bars5m, daily, weekly, vix


# --- deep-ITM option selection (2.2a) --------------------------------------
def _mid(c: OptionContract) -> float | None:
    if c.bid is None or c.ask is None:
        return None
    m = (c.bid + c.ask) / 2
    return m if m > 0 else None


def pick_deep_itm(candidates: list[OptionContract], target_delta: float) -> OptionContract | None:
    """The contract whose |delta| is closest to target_delta (default ~0.90),
    among those with a usable mid price. Pure — easy to unit-test."""
    scored = [
        (abs(abs(c.delta) - target_delta), c)
        for c in candidates
        if c.delta is not None and _mid(c) is not None
    ]
    if not scored:
        return None
    scored.sort(key=lambda t: t[0])
    return scored[0][1]


async def select_swing_option(
    broker: Broker, symbol: str, direction: int, params: dict[str, Any], *, today: date | None = None
) -> OptionContract | None:
    """Pick the deep-ITM long CALL (direction +1) or PUT (-1) at ~entry_delta,
    in the DTE band. Mirrors PMCC's select_long_call."""
    today = today or date.today()
    target = float(params.get("entry_delta", 0.90))
    dte = int(params.get("dte_target", 60))
    band = int(params.get("dte_band", 15))
    opt = "call" if direction > 0 else "put"
    filters = ChainFilters(
        dte_min=max(1, dte - band),
        dte_max=dte + band,
        delta_min=max(0.50, target - 0.12),
        delta_max=min(0.98, target + 0.08),
        open_interest_min=int(params.get("open_interest_min", 0)),
        volume_min=int(params.get("volume_min", 0)),
        bid_ask_spread_max_pct=float(params.get("bid_ask_spread_max_pct", 100.0)),
    )
    candidates = await fetch_filtered_chain(broker, symbol, opt, filters, today=today)
    best = pick_deep_itm(candidates, target)
    if best is None:
        log_checkpoint("swing_no_option_candidates", status="ok", symbol=symbol, direction=direction)
    return best


# --- SPY-level exit decision (2.2a) ----------------------------------------
def swing_stop_target(
    entry_spot: float, direction: int, prior_day_level: float, reward_risk: float
) -> tuple[float, float]:
    """Stop anchored to the prior day's low (long) / high (short); target at an
    R-multiple of the stop distance. Matches the validated backtest exits."""
    if direction > 0:
        stop = prior_day_level
        dist = max(entry_spot - stop, 1e-9)
        return stop, entry_spot + reward_risk * dist
    stop = prior_day_level
    dist = max(stop - entry_spot, 1e-9)
    return stop, entry_spot - reward_risk * dist


def swing_exit_decision(
    direction: int,
    current_spot: float,
    stop_px: float,
    target_px: float,
    hold_days: float,
    *,
    min_hold_days: float,
    max_hold_days: float,
) -> tuple[bool, str | None]:
    """Decide whether to close, checked against the latest SPY quote each tick.
    Stop is suppressed until min_hold_days (let the trade breathe); target and
    time-stop always apply. Stop is checked before target (conservative)."""
    allow_stop = hold_days >= min_hold_days
    if direction > 0:
        if allow_stop and current_spot <= stop_px:
            return True, "swing_stop"
        if current_spot >= target_px:
            return True, "swing_target"
    else:
        if allow_stop and current_spot >= stop_px:
            return True, "swing_stop"
        if current_spot <= target_px:
            return True, "swing_target"
    if hold_days >= max_hold_days:
        return True, "swing_time_stop"
    return False, None


async def propose_all_swings(
    broker,
    repos,
    config: dict[str, Any],
    universe: dict[str, Any] | None = None,
    *,
    strategy: StrategyDefinition,
    size_multiplier: float = 1.0,
    today: date | None = None,
) -> list:
    """DRY-RUN entry pass: compute + log the live signal, place nothing.

    Returns [] so the router has nothing to execute. Sub-sprint 2.2 replaces the
    log-only branch with a real deep-ITM long-option `Proposal`.
    """
    p = strategy.params or {}
    symbol = str(p.get("symbol", "SPY"))
    dry_run = bool(p.get("dry_run", True))
    try:
        bars5m, daily, weekly, _vix = fetch_swing_frames(symbol)
    except Exception as exc:  # network / keys / data — never crash the loop
        log_checkpoint("swing_fetch_fail", status="fail", strategy=strategy.id, error=str(exc))
        return []

    sig = evaluate_swing_signal(bars5m, daily, weekly=weekly)
    if sig is None:
        log_checkpoint(
            "swing_eval", status="ok", strategy=strategy.id, symbol=symbol,
            signal=0, bars5m=len(bars5m), daily=len(daily),
        )
        return []

    log_checkpoint(
        "swing_signal_fired", status="ok", strategy=strategy.id, symbol=symbol,
        direction=sig.direction, spot=round(sig.spot, 2), ts=str(sig.ts),
        dry_run=dry_run,
    )
    # 2.2: build a Proposal (deep-ITM long call/put) here when not dry_run.
    return []


async def propose_all_swing_closes(
    broker, repos, config: dict[str, Any], *, strategy: StrategyDefinition
) -> list:
    """No open swing positions until entries land in sub-sprint 2.2. Returns []."""
    return []
