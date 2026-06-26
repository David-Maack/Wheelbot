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
from core.checkpoint import log_checkpoint
from core.strategies import StrategyDefinition
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


def fetch_swing_frames(symbol: str = "SPY", *, lookback_days: int = 10, feed: str = "iex"):
    """Live data: 5-min bars from Alpaca, daily/weekly/VIX from yfinance — the
    same sources the backtest used. Returns (bars5m, daily, weekly, vix)."""
    end = datetime.now(UTC).replace(tzinfo=None)
    start = end - timedelta(days=lookback_days)
    bars5m = load_intraday_alpaca(symbol, start, end, feed=feed)
    daily = load_daily_yf(symbol, period="2y")
    weekly = load_daily_yf(symbol, period="2y", weekly=True)
    vix = load_vix_daily(period="2y")
    return bars5m, daily, weekly, vix


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
