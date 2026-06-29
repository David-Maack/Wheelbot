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
from core.models import OptionContract, Order, OptionType, OrderType, PositionState
from core.strategies import StrategyDefinition
from data.chain import ChainFilters, fetch_filtered_chain
from db.repo import Repos
from strategies.pmcc import latest_filled_order
from strategies.swing_signal import Signal, SwingParams, TimeframeSpec
from strategies.wheel import Proposal, _tier_flags

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


def prior_day_level(daily, today: date, direction: int) -> float | None:
    """Prior completed daily bar's low (long) / high (short) — the stop anchor."""
    if daily is None or len(daily) == 0:
        return None
    prior = daily[daily.index.date < today]
    if len(prior) == 0:
        return None
    row = prior.iloc[-1]
    return float(row["low"] if direction > 0 else row["high"])


def build_swing_entry(
    symbol: str, contract: OptionContract, sig: Signal, prior_level: float,
    params: dict[str, Any], universe: dict[str, Any], today: date, strategy_id: str,
) -> Proposal:
    """A deep-ITM long-option BUY_TO_OPEN with the SPY-level exit anchors stashed
    on raw_request. Long call (direction +1) or long put (-1)."""
    rr = float(params.get("reward_risk", 1.5))
    stop_px, target_px = swing_stop_target(sig.spot, sig.direction, prior_level, rr)
    side = "long" if sig.direction > 0 else "short"
    dte = (contract.expiration - today).days
    needs_screen, needs_human = _tier_flags(symbol, universe)
    return Proposal(
        symbol=symbol, contract=contract, order_type=OrderType.BUY_TO_OPEN,
        quantity=int(params.get("contracts", 1)),
        rationale=(f"swing[{symbol}] {side} {contract.strike} dte={dte} "
                   f"stop={stop_px:.2f} tgt={target_px:.2f}"),
        strategy_id=strategy_id, requires_screen=needs_screen, requires_human=needs_human,
        # Long calls news-checked as bullish; long puts have no bearish profile
        # yet, so rely on the macro-blackout gate (which fires on all proposals).
        news_check_profile="bullish_long" if sig.direction > 0 else None,
        raw_request={"swing": {
            "entry_spot": sig.spot, "stop_px": stop_px, "target_px": target_px,
            "entry_date": today.isoformat(), "direction": sig.direction,
        }},
    )


def swing_exit_from_order(
    entry_order: Order, current_spot: float, now: datetime, params: dict[str, Any]
) -> tuple[bool, str | None]:
    """Read the entry context off the entry order's raw_request and decide the
    exit against the live SPY quote. Anchored to entry, like the backtest."""
    ctx = (entry_order.raw_request or {}).get("swing", {}) if entry_order.raw_request else {}
    stop_px, target_px, entry_date = ctx.get("stop_px"), ctx.get("target_px"), ctx.get("entry_date")
    if stop_px is None or target_px is None or entry_date is None:
        return False, None
    direction = int(ctx.get("direction", 1 if entry_order.option_type == OptionType.CALL else -1))
    hold_days = float((now.date() - date.fromisoformat(entry_date)).days)
    return swing_exit_decision(
        direction, current_spot, float(stop_px), float(target_px), hold_days,
        min_hold_days=float(params.get("min_hold_days", 1)),
        max_hold_days=float(params.get("max_hold_days", 7)),
    )


def _spot_from_quote(q) -> float | None:
    return q.mid if getattr(q, "mid", None) else getattr(q, "last", None)


async def propose_all_swings(
    broker: Broker,
    repos: Repos,
    config: dict[str, Any],
    universe: dict[str, Any] | None = None,
    *,
    strategy: StrategyDefinition,
    size_multiplier: float = 1.0,
    today: date | None = None,
) -> list[Proposal]:
    """Entry pass: on a fresh signal with the position IDLE, propose a deep-ITM
    long. The SWING_PENDING state (set by the router on place) guards against
    duplicate opens next tick. In dry-run, log the full proposal and place none."""
    today = today or date.today()
    p = strategy.params or {}
    symbol = str(p.get("symbol", "SPY"))
    dry_run = bool(p.get("dry_run", True))
    account_id = config.get("account", {}).get("id", "primary")

    position = await repos.positions.get_by_symbol(account_id, symbol, strategy_id=strategy.id)
    state = position.state if position else PositionState.IDLE
    if state != PositionState.IDLE:
        log_checkpoint("swing_skip_state", status="ok", strategy=strategy.id, state=str(state))
        return []

    try:
        bars5m, daily, weekly, _vix = fetch_swing_frames(symbol)
    except Exception as exc:  # network / keys / data — never crash the loop
        log_checkpoint("swing_fetch_fail", status="fail", strategy=strategy.id, error=str(exc))
        return []

    sig = evaluate_swing_signal(bars5m, daily, weekly=weekly)
    if sig is None:
        log_checkpoint("swing_eval", status="ok", strategy=strategy.id, symbol=symbol,
                       signal=0, bars5m=len(bars5m), daily=len(daily))
        return []

    contract = await select_swing_option(broker, symbol, sig.direction, p, today=today)
    level = prior_day_level(daily, today, sig.direction)
    if contract is None or level is None:
        log_checkpoint("swing_no_entry", status="ok", strategy=strategy.id, symbol=symbol,
                       have_contract=contract is not None, have_level=level is not None)
        return []

    proposal = build_swing_entry(symbol, contract, sig, level, p, universe or {"tickers": []},
                                 today, strategy.id)
    log_checkpoint(
        "swing_signal_fired", status="ok", strategy=strategy.id, symbol=symbol,
        direction=sig.direction, spot=round(sig.spot, 2), strike=contract.strike,
        dte=(contract.expiration - today).days, stop=round(level, 2),
        rationale=proposal.rationale, dry_run=dry_run,
    )
    return [] if dry_run else [proposal]


async def propose_all_swing_closes(
    broker: Broker, repos: Repos, config: dict[str, Any], *, strategy: StrategyDefinition
) -> list[Proposal]:
    """Exit pass: for a SWING_OPEN position, recover entry context, check the
    live SPY quote against the stop/target/time rules, and propose SELL_TO_CLOSE.
    In dry-run, log and place none."""
    p = strategy.params or {}
    symbol = str(p.get("symbol", "SPY"))
    dry_run = bool(p.get("dry_run", True))
    account_id = config.get("account", {}).get("id", "primary")

    position = await repos.positions.get_by_symbol(account_id, symbol, strategy_id=strategy.id)
    if position is None or position.state != PositionState.SWING_OPEN:
        return []
    entry_order = await latest_filled_order(repos, position.current_cycle_id, OrderType.BUY_TO_OPEN)
    if entry_order is None:
        log_checkpoint("swing_close_no_entry_order", status="fail", strategy=strategy.id)
        return []

    try:
        spot = _spot_from_quote(await broker.get_quote(symbol))
    except Exception as exc:
        log_checkpoint("swing_close_quote_fail", status="fail", strategy=strategy.id, error=str(exc))
        return []
    if spot is None:
        return []

    should_exit, reason = swing_exit_from_order(entry_order, float(spot), datetime.now(UTC), p)
    if not should_exit:
        return []

    try:
        oq = await broker.get_quote(entry_order.contract_symbol)
    except Exception as exc:
        log_checkpoint("swing_close_option_quote_fail", status="fail", strategy=strategy.id, error=str(exc))
        return []
    close_contract = OptionContract(
        underlying=symbol, occ_symbol=entry_order.contract_symbol, strike=entry_order.strike,
        expiration=entry_order.expiration, option_type=entry_order.option_type,
        bid=getattr(oq, "bid", None), ask=getattr(oq, "ask", None),
    )
    proposal = Proposal(
        symbol=symbol, contract=close_contract, order_type=OrderType.SELL_TO_CLOSE,
        quantity=entry_order.quantity or 1,
        rationale=f"swing_close[{symbol}] {reason} spot={spot:.2f}",
        strategy_id=strategy.id, trigger_reason=reason,
    )
    log_checkpoint("swing_close_fired", status="ok", strategy=strategy.id, symbol=symbol,
                   reason=reason, spot=round(float(spot), 2), dry_run=dry_run)
    return [] if dry_run else [proposal]
