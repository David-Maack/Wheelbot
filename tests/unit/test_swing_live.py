"""Live swing signal evaluator (reuses the backtest signal → live == backtest)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from datetime import UTC, date, datetime

import pytest

from backtest.engine import EngineConfig
from core.models import OptionContract, OptionType, Order, OrderStatus, OrderType
from strategies.swing import (
    build_swing_entry,
    evaluate_swing_signal,
    pick_deep_itm,
    prior_day_level,
    swing_exit_decision,
    swing_exit_from_order,
    swing_stop_target,
)
from strategies.swing_signal import Signal, SwingParams, TimeframeSpec

# 2-timeframe stack (daily direction + 5-min trigger) keeps the test self-contained.
_PARAMS = SwingParams(timeframes=(
    TimeframeSpec("1D", "direction", vwap_mode="rolling", vwap_window=20),
    TimeframeSpec("5m", "trigger", vwap_mode="session"),
))
_CFG = EngineConfig(use_regime=True, regime_sma=3)  # small SMA so warm-up is short


def _frame(dates, closes, *, hl=0.2, vol=1000):
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {"open": closes, "high": closes + hl, "low": closes - hl, "close": closes,
         "volume": [vol] * len(closes)},
        index=pd.DatetimeIndex(dates),
    )


def test_evaluate_returns_signal_on_fresh_last_bar_cross():
    # Daily uptrend → direction +1 and above the 3-SMA (regime +1).
    ddates = pd.date_range("2026-06-01", periods=5, freq="1D")
    daily = _frame(ddates, [90, 92, 94, 96, 98], hl=1.0)
    # 5-min flat all session, then a sharp up-move on the FINAL bar → EMA crosses
    # VWAP exactly on the last bar (a fresh trigger).
    idx = pd.date_range("2026-06-05 09:30", periods=12, freq="5min")
    five = _frame(idx, [100] * 11 + [110])
    sig = evaluate_swing_signal(five, daily, params=_PARAMS, cfg=_CFG)
    assert sig is not None
    assert sig.direction == 1
    assert sig.spot == 110.0
    assert sig.ts == idx[-1]


def test_evaluate_returns_none_when_flat():
    ddates = pd.date_range("2026-06-01", periods=5, freq="1D")
    daily = _frame(ddates, [90, 92, 94, 96, 98], hl=1.0)
    idx = pd.date_range("2026-06-05 09:30", periods=12, freq="5min")
    flat = _frame(idx, [100] * 12)  # no cross → no signal
    assert evaluate_swing_signal(flat, daily, params=_PARAMS, cfg=_CFG) is None


def test_evaluate_handles_empty_frames():
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    assert evaluate_swing_signal(empty, empty, params=_PARAMS, cfg=_CFG) is None


# --- deep-ITM option selection (2.2a) --------------------------------------
def _call(strike, delta, bid=10.0, ask=10.2):
    return OptionContract(
        underlying="SPY", occ_symbol=f"SPY___C{strike}", strike=strike,
        expiration=date(2026, 8, 21), option_type=OptionType.CALL,
        bid=bid, ask=ask, delta=delta,
    )


def test_pick_deep_itm_closest_to_target_delta():
    cands = [_call(560, 0.85), _call(550, 0.92), _call(540, 0.97)]
    best = pick_deep_itm(cands, target_delta=0.90)
    assert best.strike == 550  # delta 0.92 is closest to 0.90


def test_pick_deep_itm_skips_unpriced():
    cands = [_call(550, 0.90, bid=None, ask=None), _call(560, 0.80)]
    best = pick_deep_itm(cands, target_delta=0.90)
    assert best.strike == 560  # the 0.90 has no mid → skipped


def test_pick_deep_itm_empty():
    assert pick_deep_itm([], target_delta=0.90) is None


# --- SPY-level stop / target / exit (2.2a) ---------------------------------
def test_swing_stop_target_long():
    stop, target = swing_stop_target(600.0, direction=1, prior_day_level=596.0, reward_risk=1.5)
    assert stop == 596.0
    assert target == 600.0 + 1.5 * 4.0  # entry + 1.5 * (600-596) = 606


def test_swing_stop_target_short():
    stop, target = swing_stop_target(600.0, direction=-1, prior_day_level=604.0, reward_risk=1.5)
    assert stop == 604.0
    assert target == 600.0 - 1.5 * 4.0  # 594


def test_swing_exit_min_hold_suppresses_stop():
    # Long, price below stop, but within min-hold → no stop yet.
    e, r = swing_exit_decision(1, 595.0, 596.0, 606.0, hold_days=0.2,
                               min_hold_days=1.0, max_hold_days=7.0)
    assert e is False and r is None
    # Past min-hold → stop fires.
    e, r = swing_exit_decision(1, 595.0, 596.0, 606.0, hold_days=1.5,
                               min_hold_days=1.0, max_hold_days=7.0)
    assert e is True and r == "swing_stop"


def test_swing_exit_target_and_time():
    e, r = swing_exit_decision(1, 607.0, 596.0, 606.0, hold_days=0.1,
                               min_hold_days=1.0, max_hold_days=7.0)
    assert e is True and r == "swing_target"  # target fires even before min-hold
    e, r = swing_exit_decision(1, 600.0, 596.0, 606.0, hold_days=7.0,
                               min_hold_days=1.0, max_hold_days=7.0)
    assert e is True and r == "swing_time_stop"


# --- entry/exit proposal builders (2.2b-ii) --------------------------------
def _daily_frame():
    idx = pd.DatetimeIndex(["2026-06-24", "2026-06-25", "2026-06-26"])
    return pd.DataFrame({"open": 600, "high": [604, 605, 606], "low": [596, 597, 595],
                         "close": 600, "volume": 1000}, index=idx)


def test_prior_day_level_long_uses_prior_low():
    daily = _daily_frame()
    # "today" = 06-26 → prior completed bar is 06-25 (low 597 / high 605).
    assert prior_day_level(daily, date(2026, 6, 26), direction=1) == 597.0
    assert prior_day_level(daily, date(2026, 6, 26), direction=-1) == 605.0


def _opt(strike, exp, ot):
    return OptionContract(underlying="SPY", occ_symbol="SPY___", strike=strike,
                          expiration=exp, option_type=ot, bid=20.0, ask=20.2, delta=0.9)


def test_build_swing_entry_long_stashes_exit_anchors():
    sig = Signal(ts=None, direction=1, spot=600.0)
    contract = _opt(585.0, date(2026, 8, 21), OptionType.CALL)
    prop = build_swing_entry("SPY", contract, sig, prior_level=596.0,
                             params={"reward_risk": 1.5, "contracts": 1},
                             universe={"tickers": []}, today=date(2026, 6, 26),
                             strategy_id="spy_swing_opt")
    assert prop.order_type == OrderType.BUY_TO_OPEN
    assert prop.news_check_profile == "bullish_long"
    sw = prop.raw_request["swing"]
    assert sw["stop_px"] == 596.0
    assert sw["target_px"] == 600.0 + 1.5 * 4.0  # 606
    assert sw["direction"] == 1


def test_build_swing_entry_short_has_no_bullish_news_profile():
    sig = Signal(ts=None, direction=-1, spot=600.0)
    contract = _opt(615.0, date(2026, 8, 21), OptionType.PUT)
    prop = build_swing_entry("SPY", contract, sig, prior_level=604.0,
                             params={}, universe={"tickers": []},
                             today=date(2026, 6, 26), strategy_id="spy_swing_opt")
    assert prop.news_check_profile is None
    assert prop.raw_request["swing"]["target_px"] == 600.0 - 1.5 * 4.0  # 594


def _entry_order(stop, target, entry_date, direction=1):
    return Order(
        account_id="t", symbol="SPY", order_type=OrderType.BUY_TO_OPEN,
        contract_symbol="SPY___", strike=585.0, expiration=date(2026, 8, 21),
        option_type=OptionType.CALL, quantity=1, status=OrderStatus.FILLED,
        placed_at=datetime(2026, 6, 24, tzinfo=UTC).replace(tzinfo=None),
        client_order_id="x", strategy_id="spy_swing_opt",
        raw_request={"swing": {"stop_px": stop, "target_px": target,
                               "entry_date": entry_date, "direction": direction}},
    )


@pytest.mark.asyncio
async def test_swing_exit_anchors_survive_broker_round_trip(db_repos, monkeypatch):
    """CRITICAL regression: brokers REPLACE Order.raw_request with their own
    request dump on placement (alpaca model_copy / paper dump). The router must
    merge the proposal's swing exit anchors back before persisting, else the
    DB row has no {swing: stop/target/...} and stops/targets/time exits NEVER
    fire. This test runs the REAL router->PaperBroker->DB path."""
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)
    monkeypatch.setattr("execution.router.within_entry_window", lambda **k: True)
    from core.config import UniverseEntry
    from execution.router import OrderRouter
    from platforms.paper_broker import PaperBroker
    from strategies.swing import build_swing_entry

    config = {
        "account": {"id": "test", "broker": "paper"},
        "strategies": [{"id": "spy_swing_opt", "type": "swing", "enabled": True,
                        "max_concurrent": 1, "params": {"symbol": "SPY"}}],
    }
    universe = {"tickers": [UniverseEntry(symbol="SPY", name="SPDR", tier=1, overrides={})],
                "banned": [], "banned_rules": []}
    router = OrderRouter(PaperBroker(cash=100_000), db_repos, config, universe)

    contract = OptionContract(
        underlying="SPY", occ_symbol="SPY260828C00590000", strike=590.0,
        expiration=date(2026, 8, 28), option_type=OptionType.CALL,
        bid=19.9, ask=20.1, delta=0.9,
    )
    prop = build_swing_entry(
        "SPY", contract, Signal(ts=None, direction=1, spot=600.0), prior_level=596.0,
        params={}, universe=universe, today=date(2026, 6, 30), strategy_id="spy_swing_opt",
    )

    async def _noop(_s):
        return None

    result = await router.place(prop, sleep=_noop, today=date(2026, 6, 30))
    assert result.placed is not None

    persisted = [o for o in await db_repos.orders.list_recent("test", limit=10)
                 if o.symbol == "SPY" and o.order_type == OrderType.BUY_TO_OPEN]
    assert persisted, "swing entry order not persisted"
    row = persisted[0]
    # The anchors must survive the broker round-trip into the DB row...
    assert (row.raw_request or {}).get("swing", {}).get("stop_px") == 596.0
    # ...and the exit logic must be able to fire from that row.
    now = datetime(2026, 7, 2, tzinfo=UTC).replace(tzinfo=None)
    assert swing_exit_from_order(row, 610.0, now, {}) == (True, "swing_target")


def test_swing_exit_from_order_reads_anchors():
    o = _entry_order(596.0, 606.0, "2026-06-24")
    now = datetime(2026, 6, 26, tzinfo=UTC).replace(tzinfo=None)  # hold 2 days
    assert swing_exit_from_order(o, 607.0, now, {}) == (True, "swing_target")
    assert swing_exit_from_order(o, 595.0, now, {}) == (True, "swing_stop")
    assert swing_exit_from_order(o, 600.0, now, {}) == (False, None)
    # 8 days held → time stop.
    later = datetime(2026, 7, 2, tzinfo=UTC).replace(tzinfo=None)
    assert swing_exit_from_order(o, 600.0, later, {}) == (True, "swing_time_stop")


# -- close orchestrator end-to-end (2026-07-23 review fix) ---------------------


@pytest.mark.asyncio
async def test_swing_close_stop_fires_end_to_end(db_repos):
    """Regression for the aware/naive datetime bug: with entry_ts persisted
    (naive UTC), the close pass used to raise TypeError on the hold-time
    subtraction and NO exit ever evaluated. This walks the real orchestrator:
    SWING_OPEN + stop breached -> SELL_TO_CLOSE proposal."""
    from datetime import UTC, date, datetime, timedelta
    from core.models import (Order, OrderStatus, OrderType, OptionType,
                             Position, PositionState, Quote, WheelCycle)
    from core.strategies import StrategyDefinition
    from platforms.paper_broker import PaperBroker
    from strategies.swing import propose_all_swing_closes

    now = datetime.now(UTC).replace(tzinfo=None)
    cycle_id = await db_repos.cycles.insert(
        WheelCycle(account_id="test", symbol="SPY", strategy_id="spy_swing_opt",
                   started_at=now - timedelta(days=2))
    )
    exp = date.today() + timedelta(days=55)
    await db_repos.orders.insert(Order(
        account_id="test", symbol="SPY", strategy_id="spy_swing_opt",
        cycle_id=cycle_id, order_type=OrderType.BUY_TO_OPEN,
        contract_symbol="SPY261016C00450000", strike=450.0, expiration=exp,
        option_type=OptionType.CALL, quantity=1, fill_price=15.0,
        status=OrderStatus.FILLED, placed_at=now - timedelta(days=2),
        client_order_id="swing-entry-1",
        raw_request={"swing": {
            "direction": 1, "stop_px": 480.0, "target_px": 520.0,
            "entry_date": (now - timedelta(days=2)).date().isoformat(),
            # entry_ts persisted NAIVE, as build_swing_entry writes it.
            "entry_ts": (now - timedelta(days=2)).isoformat(),
        }},
    ))
    await db_repos.positions.insert(Position(
        account_id="test", symbol="SPY", strategy_id="spy_swing_opt",
        state=PositionState.SWING_OPEN, shares=0, current_cycle_id=cycle_id,
        state_changed_at=now,
    ))
    broker = PaperBroker()
    broker.seed_quote(Quote(symbol="SPY", bid=469.9, ask=470.1))  # below 480 stop
    broker.seed_quote(Quote(symbol="SPY261016C00450000", bid=21.0, ask=21.6))

    strategy = StrategyDefinition(
        id="spy_swing_opt", display_name="Swing", type="swing", enabled=True,
        max_concurrent=1, params={"symbol": "SPY", "dry_run": False,
                                  "min_hold_days": 1, "max_hold_days": 7},
    )
    proposals = await propose_all_swing_closes(
        broker, db_repos, {"account": {"id": "test"}}, strategy=strategy,
    )
    assert len(proposals) == 1
    assert proposals[0].order_type == OrderType.SELL_TO_CLOSE
    assert "stop" in (proposals[0].trigger_reason or proposals[0].rationale)


# -- signal delivery window (2026-07-24 parity fix) ----------------------------


def _sig_frame(cross_offset_bars: int, n_bars: int = 10, direction: int = 1):
    """Signal frame with one nonzero bar `cross_offset_bars` from the end."""
    import pandas as pd
    idx = pd.date_range("2026-07-20 09:30", periods=n_bars, freq="5min")
    sig = [0] * n_bars
    sig[n_bars - 1 - cross_offset_bars] = direction
    close = [100.0 + i * 0.1 for i in range(n_bars)]
    return pd.DataFrame({"signal": sig, "close": close}, index=idx)


def _dummy_bars():
    import pandas as pd
    idx = pd.date_range("2026-07-20 09:30", periods=3, freq="5min")
    return pd.DataFrame({"close": [1.0, 1.0, 1.0]}, index=idx)


def test_cross_on_skipped_bar_now_fires(monkeypatch):
    """The July failure mode: the cross printed 2 bars (10 min) ago — the old
    latest-bar-only check returned None; the windowed scan fires it."""
    from strategies.swing import evaluate_swing_signal
    frame = _sig_frame(cross_offset_bars=2)
    monkeypatch.setattr("strategies.swing.generate_signals", lambda *a, **k: frame)
    diag = {}
    sig = evaluate_swing_signal(_dummy_bars(), _dummy_bars(), diag=diag)
    assert sig is not None
    assert sig.direction == 1
    assert sig.ts == frame.index[-3]                  # identity = the cross bar
    assert sig.spot == frame["close"].iloc[-1]        # economics = current close
    assert diag["n_new_since_last_eval"] == 1
    assert diag["last_cross_age_min"] == 10.0


def test_since_dedupes_a_fired_cross(monkeypatch):
    """Once `since` has advanced past the cross bar, it can never re-fire."""
    from strategies.swing import evaluate_swing_signal
    frame = _sig_frame(cross_offset_bars=2)
    monkeypatch.setattr("strategies.swing.generate_signals", lambda *a, **k: frame)
    sig = evaluate_swing_signal(_dummy_bars(), _dummy_bars(), since=frame.index[-1])
    assert sig is None


def test_stale_cross_outside_freshness_window(monkeypatch):
    """A cross older than max_age_minutes must not chase."""
    from strategies.swing import evaluate_swing_signal
    frame = _sig_frame(cross_offset_bars=6, n_bars=12)  # 30 min old
    monkeypatch.setattr("strategies.swing.generate_signals", lambda *a, **k: frame)
    assert evaluate_swing_signal(_dummy_bars(), _dummy_bars(),
                                 max_age_minutes=15.0) is None


def test_cross_on_latest_bar_still_fires(monkeypatch):
    """Regression: the original semantics remain a subset of the new ones."""
    from strategies.swing import evaluate_swing_signal
    frame = _sig_frame(cross_offset_bars=0, direction=-1)
    monkeypatch.setattr("strategies.swing.generate_signals", lambda *a, **k: frame)
    sig = evaluate_swing_signal(_dummy_bars(), _dummy_bars())
    assert sig is not None and sig.direction == -1
    assert sig.ts == frame.index[-1]
