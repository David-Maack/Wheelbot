"""2026-07-01 audit round 2: MED fixes (fractional hold, NaN as-of, ctx
overrides), the pre-FOMC tilt (AI #1), and the adversarial screener pass (AI #5)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest

from backtest.engine import asof_daily_direction
from core.models import MacroEvent, Order, OrderStatus, OrderType, OptionType, Quote
from intelligence.screener import _apply_adversarial
from platforms.paper_broker import PaperBroker
from strategies.swing import _fomc_tilt_signal, swing_exit_from_order


# -- MED-9b root fix: as-of merge must not emit NaN for unknown days ----------


def test_asof_daily_direction_missing_today_is_neutral_not_nan():
    daily_dir = pd.Series([1, 1], index=pd.to_datetime(["2026-06-29", "2026-06-30"]))
    # Intraday includes 07-01, which the (lagged) daily frame doesn't have.
    intraday = pd.to_datetime(["2026-06-30 10:00", "2026-07-01 10:00"])
    out = asof_daily_direction(pd.DatetimeIndex(intraday), daily_dir)
    assert int(out.iloc[0]) == 1      # 06-30 sees 06-29
    assert int(out.iloc[1]) == 0      # unknown day → neutral, NOT NaN/ValueError


# -- MED-8: fractional hold + per-position ctx overrides ----------------------


def _entry_order(ctx: dict) -> Order:
    return Order(
        account_id="t", symbol="SPY", order_type=OrderType.BUY_TO_OPEN,
        contract_symbol="SPY___", strike=585.0, expiration=date(2026, 8, 28),
        option_type=OptionType.CALL, quantity=1, status=OrderStatus.FILLED,
        placed_at=datetime(2026, 6, 29, 18, 0), client_order_id="x",
        strategy_id="spy_swing_opt", raw_request={"swing": ctx},
    )


def test_exit_uses_fractional_hold_from_entry_ts():
    ctx = {"entry_spot": 600.0, "stop_px": 596.0, "target_px": 606.0,
           "direction": 1, "entry_date": "2026-06-29",
           "entry_ts": "2026-06-29T18:00:00"}
    o = _entry_order(ctx)
    # 17.5h later, spot below stop: calendar-days said hold=1 (stop armed);
    # fractional says 0.73 days → still inside min_hold=1 → suppressed.
    now = datetime(2026, 6, 30, 11, 30)
    assert swing_exit_from_order(o, 595.0, now, {}) == (False, None)
    # Past 24h → stop fires.
    later = datetime(2026, 6, 30, 18, 30)
    assert swing_exit_from_order(o, 595.0, later, {}) == (True, "swing_stop")


def test_exit_ctx_overrides_max_hold_for_tilt():
    ctx = {"entry_spot": 600.0, "stop_px": 596.0, "target_px": 606.0,
           "direction": 1, "entry_date": "2026-06-29",
           "entry_ts": "2026-06-29T18:00:00",
           "fomc_tilt": True, "min_hold_days": 0.0, "max_hold_days": 1.0}
    o = _entry_order(ctx)
    # 1.05 days held, flat price → the tilt's 1-day time stop fires (the
    # strategy default of 7 would not).
    now = datetime(2026, 6, 30, 19, 15)
    assert swing_exit_from_order(o, 600.0, now, {"max_hold_days": 7}) == (
        True, "swing_time_stop",
    )


# -- AI #1: pre-FOMC tilt gating ----------------------------------------------


async def _seed_fomc(db_repos, day: date):
    now = datetime.now(UTC).replace(tzinfo=None)
    await db_repos.macro_events.upsert_many([MacroEvent(
        event_date=day, event_type="FOMC", impact="high",
        description="FOMC decision", fetched_at=now, created_at=now, source="test",
    )])


@pytest.mark.asyncio
async def test_fomc_tilt_fires_day_before_event(db_repos):
    today = date(2026, 7, 27)
    await _seed_fomc(db_repos, date(2026, 7, 28))
    broker = PaperBroker(cash=10_000)
    broker.seed_quote(Quote(symbol="SPY", bid=599.9, ask=600.1))
    sig = await _fomc_tilt_signal(broker, db_repos, {"fomc_tilt_enabled": True},
                                  "SPY", today, "spy_swing_opt", "test")
    assert sig is not None and sig.direction == 1 and sig.spot == pytest.approx(600.0)


@pytest.mark.asyncio
async def test_fomc_tilt_respects_gates(db_repos):
    today = date(2026, 7, 27)
    await _seed_fomc(db_repos, date(2026, 7, 28))
    broker = PaperBroker(cash=10_000)
    broker.seed_quote(Quote(symbol="SPY", bid=599.9, ask=600.1))
    # Disabled → no tilt.
    assert await _fomc_tilt_signal(broker, db_repos, {}, "SPY", today,
                                   "spy_swing_opt", "test") is None
    # No FOMC tomorrow → no tilt.
    assert await _fomc_tilt_signal(broker, db_repos, {"fomc_tilt_enabled": True},
                                   "SPY", date(2026, 7, 20), "spy_swing_opt", "test") is None
    # Already entered today (one per event) → no tilt.
    await db_repos.orders.insert(Order(
        account_id="test", symbol="SPY", strategy_id="spy_swing_opt",
        order_type=OrderType.BUY_TO_OPEN, contract_symbol="SPY___", strike=585.0,
        expiration=date(2026, 8, 28), option_type=OptionType.CALL, quantity=1,
        status=OrderStatus.PENDING,
        placed_at=datetime(2026, 7, 27, 15, 0), client_order_id="dup-guard",
    ))
    assert await _fomc_tilt_signal(broker, db_repos, {"fomc_tilt_enabled": True},
                                   "SPY", today, "spy_swing_opt", "test") is None


# -- AI #5: adversarial score application (pure) ------------------------------


def test_apply_adversarial_clamps_and_annotates():
    parsed = {"candidates": [
        {"symbol": "COIN", "score": 80.0, "rationale": "high ivr"},
        {"symbol": "HOOD", "score": 70.0, "rationale": "momo"},
    ]}
    reviews = [
        {"symbol": "COIN", "bull_case": "b", "bear_case": "regulatory overhang", "adjust": -40},
        {"symbol": "HOOD", "bear_case": "fine", "adjust": 5},
        {"symbol": "GHOST", "adjust": -10},  # unknown symbol ignored
    ]
    out = _apply_adversarial(parsed, reviews)
    coin, hood = out["candidates"]
    assert coin["score"] == pytest.approx(65.0)      # -40 clamped to -15
    assert "adv(-15): regulatory overhang" in coin["rationale"]
    assert hood["score"] == pytest.approx(75.0)
    # Fail-open on garbage.
    assert _apply_adversarial(parsed, "not-a-list") is parsed
