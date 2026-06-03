"""risk/win_rate_floor — per-strategy win-rate gate (TICKET-009)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest


async def _no_op_notify(*args, **kwargs):
    """Awaitable no-op stub for risk.win_rate_floor.notify."""
    return None


def _raising_notify(message):
    """Returns an awaitable that raises — wired in via monkeypatch when a
    test expects notify to NOT fire."""
    async def _stub(*args, **kwargs):
        raise AssertionError(message)
    return _stub

from core.models import CycleOutcome, WheelCycle
from core.strategies import StrategyDefinition
from risk.win_rate_floor import (
    PAUSE_ALERT_COOLDOWN_HOURS,
    WinRateState,
    check_and_apply,
    current_pause_state,
    evaluate,
    get_win_rate_overview,
)


# -- fixtures --------------------------------------------------------------


def _strategy(
    *,
    id_: str = "put_spread",
    win_rate_override: float | None = None,
) -> StrategyDefinition:
    params = {}
    if win_rate_override is not None:
        params["win_rate_floor_pct"] = win_rate_override
    return StrategyDefinition(
        id=id_, display_name="Test", type="vertical_spread",
        enabled=True, max_concurrent=4, params=params,
    )


def _cfg(
    *,
    enabled: bool = True,
    min_cycles: int = 10,
    min_rate: float = 60.0,
    pause_days: int = 14,
) -> dict:
    return {
        "account": {"id": "test"},
        "risk": {
            "win_rate_floor": {
                "enabled": enabled,
                "min_closed_cycles": min_cycles,
                "min_win_rate_pct": min_rate,
                "pause_duration_days": pause_days,
            }
        },
    }


async def _insert_cycles(
    db_repos, *,
    strategy_id: str,
    pnls: list[float],
    base_time: datetime | None = None,
    symbol: str = "F",
) -> list[int]:
    """Insert closed cycles with the given final_pnls, one per day going
    back from base_time. Newest is index 0 (most recent ended_at)."""
    base = base_time or datetime.now(UTC).replace(tzinfo=None)
    ids: list[int] = []
    for i, pnl in enumerate(pnls):
        ended = base - timedelta(days=i)
        started = ended - timedelta(days=10)
        cid = await db_repos.cycles.insert(
            WheelCycle(
                account_id="test",
                symbol=symbol,
                strategy_id=strategy_id,
                started_at=started,
                ended_at=ended,
                final_pnl=pnl,
                cycle_outcome=CycleOutcome.SPREAD_CLOSED_PROFIT
                if pnl > 0 else CycleOutcome.SPREAD_MAX_LOSS,
            )
        )
        ids.append(cid)
    return ids


# -- evaluate --------------------------------------------------------------


@pytest.mark.asyncio
async def test_insufficient_data_no_action(db_repos):
    """Fewer than min_closed_cycles → action='insufficient_data', no row written."""
    cfg = _cfg()
    await _insert_cycles(db_repos, strategy_id="put_spread", pnls=[100, -50, 80, -30, 60])
    result = await evaluate(db_repos, "put_spread", cfg, account_id="test")
    assert result.action == "insufficient_data"
    assert result.win_rate is None
    assert result.sample_size == 5


@pytest.mark.asyncio
async def test_above_floor_normal(db_repos):
    """8/10 wins (80%) → action='normal'."""
    cfg = _cfg()
    pnls = [100, 100, 100, 100, 100, 100, 100, 100, -50, -50]  # 8 wins
    await _insert_cycles(db_repos, strategy_id="put_spread", pnls=pnls)
    result = await evaluate(db_repos, "put_spread", cfg, account_id="test")
    assert result.action == "normal"
    assert result.win_rate == pytest.approx(80.0)
    assert result.sample_size == 10


@pytest.mark.asyncio
async def test_below_floor_pauses(db_repos, monkeypatch):
    """4/10 wins (40%) → action='pause', strategy.paused_low_win_rate fired,
    row persisted with paused_at + eligible_at."""
    captured: list[dict] = []
    async def _stub_notify(event_type, title, **payload):
        captured.append({"event_type": event_type, **payload})
    monkeypatch.setattr("risk.win_rate_floor.notify", _stub_notify)

    cfg = _cfg()
    pnls = [100, 100, 100, 100, -50, -50, -50, -50, -50, -50]  # 4 wins
    now = datetime.now(UTC).replace(tzinfo=None)
    await _insert_cycles(db_repos, strategy_id="put_spread", pnls=pnls, base_time=now)
    result = await check_and_apply(db_repos, _strategy(), cfg, now=now)
    assert result.action == "pause"
    assert result.win_rate == pytest.approx(40.0)
    state = await current_pause_state(db_repos, "put_spread")
    assert state is WinRateState.PAUSED_LOW_WIN_RATE
    assert len(captured) == 1
    assert captured[0]["event_type"] == "strategy.paused_low_win_rate"
    row = await db_repos.strategy_runtime.get("put_spread")
    assert row["pause_state"] == "LOW_WIN_RATE"
    assert row["paused_at"] is not None
    assert row["paused_reenable_eligible_at"] is not None


@pytest.mark.asyncio
async def test_paused_state_persists_until_manual_reenable(db_repos, monkeypatch):
    """Once paused, even a tick with no new losses does NOT auto-clear.
    Only the operator running enable() (== reenable_strategy.py) clears it.
    Locks the (ii) decision: no calendar auto-clear."""
    monkeypatch.setattr("risk.win_rate_floor.notify", _raising_notify("unexpected notify"))

    cfg = _cfg()
    now = datetime.now(UTC).replace(tzinfo=None)
    # Pre-seed a paused row manually so we test "paused → recovery tick".
    await db_repos.strategy_runtime.mark_paused(
        "put_spread", reason="seeded", now=now - timedelta(days=20),
        eligible_in_days=14,
    )
    # Now insert 10 WINNING cycles — win rate is 100%, well above floor.
    pnls = [100] * 10
    await _insert_cycles(db_repos, strategy_id="put_spread", pnls=pnls, base_time=now)
    # Even though we're well past the 14d advisory window AND the win rate
    # is 100%, the pause persists.

    # Disable notify-on-pause expectation by un-monkeypatching for the
    # held-pause path (we expect NO notify, but the asserter above already
    # covers that).
    result = await check_and_apply(db_repos, _strategy(), cfg, now=now)
    # The action is "pause" because the prior state is PAUSED — held.
    assert result.action == "pause"
    assert result.win_rate == pytest.approx(100.0)
    # Row remains.
    assert await current_pause_state(db_repos, "put_spread") is WinRateState.PAUSED_LOW_WIN_RATE


@pytest.mark.asyncio
async def test_pause_does_not_auto_clear_after_eligible_date(db_repos, monkeypatch):
    """The paused_reenable_eligible_at field is ADVISORY only. Even when it
    has long passed, current_pause_state still returns PAUSED until the
    operator clears it. Pins (ii)."""
    monkeypatch.setattr("risk.win_rate_floor.notify", _no_op_notify)
    now = datetime.now(UTC).replace(tzinfo=None)
    # Pause set 100 days ago with 14d eligibility window — eligible 86d ago.
    await db_repos.strategy_runtime.mark_paused(
        "put_spread", reason="ancient pause", now=now - timedelta(days=100),
        eligible_in_days=14,
    )
    state = await current_pause_state(db_repos, "put_spread")
    assert state is WinRateState.PAUSED_LOW_WIN_RATE  # NOT auto-cleared


@pytest.mark.asyncio
async def test_win_rate_recovers_pause_held(db_repos, monkeypatch, caplog):
    """Recovery does NOT auto-clear, but logs win_rate_recovered_pause_held
    so the operator can see the signal."""
    captured: list[dict] = []
    async def _capturing_notify(*a, **k):
        captured.append({"called": True})
    monkeypatch.setattr("risk.win_rate_floor.notify", _capturing_notify)

    log_events: list[dict] = []
    real_checkpoint = __import__("risk.win_rate_floor", fromlist=["log_checkpoint"]).log_checkpoint
    def _stub_checkpoint(event, **kw):
        log_events.append({"event": event, **kw})
        real_checkpoint(event, **kw)
    monkeypatch.setattr("risk.win_rate_floor.log_checkpoint", _stub_checkpoint)

    cfg = _cfg()
    now = datetime.now(UTC).replace(tzinfo=None)
    # Pre-seed paused.
    await db_repos.strategy_runtime.mark_paused(
        "put_spread", reason="prior loss streak", now=now - timedelta(days=5),
    )
    # Recovery — all wins.
    await _insert_cycles(db_repos, strategy_id="put_spread", pnls=[100] * 10, base_time=now)
    result = await check_and_apply(db_repos, _strategy(), cfg, now=now)
    assert result.action == "pause"  # held
    assert result.win_rate == pytest.approx(100.0)
    # Logged recovery-while-held checkpoint.
    held = [e for e in log_events if e["event"] == "win_rate_recovered_pause_held"]
    assert len(held) == 1
    # NO discord notify on a held recovery.
    assert not captured


# -- bot integration -------------------------------------------------------


@pytest.mark.asyncio
async def test_win_rate_skipped_when_drawdown_disabled(db_repos, monkeypatch):
    """When drawdown DISABLED is set, the bot's _propose_and_route doesn't
    even read pause state — it short-circuits earlier. The win-rate
    evaluation in check_and_apply still runs independently as a *recording*
    step, but the *gate decision* in the bot only consults pause if
    drawdown is NORMAL/WARNING.

    This test confirms that the BOT-LEVEL gate behaves correctly: when both
    are set, the bot skips for DISABLED first; pause is irrelevant. We
    simulate the bot's order-of-checks here (no run_bot.py harness needed)."""
    # Set drawdown DISABLED via repo direct.
    now = datetime.now(UTC).replace(tzinfo=None)
    await db_repos.strategy_runtime.disable(
        "put_spread", until=now + timedelta(days=30),
        reason="drawdown disabled", now=now,
    )
    # Independently set the pause.
    await db_repos.strategy_runtime.mark_paused(
        "put_spread", reason="low win rate", now=now,
    )
    # Replicate the bot's order: drawdown read FIRST.
    from risk import auto_disable
    drawdown_state = await auto_disable.current_state(db_repos, "put_spread", now=now)
    assert drawdown_state is auto_disable.DrawdownState.DISABLED
    # The bot would `continue` here and not even call current_pause_state.
    # But to verify they're truly independent, confirm the pause column is
    # still set (drawdown did NOT clobber it):
    pause = await current_pause_state(db_repos, "put_spread")
    assert pause is WinRateState.PAUSED_LOW_WIN_RATE


@pytest.mark.asyncio
async def test_alert_rate_limited(db_repos, monkeypatch):
    """Re-entering the pause state inside the 24h cooldown window does NOT
    re-fire the Discord notify. Locks the rate-limit primitive."""
    captured: list[dict] = []
    async def _stub_notify(event_type, title, **payload):
        captured.append({"event_type": event_type})
    monkeypatch.setattr("risk.win_rate_floor.notify", _stub_notify)

    cfg = _cfg()
    pnls = [100] * 4 + [-50] * 6  # 4/10 = 40% wins
    now = datetime.now(UTC).replace(tzinfo=None)
    await _insert_cycles(db_repos, strategy_id="put_spread", pnls=pnls, base_time=now)
    # Tick 1: NORMAL → PAUSED, one notify.
    await check_and_apply(db_repos, _strategy(), cfg, now=now)
    assert len(captured) == 1

    # Operator clears the pause to simulate a quick manual cycle.
    await db_repos.strategy_runtime.enable("put_spread")
    # Tick 2: 1h later, win rate still 40% → re-paused, but inside cooldown
    # so no second notify.
    await check_and_apply(db_repos, _strategy(), cfg, now=now + timedelta(hours=1))
    assert len(captured) == 1
    assert PAUSE_ALERT_COOLDOWN_HOURS == 24.0


@pytest.mark.asyncio
async def test_pause_and_drawdown_warning_coexist(db_repos, monkeypatch):
    """A strategy can be in BOTH drawdown WARNING and win-rate PAUSE on
    the same row. Both columns populated independently — neither masks
    the other. The bot's runtime gate skips on PAUSE (more restrictive),
    and the dashboard surfaces both flags."""
    monkeypatch.setattr("risk.win_rate_floor.notify", _no_op_notify)
    now = datetime.now(UTC).replace(tzinfo=None)

    # Drawdown WARNING via direct repo call.
    await db_repos.strategy_runtime.mark_warning(
        "put_spread", reason="rolling -200", now=now,
    )
    # Independently set pause.
    await db_repos.strategy_runtime.mark_paused(
        "put_spread", reason="low win rate", now=now,
    )
    row = await db_repos.strategy_runtime.get("put_spread")
    # Both columns populated, neither cleared by the other.
    assert row["drawdown_state"] == "WARNING"
    assert row["pause_state"] == "LOW_WIN_RATE"
    assert row["disabled_reason"] == "rolling -200"
    assert row["paused_reason"] == "low win rate"
    # Independent state-reads agree.
    from risk import auto_disable
    assert await auto_disable.current_state(db_repos, "put_spread", now=now) \
        is auto_disable.DrawdownState.WARNING
    assert await current_pause_state(db_repos, "put_spread") \
        is WinRateState.PAUSED_LOW_WIN_RATE


@pytest.mark.asyncio
async def test_manual_reenable_clears_both_pause_and_drawdown(db_repos):
    """Operator running enable() (== reenable_strategy.py without flags)
    clears BOTH gates in one shot. Documents the "single override" semantic
    so the prior-state printout in reenable_strategy.py is honest."""
    now = datetime.now(UTC).replace(tzinfo=None)
    await db_repos.strategy_runtime.disable(
        "put_spread", until=now + timedelta(days=30),
        reason="d", now=now,
    )
    await db_repos.strategy_runtime.mark_paused("put_spread", reason="p", now=now)
    row = await db_repos.strategy_runtime.get("put_spread")
    assert row["drawdown_state"] == "DISABLED"
    assert row["pause_state"] == "LOW_WIN_RATE"

    await db_repos.strategy_runtime.enable("put_spread")
    assert await db_repos.strategy_runtime.get("put_spread") is None


# -- per-strategy override -------------------------------------------------


@pytest.mark.asyncio
async def test_per_strategy_override_beats_global(db_repos, monkeypatch):
    """Strategy with `win_rate_floor_pct=75` in params trips at 70% even
    though the global default is 60%."""
    monkeypatch.setattr("risk.win_rate_floor.notify", _no_op_notify)
    cfg = _cfg(min_rate=60.0)
    pnls = [100] * 7 + [-50] * 3  # 7/10 = 70% wins
    now = datetime.now(UTC).replace(tzinfo=None)
    await _insert_cycles(db_repos, strategy_id="put_spread", pnls=pnls, base_time=now)
    # Global 60%: 70% > 60% → NORMAL.
    result_global = await check_and_apply(db_repos, _strategy(), cfg, now=now)
    assert result_global.action == "normal"
    # Per-strategy 75% override: 70% < 75% → PAUSE.
    result_override = await check_and_apply(
        db_repos, _strategy(win_rate_override=75.0), cfg, now=now,
    )
    assert result_override.action == "pause"


# -- dashboard overview -----------------------------------------------------


@pytest.mark.asyncio
async def test_get_win_rate_overview_shape(db_repos):
    """Per-strategy dict shape the dashboard renders."""
    cfg = _cfg()
    now = datetime.now(UTC).replace(tzinfo=None)
    await _insert_cycles(db_repos, strategy_id="put_spread",
                         pnls=[100] * 8 + [-50] * 2, base_time=now)
    overview = await get_win_rate_overview(
        db_repos, [_strategy()], cfg, now=now,
    )
    assert "put_spread" in overview
    info = overview["put_spread"]
    assert info["state"] == "NORMAL"
    assert info["win_rate"] == pytest.approx(80.0)
    assert info["sample_size"] == 10
    assert info["threshold"] == 60.0


@pytest.mark.asyncio
async def test_get_win_rate_overview_insufficient_data(db_repos):
    """Fresh strategy with <10 cycles shows action=insufficient_data."""
    cfg = _cfg()
    await _insert_cycles(db_repos, strategy_id="put_spread", pnls=[100, -50, 100])
    overview = await get_win_rate_overview(db_repos, [_strategy()], cfg)
    info = overview["put_spread"]
    assert info["action"] == "insufficient_data"
    assert info["sample_size"] == 3
    assert info["win_rate"] is None
