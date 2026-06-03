"""risk/auto_disable — drawdown circuit breaker."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from core.models import CycleOutcome, WheelCycle
from core.strategies import StrategyDefinition
from risk.auto_disable import (
    AUTO_RESET_DAYS,
    ROLLING_WINDOW_DAYS,
    WARNING_ALERT_COOLDOWN_HOURS,
    DrawdownState,
    check_and_apply,
    compute_rolling_pnl,
    current_state,
    is_currently_disabled,
)


def _strategy(
    *,
    threshold: float | None = -300,
    warning: float | None = None,
) -> StrategyDefinition:
    params = {}
    if threshold is not None:
        params["auto_disable_drawdown_usd"] = threshold
    if warning is not None:
        params["drawdown_warning_usd"] = warning
    return StrategyDefinition(
        id="put_spread",
        display_name="Bull Put Spread",
        type="vertical_spread",
        enabled=True,
        max_concurrent=4,
        params=params,
    )


async def _insert_closed_cycle(
    db_repos, *,
    strategy_id: str,
    final_pnl: float,
    ended_at: datetime,
    symbol: str = "F",
) -> int:
    started_at = ended_at - timedelta(days=10)
    return await db_repos.cycles.insert(
        WheelCycle(
            account_id="test",
            symbol=symbol,
            strategy_id=strategy_id,
            started_at=started_at,
            ended_at=ended_at,
            final_pnl=final_pnl,
            cycle_outcome=CycleOutcome.SPREAD_CLOSED_PROFIT
            if final_pnl > 0 else CycleOutcome.SPREAD_MAX_LOSS,
        )
    )


# -- compute_rolling_pnl ----------------------------------------------------


@pytest.mark.asyncio
async def test_rolling_pnl_returns_zero_when_no_cycles(db_repos):
    now = datetime.now(UTC).replace(tzinfo=None)
    pnl = await compute_rolling_pnl(db_repos, "put_spread", account_id="test", now=now)
    assert pnl == 0.0


@pytest.mark.asyncio
async def test_rolling_pnl_sums_recent_closed_cycles(db_repos):
    now = datetime.now(UTC).replace(tzinfo=None)
    await _insert_closed_cycle(db_repos, strategy_id="put_spread",
                                final_pnl=50.0, ended_at=now - timedelta(days=1))
    await _insert_closed_cycle(db_repos, strategy_id="put_spread",
                                final_pnl=-150.0, ended_at=now - timedelta(days=3))
    await _insert_closed_cycle(db_repos, strategy_id="put_spread",
                                final_pnl=-100.0, ended_at=now - timedelta(days=5))
    pnl = await compute_rolling_pnl(db_repos, "put_spread", account_id="test", now=now)
    assert pnl == pytest.approx(-200.0)  # 50 - 150 - 100


@pytest.mark.asyncio
async def test_rolling_pnl_excludes_cycles_older_than_window(db_repos):
    now = datetime.now(UTC).replace(tzinfo=None)
    await _insert_closed_cycle(db_repos, strategy_id="put_spread",
                                final_pnl=-500.0,
                                ended_at=now - timedelta(days=ROLLING_WINDOW_DAYS + 1))
    pnl = await compute_rolling_pnl(db_repos, "put_spread", account_id="test", now=now)
    assert pnl == 0.0


@pytest.mark.asyncio
async def test_rolling_pnl_excludes_other_strategies(db_repos):
    now = datetime.now(UTC).replace(tzinfo=None)
    await _insert_closed_cycle(db_repos, strategy_id="monthly_wheel",
                                final_pnl=-9999.0, ended_at=now - timedelta(days=1))
    pnl = await compute_rolling_pnl(db_repos, "put_spread", account_id="test", now=now)
    assert pnl == 0.0  # other strategy ignored


# -- check_and_apply --------------------------------------------------------


@pytest.mark.asyncio
async def test_check_and_apply_disables_when_threshold_breached(db_repos, monkeypatch):
    captured: list[dict] = []
    async def _stub_notify(event_type, title, **payload):
        captured.append({"event_type": event_type, "title": title, **payload})
    monkeypatch.setattr("risk.auto_disable.notify", _stub_notify)

    now = datetime.now(UTC).replace(tzinfo=None)
    # -350 rolling > threshold -300 → trigger.
    await _insert_closed_cycle(db_repos, strategy_id="put_spread",
                                final_pnl=-350.0, ended_at=now - timedelta(days=2))

    cfg = {"account": {"id": "test"}}
    triggered = await check_and_apply(db_repos, _strategy(threshold=-300), cfg, now=now)
    assert triggered is DrawdownState.DISABLED
    disabled, reason = await is_currently_disabled(db_repos, "put_spread", now=now)
    assert disabled is True
    assert "rolling" in (reason or "")
    # Discord notify fired exactly once.
    assert len(captured) == 1
    assert captured[0]["event_type"] == "strategy_auto_disabled"


@pytest.mark.asyncio
async def test_check_and_apply_no_op_when_under_threshold(db_repos, monkeypatch):
    monkeypatch.setattr("risk.auto_disable.notify",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not notify")))
    now = datetime.now(UTC).replace(tzinfo=None)
    await _insert_closed_cycle(db_repos, strategy_id="put_spread",
                                final_pnl=-50.0, ended_at=now - timedelta(days=1))
    cfg = {"account": {"id": "test"}}
    triggered = await check_and_apply(db_repos, _strategy(threshold=-300), cfg, now=now)
    assert triggered is DrawdownState.NORMAL


@pytest.mark.asyncio
async def test_check_and_apply_skipped_when_no_threshold(db_repos, monkeypatch):
    monkeypatch.setattr("risk.auto_disable.notify",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not notify")))
    now = datetime.now(UTC).replace(tzinfo=None)
    # Big drawdown but feature disabled (threshold absent).
    await _insert_closed_cycle(db_repos, strategy_id="put_spread",
                                final_pnl=-9999.0, ended_at=now - timedelta(days=1))
    cfg = {"account": {"id": "test"}}
    triggered = await check_and_apply(db_repos, _strategy(threshold=None), cfg, now=now)
    assert triggered is DrawdownState.NORMAL


@pytest.mark.asyncio
async def test_check_and_apply_doesnt_renotify_when_already_disabled(db_repos, monkeypatch):
    call_count = 0
    async def _stub_notify(event_type, title, **payload):
        nonlocal call_count
        call_count += 1
    monkeypatch.setattr("risk.auto_disable.notify", _stub_notify)

    now = datetime.now(UTC).replace(tzinfo=None)
    await _insert_closed_cycle(db_repos, strategy_id="put_spread",
                                final_pnl=-500.0, ended_at=now - timedelta(days=1))
    cfg = {"account": {"id": "test"}}
    # First check: disables + notifies.
    await check_and_apply(db_repos, _strategy(threshold=-300), cfg, now=now)
    # Second check on the next tick: still disabled, must not re-notify.
    await check_and_apply(db_repos, _strategy(threshold=-300), cfg, now=now + timedelta(minutes=5))
    assert call_count == 1


# -- is_currently_disabled auto-reset ---------------------------------------


@pytest.mark.asyncio
async def test_is_currently_disabled_auto_clears_after_reset_window(db_repos):
    now = datetime.now(UTC).replace(tzinfo=None)
    # Disable as if from a month-old event.
    past = now - timedelta(days=AUTO_RESET_DAYS + 1)
    # Manually insert a stale disabled-until.
    await db_repos.strategy_runtime.disable(
        "put_spread", until=now - timedelta(days=1),
        reason="stale", now=past,
    )
    disabled, reason = await is_currently_disabled(db_repos, "put_spread", now=now)
    assert disabled is False  # auto-cleared
    assert reason is None
    # Confirm row was deleted.
    row = await db_repos.strategy_runtime.get("put_spread")
    assert row is None


# -- TICKET-008: tri-state drawdown machine ---------------------------------


@pytest.mark.asyncio
async def test_normal_state_no_action(db_repos, monkeypatch):
    """P&L above warning threshold → NORMAL, no notify, no state row."""
    monkeypatch.setattr(
        "risk.auto_disable.notify",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not notify")),
    )
    now = datetime.now(UTC).replace(tzinfo=None)
    await _insert_closed_cycle(db_repos, strategy_id="put_spread",
                                final_pnl=-50.0, ended_at=now - timedelta(days=1))
    cfg = {"account": {"id": "test"}}
    result = await check_and_apply(
        db_repos, _strategy(threshold=-300, warning=-150), cfg, now=now,
    )
    assert result is DrawdownState.NORMAL
    state = await current_state(db_repos, "put_spread", now=now)
    assert state is DrawdownState.NORMAL


@pytest.mark.asyncio
async def test_warning_state_triggers_at_threshold(db_repos, monkeypatch):
    """P&L crosses warning but not disable → WARNING + drawdown.warning fired."""
    captured: list[dict] = []
    async def _stub_notify(event_type, title, **payload):
        captured.append({"event_type": event_type, "title": title, **payload})
    monkeypatch.setattr("risk.auto_disable.notify", _stub_notify)

    now = datetime.now(UTC).replace(tzinfo=None)
    # -200: between -150 warning and -300 disable.
    await _insert_closed_cycle(db_repos, strategy_id="put_spread",
                                final_pnl=-200.0, ended_at=now - timedelta(days=1))
    cfg = {"account": {"id": "test"}}
    result = await check_and_apply(
        db_repos, _strategy(threshold=-300, warning=-150), cfg, now=now,
    )
    assert result is DrawdownState.WARNING
    state = await current_state(db_repos, "put_spread", now=now)
    assert state is DrawdownState.WARNING
    assert len(captured) == 1
    assert captured[0]["event_type"] == "drawdown.warning"
    assert captured[0]["size_multiplier"] == 0.5


@pytest.mark.asyncio
async def test_warning_halves_spread_quantity():
    """size_multiplier=0.5 → spread sizing scales the cap by half.

    Pure unit check on `_sizing_quantity` arithmetic — no broker / chain
    needed. Verifies the 0.5 propagates into `effective_cap`."""
    from types import SimpleNamespace

    from strategies.spreads import _sizing_quantity

    strat = StrategyDefinition(
        id="put_spread", display_name="m", type="vertical_spread",
        enabled=True, max_concurrent=1,
        params={"max_capital_per_spread_usd": 500.0},
    )
    candidate = SimpleNamespace(max_loss_per_spread=100.0)

    full = _sizing_quantity(strat, {}, candidate)
    halved = _sizing_quantity(strat, {}, candidate, size_multiplier=0.5)
    assert full == 5
    assert halved == 2  # 250 / 100 floored


@pytest.mark.asyncio
async def test_disable_state_blocks_all(db_repos, monkeypatch):
    """P&L past disable threshold → DISABLED + strategy_auto_disabled fired."""
    captured: list[dict] = []
    async def _stub_notify(event_type, title, **payload):
        captured.append({"event_type": event_type, "title": title, **payload})
    monkeypatch.setattr("risk.auto_disable.notify", _stub_notify)

    now = datetime.now(UTC).replace(tzinfo=None)
    await _insert_closed_cycle(db_repos, strategy_id="put_spread",
                                final_pnl=-400.0, ended_at=now - timedelta(days=1))
    cfg = {"account": {"id": "test"}}
    result = await check_and_apply(
        db_repos, _strategy(threshold=-300, warning=-150), cfg, now=now,
    )
    assert result is DrawdownState.DISABLED
    assert len(captured) == 1
    assert captured[0]["event_type"] == "strategy_auto_disabled"


@pytest.mark.asyncio
async def test_warning_clears_when_pnl_recovers(db_repos, monkeypatch):
    """WARNING → NORMAL when P&L crosses back above warning. Silent (no notify)."""
    notify_count = 0
    async def _stub_notify(event_type, title, **payload):
        nonlocal notify_count
        notify_count += 1
    monkeypatch.setattr("risk.auto_disable.notify", _stub_notify)

    now = datetime.now(UTC).replace(tzinfo=None)
    cfg = {"account": {"id": "test"}}
    # First: -200 → WARNING (1 notify)
    losing = await _insert_closed_cycle(
        db_repos, strategy_id="put_spread",
        final_pnl=-200.0, ended_at=now - timedelta(days=1),
    )
    await check_and_apply(
        db_repos, _strategy(threshold=-300, warning=-150), cfg, now=now,
    )
    assert notify_count == 1
    # Recover: insert offsetting profit so rolling P&L = +50.
    await _insert_closed_cycle(
        db_repos, strategy_id="put_spread", symbol="GLD",
        final_pnl=250.0, ended_at=now,
    )
    result = await check_and_apply(
        db_repos, _strategy(threshold=-300, warning=-150), cfg, now=now,
    )
    assert result is DrawdownState.NORMAL
    assert notify_count == 1  # NO additional notify on recovery
    # Confirm row cleared.
    assert await db_repos.strategy_runtime.get("put_spread") is None
    _ = losing  # quiet linter — written for chronology readability


@pytest.mark.asyncio
async def test_wheel_qty_one_warning_logs_unhalved(monkeypatch, caplog):
    """WARNING wheel iteration logs that it cannot halve qty=1."""
    import logging
    caplog.set_level(logging.INFO)

    captured: list[dict] = []
    def _stub_checkpoint(event, **kwargs):
        captured.append({"event": event, **kwargs})
    monkeypatch.setattr("strategies.wheel.log_checkpoint", _stub_checkpoint)

    from strategies.wheel import propose_all
    from core.strategies import StrategyDefinition

    strat = StrategyDefinition(
        id="monthly_wheel", display_name="m", type="wheel",
        enabled=True, max_concurrent=1, params={},
    )

    class _FakeBroker: pass
    class _FakeRepos:
        positions = type("P", (), {
            "get_by_symbol": staticmethod(lambda *a, **k: __import__('asyncio').sleep(0, result=None)),
            "list_active": staticmethod(lambda *a, **k: __import__('asyncio').sleep(0, result=[])),
        })()
        chain_snapshots = None

    universe = {"tickers": []}

    await propose_all(
        _FakeBroker(), _FakeRepos(), {"account": {"id": "t"}}, universe,
        ivr=None, strategy=strat, size_multiplier=0.5,
    )
    unhalved = [c for c in captured if c["event"] == "wheel_warning_size_unhalved"]
    assert len(unhalved) == 1
    assert unhalved[0]["size_multiplier"] == 0.5


@pytest.mark.asyncio
async def test_warning_to_disabled_fires_drawdown_tripped_not_warning(
    db_repos, monkeypatch,
):
    """WARNING → DISABLED: only `strategy_auto_disabled` fires (NOT a second
    `drawdown.warning`)."""
    captured: list[dict] = []
    async def _stub_notify(event_type, title, **payload):
        captured.append({"event_type": event_type, **payload})
    monkeypatch.setattr("risk.auto_disable.notify", _stub_notify)

    now = datetime.now(UTC).replace(tzinfo=None)
    cfg = {"account": {"id": "test"}}
    # Round 1: P&L -200 → WARNING
    cyc1 = await _insert_closed_cycle(
        db_repos, strategy_id="put_spread",
        final_pnl=-200.0, ended_at=now - timedelta(days=2),
    )
    await check_and_apply(
        db_repos, _strategy(threshold=-300, warning=-150), cfg, now=now,
    )
    assert [c["event_type"] for c in captured] == ["drawdown.warning"]
    # Round 2: deeper loss → DISABLED
    await _insert_closed_cycle(
        db_repos, strategy_id="put_spread", symbol="GLD",
        final_pnl=-200.0, ended_at=now - timedelta(days=1),
    )
    result = await check_and_apply(
        db_repos, _strategy(threshold=-300, warning=-150), cfg, now=now,
    )
    assert result is DrawdownState.DISABLED
    events = [c["event_type"] for c in captured]
    # Only the disable notify added; no second drawdown.warning.
    assert events == ["drawdown.warning", "strategy_auto_disabled"]
    _ = cyc1


@pytest.mark.asyncio
async def test_drawdown_warning_alert_rate_limited(db_repos, monkeypatch):
    """Second WARNING evaluation within cooldown does not re-fire Discord."""
    captured: list[dict] = []
    async def _stub_notify(event_type, title, **payload):
        captured.append({"event_type": event_type})
    monkeypatch.setattr("risk.auto_disable.notify", _stub_notify)

    now = datetime.now(UTC).replace(tzinfo=None)
    cfg = {"account": {"id": "test"}}
    await _insert_closed_cycle(db_repos, strategy_id="put_spread",
                                final_pnl=-200.0, ended_at=now - timedelta(days=1))
    # Tick 1: NORMAL → WARNING fires.
    await check_and_apply(
        db_repos, _strategy(threshold=-300, warning=-150), cfg, now=now,
    )
    # Tick 2: 10 minutes later, still WARNING — must NOT re-fire.
    # (WARNING_held branch skips notify entirely; rate-limit is the second
    # belt-and-braces for the NORMAL→WARNING re-entry case.)
    await check_and_apply(
        db_repos, _strategy(threshold=-300, warning=-150), cfg,
        now=now + timedelta(minutes=10),
    )
    assert len(captured) == 1
    # Tick 3: simulate a recovery+re-entry inside cooldown.
    await db_repos.strategy_runtime.enable("put_spread")  # clear WARNING row
    await check_and_apply(
        db_repos, _strategy(threshold=-300, warning=-150), cfg,
        now=now + timedelta(minutes=30),
    )
    # Re-entered WARNING, but inside 4h cooldown → still no second notify.
    assert len(captured) == 1
    assert WARNING_ALERT_COOLDOWN_HOURS == 4.0
