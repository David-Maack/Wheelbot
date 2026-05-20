"""risk/auto_disable — drawdown circuit breaker."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from core.models import CycleOutcome, WheelCycle
from core.strategies import StrategyDefinition
from risk.auto_disable import (
    AUTO_RESET_DAYS,
    ROLLING_WINDOW_DAYS,
    check_and_apply,
    compute_rolling_pnl,
    is_currently_disabled,
)


def _strategy(*, threshold: float | None = -300) -> StrategyDefinition:
    params = {}
    if threshold is not None:
        params["auto_disable_drawdown_usd"] = threshold
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
    assert triggered is True
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
    assert triggered is False


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
    assert triggered is False


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
