"""execution/kill_switch — rules 8-10 of §8."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from core.models import CycleOutcome, WheelCycle
from execution.kill_switch import KillSwitch
from platforms.paper_broker import PaperBroker


def _config(stop_path: str | None = None, **risk_overrides) -> dict:
    cfg = {
        "account": {"id": "test", "broker": "paper"},
        "risk": {
            "daily_loss_kill_switch_pct": 5,
            "consecutive_losses_pause": 3,
        },
    }
    cfg["risk"].update(risk_overrides)
    if stop_path is not None:
        cfg["risk"]["stop_file_path"] = stop_path
    return cfg


@pytest.mark.asyncio
async def test_no_anchor_means_p_and_l_gate_skipped(db_repos):
    broker = PaperBroker(cash=10_000)
    ks = KillSwitch(broker, db_repos, _config())
    result = await ks.check(today=date(2025, 6, 1))
    assert result.tripped is False


@pytest.mark.asyncio
async def test_prime_session_records_session_open_equity(db_repos):
    broker = PaperBroker(cash=10_000)
    ks = KillSwitch(broker, db_repos, _config())
    entry = await ks.prime_session(today=date(2025, 6, 1))
    assert entry.session_open_equity == 10_000


@pytest.mark.asyncio
async def test_prime_session_idempotent(db_repos):
    broker = PaperBroker(cash=10_000)
    ks = KillSwitch(broker, db_repos, _config())
    await ks.prime_session(today=date(2025, 6, 1))
    # Manually drain the broker's cash so equity is now lower; prime should NOT
    # overwrite the anchor.
    broker._cash = 8_000
    entry = await ks.prime_session(today=date(2025, 6, 1))
    assert entry.session_open_equity == 10_000


@pytest.mark.asyncio
async def test_daily_loss_within_band_does_not_trip(db_repos):
    broker = PaperBroker(cash=10_000)
    ks = KillSwitch(broker, db_repos, _config())
    await ks.prime_session(today=date(2025, 6, 1))
    broker._cash = 9_700  # 3% drawdown < 5%
    result = await ks.check(today=date(2025, 6, 1))
    assert result.tripped is False


@pytest.mark.asyncio
async def test_daily_loss_past_threshold_trips(db_repos):
    broker = PaperBroker(cash=10_000)
    ks = KillSwitch(broker, db_repos, _config())
    await ks.prime_session(today=date(2025, 6, 1))
    broker._cash = 9_400  # 6% drawdown > 5%
    result = await ks.check(today=date(2025, 6, 1))
    assert result.tripped is True
    assert any("drawdown" in r for r in result.reasons)


@pytest.mark.asyncio
async def test_consecutive_losses_trips_at_threshold(db_repos):
    broker = PaperBroker(cash=10_000)
    base = datetime.now(UTC).replace(tzinfo=None)
    for i in range(3):
        cid = await db_repos.cycles.insert(
            WheelCycle(
                account_id="test",
                symbol=f"X{i}",
                started_at=base - timedelta(days=10 + i),
            )
        )
        await db_repos.cycles.update(
            cid,
            ended_at=(base - timedelta(days=i)).isoformat(),
            final_pnl=-50.0,
            cycle_outcome=CycleOutcome.MANUAL_CLOSE.value,
        )
    ks = KillSwitch(broker, db_repos, _config())
    result = await ks.check(today=date(2025, 6, 1))
    assert result.tripped is True
    assert any("consecutive" in r for r in result.reasons)


@pytest.mark.asyncio
async def test_winning_cycle_resets_consecutive_loss_streak(db_repos):
    broker = PaperBroker(cash=10_000)
    base = datetime.now(UTC).replace(tzinfo=None)
    # Two losses, then a win, then two more losses → streak from most-recent = 2.
    for i, pnl in enumerate([-50.0, -50.0, 25.0, -50.0, -50.0]):
        cid = await db_repos.cycles.insert(
            WheelCycle(
                account_id="test",
                symbol=f"X{i}",
                started_at=base - timedelta(days=20 - i),
            )
        )
        await db_repos.cycles.update(
            cid,
            ended_at=(base - timedelta(days=10 - i)).isoformat(),
            final_pnl=pnl,
            cycle_outcome=CycleOutcome.MANUAL_CLOSE.value,
        )
    ks = KillSwitch(broker, db_repos, _config(consecutive_losses_pause=3))
    result = await ks.check(today=date(2025, 6, 1))
    assert result.tripped is False


@pytest.mark.asyncio
async def test_stop_file_trips_kill_switch(db_repos, tmp_path):
    broker = PaperBroker(cash=10_000)
    stop = tmp_path / "STOP"
    stop.write_text("halt")
    ks = KillSwitch(broker, db_repos, _config(stop_path=str(stop)))
    await ks.prime_session(today=date(2025, 6, 1))
    result = await ks.check(today=date(2025, 6, 1))
    assert result.tripped is True
    assert any("manual stop" in r for r in result.reasons)
