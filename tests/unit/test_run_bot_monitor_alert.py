"""scripts/run_bot._report_monitor_failure — a crashing risk monitor must
alert (rate-limited), not just log. Regression for the win-rate floor that
threw silently on every tick 2026-08-26 .. 2026-10-19."""

from __future__ import annotations

import pytest

import scripts.run_bot as run_bot


@pytest.mark.asyncio
async def test_monitor_failure_alerts_once_per_cooldown(db_repos, monkeypatch):
    sent: list[tuple[str, dict]] = []

    async def _capture(event_type, title, **payload):
        sent.append((event_type, payload))

    monkeypatch.setattr(run_bot, "notify", _capture)
    err = ValueError("1 validation error for WheelCycle")

    await run_bot._report_monitor_failure(db_repos, "win_rate_floor_check_failed", "put_spread", err)
    await run_bot._report_monitor_failure(db_repos, "win_rate_floor_check_failed", "put_spread", err)
    assert len(sent) == 1
    assert sent[0][0] == "risk.monitor_check_failed"
    assert sent[0][1]["strategy"] == "put_spread"
    assert "validation error" in sent[0][1]["error"]

    # Different strategy or different monitor → its own cooldown key.
    await run_bot._report_monitor_failure(db_repos, "win_rate_floor_check_failed", "monthly_wheel", err)
    await run_bot._report_monitor_failure(db_repos, "expectancy_floor_check_failed", "put_spread", err)
    assert len(sent) == 3


@pytest.mark.asyncio
async def test_monitor_failure_alert_errors_are_swallowed(db_repos, monkeypatch):
    async def _boom(*args, **kwargs):
        raise RuntimeError("discord down")

    monkeypatch.setattr(run_bot, "notify", _boom)
    # Must not raise — alerting can never break the trading loop.
    await run_bot._report_monitor_failure(
        db_repos, "auto_disable_check_failed", "put_spread", ValueError("x"),
    )
