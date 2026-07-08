"""risk/manual_flags — the operator restore path out of MANUAL_INTERVENTION."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from core.models import Position, PositionState, StateLog, StateLogTrigger
from risk.manual_flags import latest_flag_entry, list_flagged, restore_position


def _utc() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


async def _seed_flagged(
    db_repos, *, symbol: str = "SOFI", strategy_id: str = "calendar",
    from_state: PositionState = PositionState.SPREAD_OPEN,
    flag_reason: str = "earnings_appeared_mid_cycle earnings=2026-07-26 short_exp=2026-07-24",
    write_log: bool = True,
) -> int:
    pos_id = await db_repos.positions.insert(
        Position(account_id="test", symbol=symbol, strategy_id=strategy_id,
                 state=PositionState.MANUAL_INTERVENTION, shares=0,
                 state_changed_at=_utc())
    )
    if write_log:
        await db_repos.state_log.insert(
            StateLog(position_id=pos_id, from_state=from_state,
                     to_state=PositionState.MANUAL_INTERVENTION,
                     reason=flag_reason, triggered_by=StateLogTrigger.STRATEGY,
                     created_at=_utc())
        )
    return pos_id


@pytest.mark.asyncio
async def test_restore_defaults_to_pre_flag_state(db_repos):
    pos_id = await _seed_flagged(db_repos)
    result = await restore_position(
        db_repos, "SOFI", "calendar", account_id="test", reason="test restore",
    )
    assert result["ok"] is True
    assert result["restored_to"] == "SPREAD_OPEN"
    pos = await db_repos.positions.get_by_symbol("test", "SOFI", strategy_id="calendar")
    assert pos.state == PositionState.SPREAD_OPEN
    # The restore writes its own audit row.
    entries = await db_repos.state_log.list_for_position(pos_id)
    newest = entries[0]
    assert newest.to_state == PositionState.SPREAD_OPEN
    assert newest.triggered_by == StateLogTrigger.MANUAL
    assert "restore_manual_intervention" in (newest.reason or "")


@pytest.mark.asyncio
async def test_restore_explicit_state_override(db_repos):
    await _seed_flagged(db_repos, from_state=PositionState.SPREAD_OPEN)
    result = await restore_position(
        db_repos, "SOFI", "calendar", account_id="test", state="idle",
    )
    assert result["ok"] is True
    assert result["restored_to"] == "IDLE"


@pytest.mark.asyncio
async def test_restore_refuses_unflagged_position(db_repos):
    await db_repos.positions.insert(
        Position(account_id="test", symbol="F", strategy_id="monthly_wheel",
                 state=PositionState.CSP_OPEN, shares=0, state_changed_at=_utc())
    )
    result = await restore_position(db_repos, "F", "monthly_wheel", account_id="test")
    assert result["ok"] is False
    assert "not MANUAL_INTERVENTION" in result["error"]


@pytest.mark.asyncio
async def test_restore_requires_state_without_flag_log(db_repos):
    await _seed_flagged(db_repos, write_log=False)
    result = await restore_position(db_repos, "SOFI", "calendar", account_id="test")
    assert result["ok"] is False
    assert "explicit state" in result["error"]


@pytest.mark.asyncio
async def test_restore_rejects_bogus_state(db_repos):
    await _seed_flagged(db_repos)
    result = await restore_position(
        db_repos, "SOFI", "calendar", account_id="test", state="NOT_A_STATE",
    )
    assert result["ok"] is False


@pytest.mark.asyncio
async def test_list_flagged_tags_earnings_provenance(db_repos):
    await _seed_flagged(db_repos)  # earnings-recheck flag
    await _seed_flagged(
        db_repos, symbol="F", strategy_id="monthly_wheel",
        from_state=PositionState.CSP_OPEN,
        flag_reason="partial fill then cancel — see PR#1 #7",
    )
    flagged = await list_flagged(db_repos, account_id="test")
    by_symbol = {f["symbol"]: f for f in flagged}
    assert by_symbol["SOFI"]["earnings_flag"] is True
    assert by_symbol["SOFI"]["from_state"] == "SPREAD_OPEN"
    assert by_symbol["F"]["earnings_flag"] is False


@pytest.mark.asyncio
async def test_latest_flag_entry_picks_newest_flag_row(db_repos):
    pos_id = await _seed_flagged(db_repos)
    # A later, second flag row (e.g. re-flagged after an earlier restore).
    await db_repos.state_log.insert(
        StateLog(position_id=pos_id, from_state=PositionState.SPREAD_OPEN,
                 to_state=PositionState.MANUAL_INTERVENTION,
                 reason="earnings_appeared_mid_cycle earnings=2026-08-01 short_exp=2026-07-30",
                 triggered_by=StateLogTrigger.STRATEGY, created_at=_utc())
    )
    entry = await latest_flag_entry(db_repos, pos_id)
    assert "2026-08-01" in (entry.reason or "")
