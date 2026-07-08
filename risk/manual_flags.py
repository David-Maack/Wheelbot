"""Restore path for MANUAL_INTERVENTION positions (2026-07-08 sprint).

MANUAL_INTERVENTION is a one-way latch by design — the bot never auto-clears
it. Until now the only way OUT was raw SQL pasted into the container (twice
so far: SOFI narrow_put_spread 2026-06-23, SOFI calendar 2026-07-08). This
module is the proper operator path, used by both `scripts/restore_position.py`
and the MCP `unflag_position` control.

The restore target defaults to the `from_state` recorded in the state_log row
that set the flag — i.e. "put it back exactly where it was when the bot
flagged it" — so the operator doesn't have to know the state machine. An
explicit `state` argument overrides for the rare case the world changed while
flagged. Every restore writes its own state_log row (triggered_by=MANUAL).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from core.checkpoint import log_checkpoint
from core.models import PositionState, StateLog, StateLogTrigger
from db.repo import Repos

# The reason prefix _flag_manual (risk/earnings_recheck.py) writes — used to
# recognize earnings-recheck-originated flags in both the watchdog and the
# restore hint.
EARNINGS_FLAG_REASON_PREFIX = "earnings_appeared_mid_cycle"


def _iso(v: Any) -> str | None:
    """JSON-safe timestamp: DB round-trips can hand back str or datetime."""
    if v is None:
        return None
    return v.isoformat() if isinstance(v, datetime) else str(v)


def _state_str(v: Any) -> str | None:
    """JSON-safe state: str-Enum gotcha — rows may carry plain strings."""
    if v is None:
        return None
    return v.value if hasattr(v, "value") else str(v)


async def latest_flag_entry(repos: Repos, position_id: int) -> StateLog | None:
    """Newest state_log row that moved this position INTO MANUAL_INTERVENTION."""
    for entry in await repos.state_log.list_for_position(position_id, limit=100):
        if entry.to_state == PositionState.MANUAL_INTERVENTION:
            return entry
    return None


async def restore_position(
    repos: Repos,
    symbol: str,
    strategy_id: str,
    *,
    account_id: str = "primary",
    state: str | None = None,
    reason: str = "operator restore",
) -> dict[str, Any]:
    """Move a MANUAL_INTERVENTION position back to a managed state.

    Returns a JSON-safe result dict (also the MCP tool's response shape).
    Refuses when the position isn't flagged, or when no target state can be
    determined (no flag state_log row AND no explicit `state`).
    """
    symbol = symbol.upper()
    position = await repos.positions.get_by_symbol(
        account_id, symbol, strategy_id=strategy_id,
    )
    if position is None:
        return {"ok": False, "error": f"no {strategy_id} position on {symbol}"}
    if position.state != PositionState.MANUAL_INTERVENTION:
        return {
            "ok": False,
            "error": f"position is {position.state}, not MANUAL_INTERVENTION — nothing to restore",
        }
    assert position.id is not None

    flag_entry = await latest_flag_entry(repos, position.id)
    if state is not None:
        try:
            target = PositionState(state.upper())
        except ValueError:
            return {"ok": False, "error": f"unknown position state {state!r}"}
    elif flag_entry is not None and flag_entry.from_state is not None:
        # DB round-trip may hand back a plain str — coerce (str-Enum gotcha).
        target = PositionState(flag_entry.from_state)
    else:
        return {
            "ok": False,
            "error": "no flag state_log row to recover from_state — pass an explicit state",
        }
    if target == PositionState.MANUAL_INTERVENTION:
        return {"ok": False, "error": "target state is MANUAL_INTERVENTION — refusing no-op"}

    full_reason = f"restore_manual_intervention: {reason}"
    await repos.positions.update_state(position.id, target, full_reason)
    await repos.state_log.insert(
        StateLog(
            position_id=position.id,
            from_state=PositionState.MANUAL_INTERVENTION,
            to_state=target,
            reason=full_reason,
            triggered_by=StateLogTrigger.MANUAL,
            created_at=datetime.now(UTC).replace(tzinfo=None),
        )
    )
    log_checkpoint(
        "manual_flag_restored",
        status="ok",
        symbol=symbol,
        strategy=strategy_id,
        position_id=position.id,
        restored_to=target.value,
        flag_reason=(flag_entry.reason if flag_entry else None),
        reason=reason,
    )
    return {
        "ok": True,
        "symbol": symbol,
        "strategy_id": strategy_id,
        "restored_to": target.value,
        "flag_reason": flag_entry.reason if flag_entry else None,
        "flagged_at": _iso(flag_entry.created_at) if flag_entry else None,
    }


async def list_flagged(repos: Repos, *, account_id: str = "primary") -> list[dict[str, Any]]:
    """All MANUAL_INTERVENTION positions with their flag provenance — the
    operator's worklist (used by the CLI --list and the MCP tool docs)."""
    out: list[dict[str, Any]] = []
    for p in await repos.positions.list_active(account_id):
        if p.state != PositionState.MANUAL_INTERVENTION or p.id is None:
            continue
        entry = await latest_flag_entry(repos, p.id)
        out.append({
            "symbol": p.symbol,
            "strategy_id": p.strategy_id,
            "position_id": p.id,
            "from_state": _state_str(entry.from_state) if entry else None,
            "flag_reason": entry.reason if entry else None,
            "flagged_at": _iso(entry.created_at) if entry else None,
            "earnings_flag": bool(
                entry and (entry.reason or "").startswith(EARNINGS_FLAG_REASON_PREFIX)
            ),
        })
    return out
