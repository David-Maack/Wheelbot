"""Async repository layer over SQLite (aiosqlite).

One repository class per table. Repositories own row → model conversion. The strategy and
execution layers MUST go through repositories — never raw SQL — so swapping SQLite for
Postgres later is mechanical.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite

from core.models import (
    Candidate,
    ChainSnapshot,
    DailyState,
    IvHistory,
    LlmDecision,
    Order,
    OrderType,
    Position,
    PositionState,
    RegimeSnapshot,
    StateLog,
    WatchlistEntry,
    WatchlistRun,
    WatchlistRunStatus,
    WheelCycle,
)

JSON_FIELDS_BY_TABLE: dict[str, tuple[str, ...]] = {
    "orders": ("raw_request", "raw_response"),
    "state_log": ("metadata",),
    "candidates": ("raw_llm_response",),
    "llm_decisions": ("context", "response"),
    "chain_snapshots": ("contracts",),
}


def _dump_json(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, default=str)


def _load_json(value: Any) -> Any:
    if value is None or value == "":
        return None
    if isinstance(value, (dict, list)):
        return value
    return json.loads(value)


def _row_to_dict(row: aiosqlite.Row, json_fields: tuple[str, ...] = ()) -> dict[str, Any]:
    d = dict(row)
    for f in json_fields:
        if f in d:
            d[f] = _load_json(d[f])
    return d


class Database:
    """Connection holder. One per process. Open with `connect()` or use as context manager.

    Concurrent-access policy (TICKET-003):
      The same SQLite file is touched by multiple processes — the bot loop, the
      screener cron, the regime cron, the daily-summary cron, and the dashboard
      readers. WAL is required (multiple readers + one writer concurrently), and
      a `busy_timeout` is required so a contending writer waits a few seconds
      instead of failing immediately with SQLITE_BUSY.

      Readers should use the default DEFERRED transaction mode (which aiosqlite
      gives via `async with conn.execute(...)`) — NEVER use `BEGIN IMMEDIATE`
      from a reader, which would block the writer. The single writer is the bot
      loop; everything else (screener, dashboard) reads.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> aiosqlite.Connection:
        if self._conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = await aiosqlite.connect(self.path)
            self._conn.row_factory = aiosqlite.Row
            # Order matters: foreign_keys + WAL first, then performance/contention pragmas.
            await self._conn.execute("PRAGMA foreign_keys = ON")
            await self._conn.execute("PRAGMA journal_mode = WAL")
            # synchronous=NORMAL is safe with WAL (durability still survives an OS
            # crash; only a power-loss-mid-commit risks losing the last txn). Trades
            # a small durability window for a large write-throughput win — fine for
            # an audit-trail DB where the broker is the source of truth.
            await self._conn.execute("PRAGMA synchronous = NORMAL")
            # busy_timeout in ms — how long any contending statement waits for the
            # writer to release before returning SQLITE_BUSY. 5s comfortably covers
            # the bot loop's slowest commits.
            await self._conn.execute("PRAGMA busy_timeout = 5000")
            await self._conn.commit()
        return self._conn

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database not connected; call connect() first")
        return self._conn

    async def __aenter__(self) -> Database:
        await self.connect()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        c = await self.connect()
        try:
            yield c
            await c.commit()
        except Exception:
            await c.rollback()
            raise


class _Repo:
    table: str = ""
    json_fields: tuple[str, ...] = ()

    def __init__(self, db: Database):
        self.db = db

    async def _fetch_one(self, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        c = await self.db.connect()
        async with c.execute(sql, params) as cur:
            row = await cur.fetchone()
        return _row_to_dict(row, self.json_fields) if row else None

    async def _fetch_all(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        c = await self.db.connect()
        async with c.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [_row_to_dict(r, self.json_fields) for r in rows]

    async def _insert(self, data: dict[str, Any]) -> int:
        cols = list(data.keys())
        placeholders = ", ".join(["?"] * len(cols))
        col_list = ", ".join(cols)
        sql = f"INSERT INTO {self.table} ({col_list}) VALUES ({placeholders})"
        c = await self.db.connect()
        async with c.execute(sql, tuple(data.values())) as cur:
            new_id = cur.lastrowid
        await c.commit()
        assert new_id is not None
        return new_id

    async def _update(self, row_id: int, data: dict[str, Any]) -> None:
        if not data:
            return
        assignments = ", ".join(f"{k} = ?" for k in data)
        sql = f"UPDATE {self.table} SET {assignments} WHERE id = ?"
        c = await self.db.connect()
        await c.execute(sql, (*data.values(), row_id))
        await c.commit()

    def _serialize(self, model: Any) -> dict[str, Any]:
        d: dict[str, Any] = model.model_dump(mode="json", exclude_none=True)
        d.pop("id", None)
        for f in self.json_fields:
            if f in d:
                d[f] = _dump_json(d[f])
        return d


class PositionsRepo(_Repo):
    table = "positions"

    async def get(self, position_id: int) -> Position | None:
        row = await self._fetch_one("SELECT * FROM positions WHERE id = ?", (position_id,))
        return Position(**row) if row else None

    async def get_by_symbol(
        self,
        account_id: str,
        symbol: str,
        strategy_id: str | None = None,
    ) -> Position | None:
        """Look up a position. When strategy_id is None, returns the first
        match for the symbol (legacy behavior — fine for single-strategy
        callers, but multi-strategy callers should pass strategy_id)."""
        if strategy_id is not None:
            row = await self._fetch_one(
                "SELECT * FROM positions WHERE account_id = ? AND symbol = ? AND strategy_id = ?",
                (account_id, symbol, strategy_id),
            )
        else:
            row = await self._fetch_one(
                "SELECT * FROM positions WHERE account_id = ? AND symbol = ?",
                (account_id, symbol),
            )
        return Position(**row) if row else None

    async def list_active(
        self,
        account_id: str,
        strategy_id: str | None = None,
    ) -> list[Position]:
        if strategy_id is not None:
            rows = await self._fetch_all(
                "SELECT * FROM positions WHERE account_id = ? AND strategy_id = ? "
                "AND state != ? ORDER BY symbol",
                (account_id, strategy_id, PositionState.IDLE.value),
            )
        else:
            rows = await self._fetch_all(
                "SELECT * FROM positions WHERE account_id = ? AND state != ? ORDER BY symbol",
                (account_id, PositionState.IDLE.value),
            )
        return [Position(**r) for r in rows]

    async def list_all(
        self,
        account_id: str,
        strategy_id: str | None = None,
    ) -> list[Position]:
        if strategy_id is not None:
            rows = await self._fetch_all(
                "SELECT * FROM positions WHERE account_id = ? AND strategy_id = ? ORDER BY symbol",
                (account_id, strategy_id),
            )
        else:
            rows = await self._fetch_all(
                "SELECT * FROM positions WHERE account_id = ? ORDER BY symbol",
                (account_id,),
            )
        return [Position(**r) for r in rows]

    async def insert(self, position: Position) -> int:
        return await self._insert(self._serialize(position))

    async def update_state(
        self,
        position_id: int,
        new_state: PositionState,
        reason: str | None,
        when: datetime | None = None,
    ) -> None:
        await self._update(
            position_id,
            {
                "state": new_state.value,
                "state_change_reason": reason,
                "state_changed_at": (when or datetime.utcnow()).isoformat(),
            },
        )

    async def update(self, position_id: int, **fields: Any) -> None:
        await self._update(position_id, fields)


class OrdersRepo(_Repo):
    table = "orders"
    json_fields = JSON_FIELDS_BY_TABLE["orders"]

    async def get(self, order_id: int) -> Order | None:
        row = await self._fetch_one("SELECT * FROM orders WHERE id = ?", (order_id,))
        return Order(**row) if row else None

    async def get_by_client_id(self, client_order_id: str) -> Order | None:
        row = await self._fetch_one(
            "SELECT * FROM orders WHERE client_order_id = ?", (client_order_id,)
        )
        return Order(**row) if row else None

    async def get_by_broker_id(self, broker_order_id: str) -> Order | None:
        row = await self._fetch_one(
            "SELECT * FROM orders WHERE broker_order_id = ?", (broker_order_id,)
        )
        return Order(**row) if row else None

    async def list_pending(self, account_id: str) -> list[Order]:
        rows = await self._fetch_all(
            "SELECT * FROM orders WHERE account_id = ? AND status IN ('PENDING','PARTIAL') "
            "ORDER BY placed_at",
            (account_id,),
        )
        return [Order(**r) for r in rows]

    async def oldest_pending_placed_at(self, account_id: str) -> datetime | None:
        """Earliest placed_at among PENDING/PARTIAL orders for this account.

        Used by Reconciler to keep the orders cursor from advancing past
        in-flight orders so subsequent get_orders_since() calls still see them
        when the broker eventually flips them to FILLED.
        """
        c = await self.db.connect()
        async with c.execute(
            "SELECT MIN(placed_at) AS placed_at FROM orders "
            "WHERE account_id = ? AND status IN ('PENDING','PARTIAL')",
            (account_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None or row["placed_at"] is None:
            return None
        return datetime.fromisoformat(row["placed_at"])

    async def list_recent(self, account_id: str, limit: int = 50) -> list[Order]:
        rows = await self._fetch_all(
            "SELECT * FROM orders WHERE account_id = ? ORDER BY placed_at DESC LIMIT ?",
            (account_id, limit),
        )
        return [Order(**r) for r in rows]

    async def last_close_trigger_for_position(
        self, position_id: int
    ) -> str | None:
        """Most recent close-side order's `trigger_reason` for this position's
        current cycle. Returns None when the position has no current cycle or
        the cycle has no close orders yet (or the close had no trigger_reason
        — pre-TICKET-005 orders won't).

        Used by the dashboard /positions table to show WHY the most recent
        close fired (profit / time / stop_loss / delta_stop_close / ...).
        """
        c = await self.db.connect()
        async with c.execute(
            "SELECT trigger_reason FROM orders WHERE cycle_id = ("
            "SELECT current_cycle_id FROM positions WHERE id = ?"
            ") AND order_type IN (?, ?) "
            "ORDER BY placed_at DESC LIMIT 1",
            (
                position_id,
                OrderType.BUY_TO_CLOSE.value,
                OrderType.MULTI_LEG_CLOSE.value,
            ),
        ) as cur:
            row = await cur.fetchone()
        if row is None or row["trigger_reason"] is None:
            return None
        return str(row["trigger_reason"])

    async def insert(self, order: Order) -> int:
        return await self._insert(self._serialize(order))

    async def update(self, order_id: int, **fields: Any) -> None:
        for f in self.json_fields:
            if f in fields:
                fields[f] = _dump_json(fields[f])
        await self._update(order_id, fields)


class WheelCyclesRepo(_Repo):
    table = "wheel_cycles"

    async def get(self, cycle_id: int) -> WheelCycle | None:
        row = await self._fetch_one("SELECT * FROM wheel_cycles WHERE id = ?", (cycle_id,))
        return WheelCycle(**row) if row else None

    async def list_open(self, account_id: str) -> list[WheelCycle]:
        rows = await self._fetch_all(
            "SELECT * FROM wheel_cycles WHERE account_id = ? AND ended_at IS NULL "
            "ORDER BY started_at",
            (account_id,),
        )
        return [WheelCycle(**r) for r in rows]

    async def list_closed(
        self, account_id: str, limit: int = 100
    ) -> list[WheelCycle]:
        rows = await self._fetch_all(
            "SELECT * FROM wheel_cycles WHERE account_id = ? AND ended_at IS NOT NULL "
            "ORDER BY ended_at DESC LIMIT ?",
            (account_id, limit),
        )
        return [WheelCycle(**r) for r in rows]

    async def list_closed_for_strategy(
        self, account_id: str, strategy_id: str, limit: int = 10,
    ) -> list[WheelCycle]:
        """TICKET-009: most-recent closed cycles for a specific strategy,
        newest first. Used by `risk.win_rate_floor.evaluate` for the
        last-N-cycles win-rate computation. `final_pnl IS NOT NULL` filter
        matches the drawdown breaker's `compute_rolling_pnl` filter — open
        cycles and partially-recorded closes are excluded so unrealized
        moves don't bias the rate."""
        rows = await self._fetch_all(
            "SELECT * FROM wheel_cycles "
            "WHERE account_id = ? AND strategy_id = ? "
            "  AND ended_at IS NOT NULL AND final_pnl IS NOT NULL "
            "ORDER BY ended_at DESC LIMIT ?",
            (account_id, strategy_id, limit),
        )
        return [WheelCycle(**r) for r in rows]

    async def insert(self, cycle: WheelCycle) -> int:
        return await self._insert(self._serialize(cycle))

    async def update(self, cycle_id: int, **fields: Any) -> None:
        await self._update(cycle_id, fields)


class StateLogRepo(_Repo):
    table = "state_log"
    json_fields = JSON_FIELDS_BY_TABLE["state_log"]

    async def insert(self, entry: StateLog) -> int:
        return await self._insert(self._serialize(entry))

    async def list_for_position(
        self, position_id: int, limit: int = 100
    ) -> list[StateLog]:
        rows = await self._fetch_all(
            "SELECT * FROM state_log WHERE position_id = ? ORDER BY created_at DESC LIMIT ?",
            (position_id, limit),
        )
        return [StateLog(**r) for r in rows]


class CandidatesRepo(_Repo):
    table = "candidates"
    json_fields = JSON_FIELDS_BY_TABLE["candidates"]

    async def insert(self, candidate: Candidate) -> int:
        return await self._insert(self._serialize(candidate))

    async def list_for_date(self, run_date: date) -> list[Candidate]:
        rows = await self._fetch_all(
            "SELECT * FROM candidates WHERE run_date = ? ORDER BY rank ASC NULLS LAST",
            (run_date.isoformat(),),
        )
        return [Candidate(**r) for r in rows]

    async def latest(self, limit: int = 50) -> list[Candidate]:
        rows = await self._fetch_all(
            "SELECT * FROM candidates ORDER BY run_date DESC, rank ASC LIMIT ?",
            (limit,),
        )
        return [Candidate(**r) for r in rows]

    async def mark_acted(self, candidate_id: int) -> None:
        await self._update(candidate_id, {"acted_on": True})


class RegimeSnapshotsRepo(_Repo):
    table = "regime_snapshots"

    async def insert(self, snapshot: RegimeSnapshot) -> int:
        return await self._insert(self._serialize(snapshot))

    async def latest(self) -> RegimeSnapshot | None:
        row = await self._fetch_one(
            "SELECT * FROM regime_snapshots ORDER BY snapshot_date DESC LIMIT 1"
        )
        return RegimeSnapshot(**row) if row else None

    async def get_for_date(self, snapshot_date: date) -> RegimeSnapshot | None:
        row = await self._fetch_one(
            "SELECT * FROM regime_snapshots WHERE snapshot_date = ?",
            (snapshot_date.isoformat(),),
        )
        return RegimeSnapshot(**row) if row else None


class LlmDecisionsRepo(_Repo):
    table = "llm_decisions"
    json_fields = JSON_FIELDS_BY_TABLE["llm_decisions"]

    async def insert(self, decision: LlmDecision) -> int:
        return await self._insert(self._serialize(decision))

    async def list_recent(self, decision_type: str | None = None, limit: int = 100) -> list[LlmDecision]:
        if decision_type:
            rows = await self._fetch_all(
                "SELECT * FROM llm_decisions WHERE decision_type = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (decision_type, limit),
            )
        else:
            rows = await self._fetch_all(
                "SELECT * FROM llm_decisions ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
        return [LlmDecision(**r) for r in rows]

    async def set_outcome(self, decision_id: int, outcome: str) -> None:
        await self._update(decision_id, {"outcome": outcome})

    async def total_cost_today(self) -> float:
        c = await self.db.connect()
        async with c.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM llm_decisions "
            "WHERE date(created_at) = date('now')"
        ) as cur:
            row = await cur.fetchone()
        return float(row[0]) if row else 0.0


class IvHistoryRepo(_Repo):
    table = "iv_history"

    async def upsert(self, entry: IvHistory) -> None:
        c = await self.db.connect()
        await c.execute(
            "INSERT INTO iv_history (symbol, snapshot_date, iv_30d) VALUES (?, ?, ?) "
            "ON CONFLICT(symbol, snapshot_date) DO UPDATE SET iv_30d = excluded.iv_30d",
            (entry.symbol, entry.snapshot_date.isoformat(), entry.iv_30d),
        )
        await c.commit()

    async def history_for(self, symbol: str, days: int = 365) -> list[IvHistory]:
        rows = await self._fetch_all(
            "SELECT * FROM iv_history WHERE symbol = ? "
            "AND snapshot_date >= date('now', ? || ' days') "
            "ORDER BY snapshot_date",
            (symbol, f"-{days}"),
        )
        return [IvHistory(**r) for r in rows]


class DailyStateRepo(_Repo):
    table = "daily_state"

    async def get(self, account_id: str, snapshot_date: date) -> DailyState | None:
        row = await self._fetch_one(
            "SELECT * FROM daily_state WHERE account_id = ? AND snapshot_date = ?",
            (account_id, snapshot_date.isoformat()),
        )
        return DailyState(**row) if row else None

    async def upsert(self, entry: DailyState) -> None:
        c = await self.db.connect()
        await c.execute(
            "INSERT INTO daily_state "
            "(account_id, snapshot_date, session_open_equity, consecutive_losses, "
            " kill_switch_armed, kill_switch_reason) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(account_id, snapshot_date) DO UPDATE SET "
            " session_open_equity = COALESCE(excluded.session_open_equity, daily_state.session_open_equity), "
            " consecutive_losses = excluded.consecutive_losses, "
            " kill_switch_armed = excluded.kill_switch_armed, "
            " kill_switch_reason = excluded.kill_switch_reason",
            (
                entry.account_id,
                entry.snapshot_date.isoformat(),
                entry.session_open_equity,
                entry.consecutive_losses,
                int(entry.kill_switch_armed),
                entry.kill_switch_reason,
            ),
        )
        await c.commit()


class ChainSnapshotsRepo(_Repo):
    table = "chain_snapshots"
    json_fields = JSON_FIELDS_BY_TABLE["chain_snapshots"]

    async def insert(self, snapshot: ChainSnapshot) -> int:
        return await self._insert(self._serialize(snapshot))

    async def get(self, snapshot_id: int) -> ChainSnapshot | None:
        row = await self._fetch_one("SELECT * FROM chain_snapshots WHERE id = ?", (snapshot_id,))
        return ChainSnapshot(**row) if row else None

    async def for_cycle(self, cycle_id: int) -> list[ChainSnapshot]:
        rows = await self._fetch_all(
            "SELECT * FROM chain_snapshots WHERE cycle_id = ? ORDER BY captured_at",
            (cycle_id,),
        )
        return [ChainSnapshot(**r) for r in rows]

    async def latest_for(self, symbol: str, side: str) -> ChainSnapshot | None:
        row = await self._fetch_one(
            "SELECT * FROM chain_snapshots WHERE symbol = ? AND side = ? "
            "ORDER BY captured_at DESC LIMIT 1",
            (symbol.upper(), side),
        )
        return ChainSnapshot(**row) if row else None


class StrategyRuntimeStateRepo(_Repo):
    """Per-strategy runtime state — auto-disable from drawdown circuit breaker.

    Disabled when `disabled_until` is set and in the future. `is_disabled()`
    auto-clears stale entries (past disabled_until) and returns the cleaned-up
    truth, so callers never see "disabled forever after timeout" state.
    """

    table = "strategy_runtime_state"

    async def get(self, strategy_id: str) -> dict | None:
        return await self._fetch_one(
            "SELECT * FROM strategy_runtime_state WHERE strategy_id = ?",
            (strategy_id,),
        )

    async def disable(
        self,
        strategy_id: str,
        *,
        until: datetime,
        reason: str,
        now: datetime | None = None,
    ) -> None:
        now = now or datetime.now(UTC).replace(tzinfo=None)
        c = await self.db.connect()
        await c.execute(
            "INSERT INTO strategy_runtime_state "
            "(strategy_id, disabled_at, disabled_until, disabled_reason, drawdown_state) "
            "VALUES (?, ?, ?, ?, 'DISABLED') "
            "ON CONFLICT(strategy_id) DO UPDATE SET "
            "  disabled_at = excluded.disabled_at, "
            "  disabled_until = excluded.disabled_until, "
            "  disabled_reason = excluded.disabled_reason, "
            "  drawdown_state = excluded.drawdown_state",
            (strategy_id, now.isoformat(), until.isoformat(), reason),
        )
        await c.commit()

    async def mark_warning(
        self,
        strategy_id: str,
        *,
        reason: str,
        now: datetime | None = None,
    ) -> None:
        """TICKET-008: persist a WARNING-tier row. No disabled_until — the
        state persists until the next check_and_apply observes recovery."""
        now = now or datetime.now(UTC).replace(tzinfo=None)
        c = await self.db.connect()
        await c.execute(
            "INSERT INTO strategy_runtime_state "
            "(strategy_id, disabled_at, disabled_until, disabled_reason, drawdown_state) "
            "VALUES (?, ?, NULL, ?, 'WARNING') "
            "ON CONFLICT(strategy_id) DO UPDATE SET "
            "  disabled_at = excluded.disabled_at, "
            "  disabled_until = NULL, "
            "  disabled_reason = excluded.disabled_reason, "
            "  drawdown_state = 'WARNING'",
            (strategy_id, now.isoformat(), reason),
        )
        await c.commit()

    async def set_drawdown_state(self, strategy_id: str, state: str) -> None:
        """Update just the drawdown_state column. Used after disable() to
        record the explicit DISABLED tag for the dashboard query."""
        c = await self.db.connect()
        await c.execute(
            "UPDATE strategy_runtime_state SET drawdown_state = ? WHERE strategy_id = ?",
            (state, strategy_id),
        )
        await c.commit()

    async def enable(self, strategy_id: str) -> None:
        """Manual re-enable. Clears the runtime state row entirely — BOTH the
        drawdown gate (DISABLED/WARNING) AND the pause gate (LOW_WIN_RATE)
        in one shot.

        This is the operator-override semantic used by
        scripts/reenable_strategy.py: a single manual action clears
        everything. The script prints the prior state before clearing so the
        operator sees exactly what they're undoing.

        For pause-only or drawdown-only clears, use `clear_pause` /
        `enable_drawdown_only` (not yet implemented — not needed at v1
        because the script always clears the whole row).
        """
        c = await self.db.connect()
        await c.execute(
            "DELETE FROM strategy_runtime_state WHERE strategy_id = ?",
            (strategy_id,),
        )
        await c.commit()

    async def mark_paused(
        self,
        strategy_id: str,
        *,
        reason: str,
        now: datetime | None = None,
        eligible_in_days: int = 14,
        state: str = "LOW_WIN_RATE",
    ) -> None:
        """TICKET-009: persist a pause. Independent of drawdown
        columns — if a row already exists (e.g. drawdown WARNING was set
        first), only the pause_* columns are touched. The disabled_* columns
        and drawdown_state column are NOT modified.

        `state` is the pause_state literal: 'LOW_WIN_RATE' (win-rate floor,
        TICKET-009) or 'NEGATIVE_EXPECTANCY' (expectancy floor, 2026-07-06).
        The floors guard against overwriting each other's pause BEFORE
        calling this — this method last-writer-wins on the pause_* columns.

        paused_reenable_eligible_at is advisory (cached `paused_at +
        eligible_in_days`) — the bot does NOT auto-clear when this passes.
        Operator must manually reenable.
        """
        now = now or datetime.now(UTC).replace(tzinfo=None)
        eligible_at = now + timedelta(days=eligible_in_days)
        c = await self.db.connect()
        await c.execute(
            "INSERT INTO strategy_runtime_state "
            "(strategy_id, pause_state, paused_at, paused_reason, "
            " paused_reenable_eligible_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(strategy_id) DO UPDATE SET "
            "  pause_state = excluded.pause_state, "
            "  paused_at = excluded.paused_at, "
            "  paused_reason = excluded.paused_reason, "
            "  paused_reenable_eligible_at = excluded.paused_reenable_eligible_at",
            (strategy_id, state, now.isoformat(), reason, eligible_at.isoformat()),
        )
        await c.commit()

    async def clear_pause(self, strategy_id: str) -> None:
        """TICKET-009: clear only the pause columns. Leaves drawdown_state /
        disabled_until / disabled_reason intact. Not used by the bot
        directly (recovery doesn't auto-clear per locked decision); reserved
        for a future scripts/reenable_strategy.py --pause-only flag if we
        ever want finer-grained operator control. v1 always uses enable().
        """
        c = await self.db.connect()
        await c.execute(
            "UPDATE strategy_runtime_state SET "
            "  pause_state = NULL, "
            "  paused_at = NULL, "
            "  paused_reason = NULL, "
            "  paused_reenable_eligible_at = NULL "
            "WHERE strategy_id = ?",
            (strategy_id,),
        )
        # If after the clear the row has no remaining state at all, drop it.
        await c.execute(
            "DELETE FROM strategy_runtime_state WHERE strategy_id = ? "
            "AND disabled_until IS NULL AND drawdown_state IS NULL "
            "AND pause_state IS NULL",
            (strategy_id,),
        )
        await c.commit()

    async def list_paused(self) -> list[dict]:
        """TICKET-009: rows currently paused (any pause_state literal —
        LOW_WIN_RATE or NEGATIVE_EXPECTANCY). Independent of list_disabled
        (which only returns rows with an active disabled_until). A strategy
        in BOTH drawdown DISABLED and a pause appears in both lists — by
        design, so reenable_strategy.py --list surfaces every active gate."""
        return await self._fetch_all(
            "SELECT * FROM strategy_runtime_state WHERE pause_state IS NOT NULL",
        )

    async def is_disabled(
        self,
        strategy_id: str,
        *,
        now: datetime | None = None,
    ) -> tuple[bool, str | None]:
        """Returns (is_disabled, reason). Auto-clears stale disable records."""
        now = now or datetime.now(UTC).replace(tzinfo=None)
        row = await self.get(strategy_id)
        if row is None or row.get("disabled_until") is None:
            return False, None
        until = datetime.fromisoformat(row["disabled_until"])
        if until <= now:
            await self.enable(strategy_id)  # auto-clear
            return False, None
        return True, row.get("disabled_reason")

    async def list_disabled(
        self, *, now: datetime | None = None,
    ) -> list[dict]:
        """All currently-disabled strategies (auto-clears stale rows)."""
        now = now or datetime.now(UTC).replace(tzinfo=None)
        rows = await self._fetch_all("SELECT * FROM strategy_runtime_state")
        out: list[dict] = []
        for r in rows:
            until_raw = r.get("disabled_until")
            if until_raw is None:
                continue
            until = datetime.fromisoformat(until_raw)
            if until <= now:
                await self.enable(r["strategy_id"])
                continue
            out.append(r)
        return out


class MacroEventsRepo(_Repo):
    """Macro economic events used by the blackout rule (TICKET-007).

    Populated by `scripts.refresh_macro_calendar` (daily cron); consumed by
    `risk.limits._rule_macro_blackout` and the `/macro` dashboard page.
    Idempotent inserts via the UNIQUE(event_date, event_type) constraint.
    """

    table = "macro_events"

    async def upsert_many(self, events: list) -> tuple[int, int]:
        """Upsert a batch. Returns `(upsert_attempts, distinct_rows_after)`.

        The two diverge when several input items share an `(event_date,
        event_type)` tuple — UNIQUE collapses them to one row, last-writer
        wins on description. Finnhub returns 4 separate items per FOMC day
        (Statement / Press Conference / Minutes / Economic Projections) and
        ~3 per CPI day, so ~50 attempts commonly produce ~18 distinct rows.
        Returning both makes the refresh log honest:
            macro_refresh_done upsert_attempts=50 distinct_rows_after=18
        """
        if not events:
            return 0, await self.count()
        c = await self.db.connect()
        attempts = 0
        for evt in events:
            await c.execute(
                "INSERT INTO macro_events "
                "(event_date, event_type, impact, description, fetched_at, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(event_date, event_type) DO UPDATE SET "
                "  impact = excluded.impact, "
                "  description = excluded.description, "
                "  fetched_at = excluded.fetched_at",
                (
                    evt.event_date.isoformat(),
                    evt.event_type,
                    evt.impact,
                    evt.description,
                    evt.fetched_at.isoformat(),
                    evt.created_at.isoformat(),
                ),
            )
            attempts += 1
        await c.commit()
        return attempts, await self.count()

    async def between(self, start: date, end: date) -> list:
        """All events with event_date in [start, end] inclusive, sorted ascending."""
        from core.models import MacroEvent
        rows = await self._fetch_all(
            "SELECT * FROM macro_events WHERE event_date >= ? AND event_date <= ? "
            "ORDER BY event_date, event_type",
            (start.isoformat(), end.isoformat()),
        )
        return [MacroEvent(**r) for r in rows]

    async def count(self) -> int:
        c = await self.db.connect()
        async with c.execute("SELECT COUNT(*) AS n FROM macro_events") as cur:
            row = await cur.fetchone()
        return int(row["n"]) if row else 0

    async def last_fetched_at(self) -> datetime | None:
        """Most-recent fetched_at across the table — used by is_stale()."""
        c = await self.db.connect()
        async with c.execute(
            "SELECT MAX(fetched_at) AS last FROM macro_events",
        ) as cur:
            row = await cur.fetchone()
        if row is None or row["last"] is None:
            return None
        return datetime.fromisoformat(row["last"])


class AlertRateLimitsRepo(_Repo):
    """Reusable rate-limit primitive for Discord/notify alerts (TICKET-007).

    Usage:
        if await repos.alert_rate_limits.try_fire("macro_calendar_stale", 20.0):
            await notify(...)

    Persists across process restarts so a flaky cron can't silence itself by
    restarting; persists across container rebuilds (data dir is bind-mounted)
    so deploy churn doesn't reset.
    """

    table = "alert_rate_limits"

    async def try_fire(self, alert_key: str, cooldown_hours: float) -> bool:
        """Returns True iff the alert hasn't fired within `cooldown_hours`,
        and atomically records the new firing timestamp. False means the
        alert is rate-limited — caller should not notify."""
        now = datetime.now(UTC).replace(tzinfo=None)
        c = await self.db.connect()
        async with c.execute(
            "SELECT last_fired_at FROM alert_rate_limits WHERE alert_key = ?",
            (alert_key,),
        ) as cur:
            row = await cur.fetchone()
        if row is not None:
            last = datetime.fromisoformat(row["last_fired_at"])
            if (now - last) < timedelta(hours=cooldown_hours):
                return False
        await c.execute(
            "INSERT INTO alert_rate_limits (alert_key, last_fired_at) VALUES (?, ?) "
            "ON CONFLICT(alert_key) DO UPDATE SET last_fired_at = excluded.last_fired_at",
            (alert_key, now.isoformat()),
        )
        await c.commit()
        return True

    async def last_fired(self, alert_key: str) -> datetime | None:
        c = await self.db.connect()
        async with c.execute(
            "SELECT last_fired_at FROM alert_rate_limits WHERE alert_key = ?",
            (alert_key,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return datetime.fromisoformat(row["last_fired_at"])


class BrokerParityLogRepo(_Repo):
    """TICKET-022: per-contract pricing-parity rows.

    Writer: scripts/parity_run.py (6x/day cron during NYSE hours).
    Readers: dashboard /parity view + daily markdown report generator.

    contract_symbol is canonical OCC dense form on both sides — both
    feeds normalize through scripts.parity_run._normalize_occ before
    insert, so the join between brokers is deterministic.
    """

    table = "broker_parity_log"

    async def insert_many(self, rows: list[dict[str, Any]]) -> int:
        """Bulk insert. Returns count of rows inserted."""
        if not rows:
            return 0
        c = await self.db.connect()
        await c.executemany(
            "INSERT INTO broker_parity_log "
            "(fetched_at, symbol, contract_symbol, alpaca_mid, tasty_mid, "
            " mid_diff_pct, alpaca_bid, alpaca_ask, tasty_bid, tasty_ask, "
            " asymmetric_liquidity) "
            "VALUES (:fetched_at, :symbol, :contract_symbol, :alpaca_mid, "
            " :tasty_mid, :mid_diff_pct, :alpaca_bid, :alpaca_ask, :tasty_bid, "
            " :tasty_ask, :asymmetric_liquidity)",
            rows,
        )
        await c.commit()
        return len(rows)

    async def stats_by_symbol(
        self, since: datetime, *, until: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Per-symbol summary over [since, until]: sample count, avg / worst
        mid_diff_pct, asymmetric-liquidity count.

        Worst is absolute value (operator cares about magnitude regardless of
        direction). Asymmetric counts only rows flagged at insert time."""
        until_str = until.isoformat() if until is not None else datetime.now(UTC).replace(tzinfo=None).isoformat()
        c = await self.db.connect()
        async with c.execute(
            "SELECT symbol, "
            "       COUNT(*) AS samples, "
            "       AVG(ABS(mid_diff_pct)) AS avg_abs_diff_pct, "
            "       MAX(ABS(mid_diff_pct)) AS worst_abs_diff_pct, "
            "       SUM(asymmetric_liquidity) AS asymmetric_count "
            "FROM broker_parity_log "
            "WHERE fetched_at >= ? AND fetched_at <= ? AND mid_diff_pct IS NOT NULL "
            "GROUP BY symbol "
            "ORDER BY worst_abs_diff_pct DESC",
            (since.isoformat(), until_str),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def daily_trend(
        self, since: datetime, *, until: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Mean-of-means per day across all symbols in [since, until].
        Drives the /parity trend chart."""
        until_str = until.isoformat() if until is not None else datetime.now(UTC).replace(tzinfo=None).isoformat()
        c = await self.db.connect()
        async with c.execute(
            "SELECT DATE(fetched_at) AS day, "
            "       AVG(ABS(mid_diff_pct)) AS avg_abs_diff_pct, "
            "       COUNT(*) AS samples "
            "FROM broker_parity_log "
            "WHERE fetched_at >= ? AND fetched_at <= ? AND mid_diff_pct IS NOT NULL "
            "GROUP BY DATE(fetched_at) "
            "ORDER BY day",
            (since.isoformat(), until_str),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def top_outliers(
        self, since: datetime, *, limit: int = 10, until: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Largest absolute mid_diff_pct rows in window. Used for the
        daily report's worst-N section."""
        until_str = until.isoformat() if until is not None else datetime.now(UTC).replace(tzinfo=None).isoformat()
        c = await self.db.connect()
        async with c.execute(
            "SELECT fetched_at, symbol, contract_symbol, alpaca_mid, tasty_mid, mid_diff_pct "
            "FROM broker_parity_log "
            "WHERE fetched_at >= ? AND fetched_at <= ? AND mid_diff_pct IS NOT NULL "
            "ORDER BY ABS(mid_diff_pct) DESC LIMIT ?",
            (since.isoformat(), until_str, int(limit)),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]


class WatchlistsRepo(_Repo):
    """Universe-refresh runs + per-strategy watchlist entries (migration 014).

    Invariant: at most ONE run has status='applied'. `apply_run` enforces it
    inside a transaction by marking any previously-applied run 'superseded'.
    """

    table = "watchlist_runs"

    async def insert_run(self, run: WatchlistRun) -> int:
        return await self._insert(self._serialize(run))

    async def insert_entry(self, entry: WatchlistEntry) -> int:
        data = entry.model_dump(mode="json", exclude_none=True)
        data.pop("id", None)
        cols = ", ".join(data)
        placeholders = ", ".join(["?"] * len(data))
        c = await self.db.connect()
        async with c.execute(
            f"INSERT INTO watchlist_entries ({cols}) VALUES ({placeholders})",
            tuple(data.values()),
        ) as cur:
            new_id = cur.lastrowid
        await c.commit()
        assert new_id is not None
        return new_id

    async def get_run(self, run_id: int) -> WatchlistRun | None:
        row = await self._fetch_one(
            "SELECT * FROM watchlist_runs WHERE id = ?", (run_id,)
        )
        return WatchlistRun(**row) if row else None

    async def latest_run(self, status: str | None = None) -> WatchlistRun | None:
        if status is not None:
            row = await self._fetch_one(
                "SELECT * FROM watchlist_runs WHERE status = ? "
                "ORDER BY created_at DESC, id DESC LIMIT 1",
                (status,),
            )
        else:
            row = await self._fetch_one(
                "SELECT * FROM watchlist_runs ORDER BY created_at DESC, id DESC LIMIT 1"
            )
        return WatchlistRun(**row) if row else None

    async def entries_for_run(self, run_id: int) -> list[WatchlistEntry]:
        rows = await self._fetch_all(
            "SELECT * FROM watchlist_entries WHERE run_id = ? "
            "ORDER BY strategy_id, action, symbol",
            (run_id,),
        )
        return [WatchlistEntry(**r) for r in rows]

    async def applied_membership(self) -> dict[str, list[str]]:
        """Membership map {strategy_id: [symbols]} from the currently-applied
        run (entries with action != 'drop'). Empty dict when no run is applied."""
        run = await self.latest_run(status=WatchlistRunStatus.APPLIED.value)
        if run is None or run.id is None:
            return {}
        out: dict[str, list[str]] = {}
        for e in await self.entries_for_run(run.id):
            if e.action == "drop":
                continue
            out.setdefault(e.strategy_id, []).append(e.symbol.upper())
        return out

    async def apply_run(self, run_id: int, *, applied_by: str) -> None:
        """Mark run applied; supersede any previously-applied run. Transactional
        so the one-applied-run invariant can't be violated by a crash between
        the two writes."""
        now = datetime.now(UTC).isoformat()
        async with self.db.transaction() as c:
            await c.execute(
                "UPDATE watchlist_runs SET status = ? WHERE status = ? AND id != ?",
                (WatchlistRunStatus.SUPERSEDED.value, WatchlistRunStatus.APPLIED.value, run_id),
            )
            await c.execute(
                "UPDATE watchlist_runs SET status = ?, applied_at = ?, applied_by = ? WHERE id = ?",
                (WatchlistRunStatus.APPLIED.value, now, applied_by, run_id),
            )

    async def set_status(self, run_id: int, status: str) -> None:
        await self._update(run_id, {"status": status})


class Repos:
    """Convenience bundle so callers can pass a single object."""

    def __init__(self, db: Database):
        self.db = db
        self.positions = PositionsRepo(db)
        self.orders = OrdersRepo(db)
        self.cycles = WheelCyclesRepo(db)
        self.state_log = StateLogRepo(db)
        self.candidates = CandidatesRepo(db)
        self.regime = RegimeSnapshotsRepo(db)
        self.llm_decisions = LlmDecisionsRepo(db)
        self.iv_history = IvHistoryRepo(db)
        self.daily_state = DailyStateRepo(db)
        self.chain_snapshots = ChainSnapshotsRepo(db)
        self.strategy_runtime = StrategyRuntimeStateRepo(db)
        self.macro_events = MacroEventsRepo(db)
        self.alert_rate_limits = AlertRateLimitsRepo(db)
        self.broker_parity_log = BrokerParityLogRepo(db)
        self.watchlists = WatchlistsRepo(db)
