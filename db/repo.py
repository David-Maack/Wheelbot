"""Async repository layer over SQLite (aiosqlite).

One repository class per table. Repositories own row → model conversion. The strategy and
execution layers MUST go through repositories — never raw SQL — so swapping SQLite for
Postgres later is mechanical.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from core.models import (
    Candidate,
    DailyState,
    IvHistory,
    LlmDecision,
    Order,
    Position,
    PositionState,
    RegimeSnapshot,
    StateLog,
    WheelCycle,
)

JSON_FIELDS_BY_TABLE: dict[str, tuple[str, ...]] = {
    "orders": ("raw_request", "raw_response"),
    "state_log": ("metadata",),
    "candidates": ("raw_llm_response",),
    "llm_decisions": ("context", "response"),
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
    """Connection holder. One per process. Open with `connect()` or use as context manager."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> aiosqlite.Connection:
        if self._conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = await aiosqlite.connect(self.path)
            self._conn.row_factory = aiosqlite.Row
            await self._conn.execute("PRAGMA foreign_keys = ON")
            await self._conn.execute("PRAGMA journal_mode = WAL")
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

    async def get_by_symbol(self, account_id: str, symbol: str) -> Position | None:
        row = await self._fetch_one(
            "SELECT * FROM positions WHERE account_id = ? AND symbol = ?",
            (account_id, symbol),
        )
        return Position(**row) if row else None

    async def list_active(self, account_id: str) -> list[Position]:
        rows = await self._fetch_all(
            "SELECT * FROM positions WHERE account_id = ? AND state != ? ORDER BY symbol",
            (account_id, PositionState.IDLE.value),
        )
        return [Position(**r) for r in rows]

    async def list_all(self, account_id: str) -> list[Position]:
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

    async def list_recent(self, account_id: str, limit: int = 50) -> list[Order]:
        rows = await self._fetch_all(
            "SELECT * FROM orders WHERE account_id = ? ORDER BY placed_at DESC LIMIT ?",
            (account_id, limit),
        )
        return [Order(**r) for r in rows]

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
