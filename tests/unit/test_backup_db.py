"""scripts/backup_db — gzip, backup retention, chain_snapshots retention."""

from __future__ import annotations

import gzip
import shutil
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pytest

from scripts.backup_db import _prune, prune_chain_snapshots, run


def _seed_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    conn.executemany("INSERT INTO t (v) VALUES (?)", [("a",), ("b",), ("c",)])
    conn.commit()
    conn.close()


def test_backup_produces_valid_gzipped_sqlite(tmp_path: Path):
    db_path = tmp_path / "src.db"
    _seed_db(db_path)
    out_dir = tmp_path / "backups"
    out = run(db_path, out_dir, keep_daily=30, today=date(2025, 6, 5))
    assert out.exists()
    assert out.name == "wheelbot-2025-06-05.sql.gz"

    restored = tmp_path / "restored.db"
    with gzip.open(out, "rb") as f_in, restored.open("wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    conn = sqlite3.connect(restored)
    rows = conn.execute("SELECT COUNT(*) FROM t").fetchone()[0]
    conn.close()
    assert rows == 3


def test_prune_removes_only_old_files(tmp_path: Path):
    backups = tmp_path / "backups"
    backups.mkdir()
    today = date(2025, 6, 5)
    keep_days = 30

    keep1 = backups / f"wheelbot-{today.isoformat()}.sql.gz"
    keep2 = backups / f"wheelbot-{(today - timedelta(days=29)).isoformat()}.sql.gz"
    drop1 = backups / f"wheelbot-{(today - timedelta(days=31)).isoformat()}.sql.gz"
    drop2 = backups / f"wheelbot-{(today - timedelta(days=365)).isoformat()}.sql.gz"
    other = backups / "irrelevant.txt"

    for p in (keep1, keep2, drop1, drop2, other):
        p.write_bytes(b"x")

    removed = _prune(backups, keep_days, today=today)

    assert {p.name for p in removed} == {drop1.name, drop2.name}
    assert keep1.exists() and keep2.exists() and other.exists()
    assert not drop1.exists() and not drop2.exists()


def test_run_prunes_after_backup(tmp_path: Path):
    db_path = tmp_path / "src.db"
    _seed_db(db_path)
    backups = tmp_path / "backups"
    backups.mkdir()

    today = date(2025, 6, 5)
    stale = backups / f"wheelbot-{(today - timedelta(days=60)).isoformat()}.sql.gz"
    stale.write_bytes(b"x")

    run(db_path, backups, keep_daily=30, keep_weekly=0, today=today)

    assert not stale.exists()
    fresh = backups / f"wheelbot-{today.isoformat()}.sql.gz"
    assert fresh.exists()


def test_prune_keeps_daily_window_plus_newest_per_week(tmp_path: Path):
    """7 daily + 4 weekly: everything from the last 7 days, then the newest
    file of each of the 4 most recent ISO weeks before that."""
    backups = tmp_path / "backups"
    backups.mkdir()
    today = date(2026, 10, 27)                       # a Tuesday
    for back in range(0, 60):
        day = (today - timedelta(days=back)).isoformat()
        (backups / f"wheelbot-{day}.sql.gz").write_bytes(b"x")

    _prune(backups, keep_daily=7, keep_weekly=4, today=today)

    kept = sorted(p.name for p in backups.iterdir())
    daily = [f"wheelbot-{(today - timedelta(days=b)).isoformat()}.sql.gz" for b in range(0, 7)]
    assert set(daily) <= set(kept)
    older = [n for n in kept if n not in daily]
    assert len(older) == 4
    weeks = {date.fromisoformat(n[9:19]).isocalendar()[:2] for n in older}
    assert len(weeks) == 4                           # one per ISO week


def _seed_chain_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE chain_snapshots (id INTEGER PRIMARY KEY, captured_at TEXT, "
        "symbol TEXT, side TEXT, cycle_id INTEGER);"
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, symbol TEXT, status TEXT, placed_at TEXT);"
    )
    conn.executemany(
        "INSERT INTO chain_snapshots (captured_at, symbol, side, cycle_id) VALUES (?, ?, ?, ?)",
        [
            ("2026-10-25T15:00:00", "F", "put", None),     # recent -> kept
            ("2026-09-01T15:00:00", "F", "put", None),     # old, no trade -> deleted
            ("2026-09-01T15:00:00", "F", "put", 7),        # old, tied to a cycle -> kept
            ("2026-09-02T15:00:00", "NOK", "put", None),   # old, FILLED order that day -> kept
            ("2026-09-03T15:00:00", "NOK", "put", None),   # old, only a CANCELLED order -> deleted
        ],
    )
    conn.executemany(
        "INSERT INTO orders (symbol, status, placed_at) VALUES (?, ?, ?)",
        [("NOK", "FILLED", "2026-09-02T14:30:00"), ("NOK", "CANCELLED", "2026-09-03T14:30:00")],
    )
    conn.commit()
    conn.close()


def test_prune_chain_snapshots_keeps_recent_and_traded_chains(tmp_path: Path):
    db_path = tmp_path / "w.db"
    _seed_chain_db(db_path)
    assert prune_chain_snapshots(db_path, 0, today=date(2026, 10, 27)) == 0   # off
    assert prune_chain_snapshots(db_path, 14, today=date(2026, 10, 27)) == 2
    conn = sqlite3.connect(db_path)
    left = conn.execute(
        "SELECT captured_at, symbol, cycle_id FROM chain_snapshots ORDER BY id"
    ).fetchall()
    conn.close()
    assert left == [
        ("2026-10-25T15:00:00", "F", None),
        ("2026-09-01T15:00:00", "F", 7),
        ("2026-09-02T15:00:00", "NOK", None),
    ]


def test_run_prunes_chains_and_can_vacuum(tmp_path: Path):
    db_path = tmp_path / "w.db"
    _seed_chain_db(db_path)
    out = run(db_path, tmp_path / "backups", 7, today=date(2026, 10, 27),
              chain_retention_days=14, do_vacuum=True)
    assert out.exists()
    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM chain_snapshots").fetchone()[0] == 3
    conn.close()
