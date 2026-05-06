"""scripts/backup_db — gzip + prune."""

from __future__ import annotations

import gzip
import shutil
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pytest

from scripts.backup_db import _prune, run


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
    out = run(db_path, out_dir, keep_days=30, today=date(2025, 6, 5))
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

    run(db_path, backups, keep_days=30, today=today)

    assert not stale.exists()
    fresh = backups / f"wheelbot-{today.isoformat()}.sql.gz"
    assert fresh.exists()
