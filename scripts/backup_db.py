"""Nightly SQLite backup + retention.

    python -m scripts.backup_db
    python -m scripts.backup_db --backups-dir /custom/path --keep-daily 7 --keep-weekly 4
    python -m scripts.backup_db --vacuum      # one-off compaction; STOP the bot first

Uses SQLite's `.backup` API so concurrent reconciler reads don't tear the file.
Outputs `<backups_dir>/wheelbot-YYYY-MM-DD.sql.gz`.

Retention (2026-10-27 audit, plan step D — the LXC disk was 72% full with a
1.25 GB DB growing ~220 MB/month and 3.8 GB of 31 daily backups):

  * chain_snapshots — rows older than `database.chain_snapshot_retention_days`
    (config; 0/absent = keep everything) are deleted BEFORE the backup, unless
    tied to a cycle or to a symbol-day with a FILLED order (the chains behind
    real trades stay for scripts/backtest_cycle). ~200 MB/month of the growth
    was chains evaluated by strategies that never traded.
  * backups — keep every file from the last `--keep-daily` days (7) plus the
    newest file of each of the `--keep-weekly` (4) ISO weeks before that.

Deleting rows doesn't shrink the file; `--vacuum` rewrites it after pruning.
VACUUM needs exclusive access, so run it once with the bot, dashboard and
MCP containers stopped.

Cron line (in the LXC):

    30 23 * * *  /opt/wheelbot/.venv/bin/python -m scripts.backup_db >>/var/log/wheelbot-backup.log 2>&1
"""

from __future__ import annotations

import argparse
import gzip
import re
import shutil
import sqlite3
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from core.config import load_config


_FILENAME_RE = re.compile(r"^wheelbot-(\d{4})-(\d{2})-(\d{2})\.sql\.gz$")


def _today_utc() -> date:
    return datetime.now(timezone.utc).date()


def _backup_to_temp(src: Path) -> Path:
    """SQLite online backup → temporary uncompressed copy."""
    tmp = Path(tempfile.mkdtemp(prefix="wheelbot-backup-")) / "wheelbot.sql"
    src_conn = sqlite3.connect(src)
    try:
        dest_conn = sqlite3.connect(tmp)
        try:
            with dest_conn:
                src_conn.backup(dest_conn)
        finally:
            dest_conn.close()
    finally:
        src_conn.close()
    return tmp


def _gzip_into(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as f_in, gzip.open(dest, "wb", compresslevel=6) as f_out:
        shutil.copyfileobj(f_in, f_out)


def _prune(
    backups_dir: Path,
    keep_daily: int,
    keep_weekly: int = 0,
    today: date | None = None,
) -> list[Path]:
    """Keep every wheelbot-YYYY-MM-DD.sql.gz from the last `keep_daily` days
    plus the newest one of each of the `keep_weekly` most recent ISO weeks
    older than that; remove the rest."""
    today = today or _today_utc()
    daily_cutoff = today - timedelta(days=max(keep_daily, 1) - 1)  # today counts as day 1
    removed: list[Path] = []
    if not backups_dir.exists():
        return removed
    older: list[tuple[date, Path]] = []
    for p in backups_dir.iterdir():
        m = _FILENAME_RE.match(p.name)
        if not m:
            continue
        try:
            file_date = date(int(m[1]), int(m[2]), int(m[3]))
        except ValueError:
            continue
        if file_date < daily_cutoff:
            older.append((file_date, p))
    keep: set[Path] = set()
    weeks_kept: list[tuple[int, int]] = []
    for file_date, p in sorted(older, reverse=True):
        week = file_date.isocalendar()[:2]
        if week in weeks_kept:
            continue
        if len(weeks_kept) >= keep_weekly:
            break
        weeks_kept.append(week)
        keep.add(p)
    for _, p in older:
        if p not in keep:
            p.unlink()
            removed.append(p)
    return removed


def prune_chain_snapshots(db_path: Path, retention_days: int, today: date | None = None) -> int:
    """Delete chain_snapshots older than `retention_days` that are not tied to
    a cycle or to a symbol-day with a FILLED order. Returns rows deleted;
    retention_days <= 0 keeps everything."""
    if retention_days <= 0:
        return 0
    today = today or _today_utc()
    cutoff = (today - timedelta(days=retention_days)).isoformat()
    conn = sqlite3.connect(db_path)
    try:
        with conn:
            cur = conn.execute(
                "DELETE FROM chain_snapshots WHERE captured_at < ? AND cycle_id IS NULL "
                "AND NOT EXISTS (SELECT 1 FROM orders o "
                "WHERE o.symbol = chain_snapshots.symbol AND o.status = 'FILLED' "
                "AND date(o.placed_at) = date(chain_snapshots.captured_at))",
                (cutoff,),
            )
            return cur.rowcount
    finally:
        conn.close()


def vacuum(db_path: Path) -> None:
    """Rewrite the DB file to reclaim pruned pages. Needs exclusive access."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("VACUUM")
    finally:
        conn.close()


def run(
    db_path: Path,
    backups_dir: Path,
    keep_daily: int,
    today: date | None = None,
    *,
    keep_weekly: int = 4,
    chain_retention_days: int = 0,
    do_vacuum: bool = False,
) -> Path:
    today = today or _today_utc()
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")
    deleted = prune_chain_snapshots(db_path, chain_retention_days, today=today)
    if deleted:
        print(f"chain_snapshots pruned: {deleted} rows older than {chain_retention_days}d")
    if do_vacuum:
        vacuum(db_path)
    tmp = _backup_to_temp(db_path)
    try:
        out_path = backups_dir / f"wheelbot-{today.isoformat()}.sql.gz"
        _gzip_into(tmp, out_path)
    finally:
        # Best-effort cleanup of the tmpdir.
        try:
            tmp.unlink()
            tmp.parent.rmdir()
        except OSError:
            pass
    _prune(backups_dir, keep_daily, keep_weekly, today=today)
    return out_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path)
    parser.add_argument("--backups-dir", type=Path)
    parser.add_argument("--keep-daily", "--keep", dest="keep_daily", type=int, default=7)
    parser.add_argument("--keep-weekly", type=int, default=4)
    parser.add_argument("--vacuum", action="store_true",
                        help="VACUUM after pruning (stop the bot containers first)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = load_config()
    db_path = args.db_path or Path(config.get("database", {}).get("path", "wheelbot.db")).expanduser()
    backups_dir = args.backups_dir or (db_path.parent / "backups")
    retention = int(config.get("database", {}).get("chain_snapshot_retention_days", 0) or 0)
    out_path = run(
        db_path, backups_dir, args.keep_daily,
        keep_weekly=args.keep_weekly,
        chain_retention_days=retention,
        do_vacuum=args.vacuum,
    )
    print(f"Backup written: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
