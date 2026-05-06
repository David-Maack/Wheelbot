"""Nightly SQLite backup.

    python -m scripts.backup_db
    python -m scripts.backup_db --backups-dir /custom/path --keep 30

Uses SQLite's `.backup` API so concurrent reconciler reads don't tear the file.
Outputs `<backups_dir>/wheelbot-YYYY-MM-DD.sql.gz`. Prunes anything older than
`--keep` days (default 30) per spec §13 #25.

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


def _prune(backups_dir: Path, keep_days: int, today: date | None = None) -> list[Path]:
    """Remove wheelbot-YYYY-MM-DD.sql.gz files older than `keep_days`."""
    today = today or _today_utc()
    cutoff = today - timedelta(days=keep_days)
    removed: list[Path] = []
    if not backups_dir.exists():
        return removed
    for p in backups_dir.iterdir():
        m = _FILENAME_RE.match(p.name)
        if not m:
            continue
        try:
            file_date = date(int(m[1]), int(m[2]), int(m[3]))
        except ValueError:
            continue
        if file_date < cutoff:
            p.unlink()
            removed.append(p)
    return removed


def run(db_path: Path, backups_dir: Path, keep_days: int, today: date | None = None) -> Path:
    today = today or _today_utc()
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")
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
    _prune(backups_dir, keep_days, today=today)
    return out_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path)
    parser.add_argument("--backups-dir", type=Path)
    parser.add_argument("--keep", type=int, default=30)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = load_config()
    db_path = args.db_path or Path(config.get("database", {}).get("path", "wheelbot.db")).expanduser()
    backups_dir = args.backups_dir or (db_path.parent / "backups")
    out_path = run(db_path, backups_dir, args.keep)
    print(f"Backup written: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
