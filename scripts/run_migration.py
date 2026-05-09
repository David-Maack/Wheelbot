"""Apply schema migrations to the WheelBot SQLite database.

    docker exec wheelbot python -m scripts.run_migration --list
    docker exec wheelbot python -m scripts.run_migration --version 004
    docker exec wheelbot python -m scripts.run_migration --all-pending

Migrations live in `db/migrations/NNN_name.sql`. Applied versions are
tracked in a `schema_migrations` table — re-running an already-applied
migration is a no-op.

Idempotent. Single-file SQLite. Crash-safe via per-migration transactions.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

from core.config import load_config


MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "db" / "migrations"
FILENAME_RE = re.compile(r"^(\d{3,})_([a-zA-Z0-9_]+)\.sql$")


@dataclass(frozen=True, slots=True)
class Migration:
    version: str
    name: str
    path: Path


def _ensure_schema_migrations(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        " version TEXT PRIMARY KEY,"
        " name TEXT NOT NULL,"
        " applied_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP"
        ")"
    )


def _discover() -> list[Migration]:
    if not MIGRATIONS_DIR.exists():
        return []
    out: list[Migration] = []
    for p in sorted(MIGRATIONS_DIR.iterdir()):
        m = FILENAME_RE.match(p.name)
        if not m:
            continue
        out.append(Migration(version=m.group(1), name=m.group(2), path=p))
    return out


def _applied_versions(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT version FROM schema_migrations").fetchall()
    return {r[0] for r in rows}


def apply_migration(conn: sqlite3.Connection, migration: Migration) -> None:
    sql = migration.path.read_text(encoding="utf-8")
    # SQLite executes statements one at a time. executescript() runs the whole
    # file in implicit transactions; wrap explicitly so a partial failure
    # doesn't leave the schema half-migrated.
    try:
        conn.execute("BEGIN")
        conn.executescript(sql)
        conn.execute(
            "INSERT INTO schema_migrations (version, name) VALUES (?, ?)",
            (migration.version, migration.name),
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--list", action="store_true", help="List migrations + status")
    group.add_argument("--version", help="Apply a specific migration version")
    group.add_argument("--all-pending", action="store_true", help="Apply all unapplied migrations")
    parser.add_argument(
        "--db-path",
        type=Path,
        help="Override DB path (defaults to config.database.path)",
    )
    args = parser.parse_args(argv)

    config = load_config()
    db_path = args.db_path or Path(config.get("database", {}).get("path", "wheelbot.db"))
    if not db_path.exists():
        print(f"DB not found: {db_path}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    _ensure_schema_migrations(conn)
    applied = _applied_versions(conn)
    discovered = _discover()

    if args.list or (not args.version and not args.all_pending):
        print(f"DB: {db_path}")
        print(f"{'version':<10} {'name':<32} {'status':<10}")
        for m in discovered:
            status = "applied" if m.version in applied else "pending"
            print(f"{m.version:<10} {m.name:<32} {status:<10}")
        return 0

    targets: list[Migration] = []
    if args.version:
        match = next((m for m in discovered if m.version == args.version), None)
        if match is None:
            print(f"No migration with version {args.version}", file=sys.stderr)
            return 1
        if match.version in applied:
            print(f"Migration {match.version} already applied — no-op")
            return 0
        targets = [match]
    elif args.all_pending:
        targets = [m for m in discovered if m.version not in applied]
        if not targets:
            print("Nothing to apply.")
            return 0

    for m in targets:
        print(f"Applying {m.version}_{m.name}...")
        apply_migration(conn, m)
        print(f"  ok ({m.path.stat().st_size} bytes)")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
