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


def _split_statements(sql: str) -> list[str]:
    """Tokenize a migration SQL file into individual statements.

    Strips full-line comments and splits on ';'. Migration files don't
    contain semicolons inside strings so this naive split is safe.
    """
    lines = []
    for raw in sql.split("\n"):
        stripped = raw.strip()
        if stripped.startswith("--") or not stripped:
            continue
        lines.append(raw)
    text = "\n".join(lines)
    return [s.strip() for s in text.split(";") if s.strip()]


def apply_migration(conn: sqlite3.Connection, migration: Migration) -> None:
    """Apply a migration atomically.

    `with conn:` brackets the body in a transaction — COMMITs on clean
    exit, ROLLBACKs on exception. Each statement runs via `conn.execute()`
    (NOT `executescript`, which clobbers explicit transactions).

    Foreign keys are disabled for the duration of the migration. This is
    the canonical SQLite pattern for schema-changing migrations that
    rename or recreate tables that other tables reference (alternative is
    recreating every dependent table — bad). PRAGMA foreign_keys must run
    OUTSIDE a transaction — it's a no-op inside. After the body runs, we
    `PRAGMA foreign_key_check` (still inside the transaction) and abort the
    commit if any orphans appear, so we never commit a broken FK graph.
    """
    sql = migration.path.read_text(encoding="utf-8")
    statements = _split_statements(sql)
    if not statements:
        raise ValueError(f"migration {migration.version} has no statements")

    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        with conn:
            for i, stmt in enumerate(statements, start=1):
                try:
                    conn.execute(stmt)
                except sqlite3.Error as exc:
                    preview = stmt.replace("\n", " ")[:140]
                    raise sqlite3.Error(
                        f"migration {migration.version} statement {i}/{len(statements)} failed: "
                        f"{preview!r} — {exc}"
                    ) from exc
            violations = list(conn.execute("PRAGMA foreign_key_check"))
            if violations:
                raise sqlite3.IntegrityError(
                    f"FK violations after migration {migration.version}: {violations[:5]}"
                )
            conn.execute(
                "INSERT INTO schema_migrations (version, name) VALUES (?, ?)",
                (migration.version, migration.name),
            )
    finally:
        conn.execute("PRAGMA foreign_keys = ON")


def apply_pending(db_path: Path | str | None = None) -> list[Migration]:
    """Apply every unapplied migration, in order. Returns the list of
    migrations that were applied (empty when up-to-date).

    Programmatic counterpart of `python -m scripts.run_migration --all-pending`,
    intended to be called from `scripts/run_bot.py` at startup so a deploy
    that ships a new migration self-heals on `docker compose up -d --build`
    instead of requiring an extra step the operator can (and did) forget.

    Raises:
        FileNotFoundError: the DB file doesn't exist yet (fresh install —
            run `scripts/bootstrap_db.py` first to create the base schema).
        sqlite3.Error: any migration's SQL failed; the writer transaction
            rolled back, the schema is in its pre-migration state, and the
            bot caller should NOT proceed.
    """
    config = load_config()
    resolved = (
        Path(db_path)
        if db_path is not None
        else Path(config.get("database", {}).get("path", "wheelbot.db"))
    )
    if not resolved.exists():
        raise FileNotFoundError(f"DB not found: {resolved}")
    conn = sqlite3.connect(resolved)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        _ensure_schema_migrations(conn)
        applied = _applied_versions(conn)
        pending = [m for m in _discover() if m.version not in applied]
        for m in pending:
            apply_migration(conn, m)
        return pending
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--list", action="store_true", help="List migrations + status")
    group.add_argument("--version", help="Apply a specific migration version")
    group.add_argument("--all-pending", action="store_true", help="Apply all unapplied migrations")
    group.add_argument(
        "--mark-applied",
        help="Record a migration as applied WITHOUT running its SQL. Use "
        "after a manual recovery from a broken migration runner — DO NOT "
        "use to skip a migration that hasn't actually been applied.",
    )
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

    if args.mark_applied:
        m = next((x for x in discovered if x.version == args.mark_applied), None)
        if m is None:
            print(f"No migration with version {args.mark_applied}", file=sys.stderr)
            return 1
        if m.version in applied:
            print(f"Migration {m.version} already marked applied — no-op")
            return 0
        with conn:
            conn.execute(
                "INSERT INTO schema_migrations (version, name) VALUES (?, ?)",
                (m.version, m.name),
            )
        print(f"Marked {m.version}_{m.name} as applied (without running SQL).")
        return 0

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
