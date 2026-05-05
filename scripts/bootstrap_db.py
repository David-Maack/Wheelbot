"""Create the WheelBot SQLite database from db/schema.sql.

Idempotent: re-running on an existing DB is safe (CREATE TABLE IF NOT EXISTS).
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

from core.checkpoint import checkpoint
from core.config import load_config

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "db" / "schema.sql"


def bootstrap(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    schema = SCHEMA_PATH.read_text(encoding="utf-8")

    with checkpoint("bootstrap_db", db_path=str(db_path)):
        with sqlite3.connect(db_path) as conn:
            conn.executescript(schema)
            conn.commit()


def main() -> int:
    config = load_config()
    db_path = Path(config["database"]["path"]).expanduser()
    bootstrap(db_path)
    print(f"Database ready at {db_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
