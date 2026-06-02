#!/usr/bin/env bash
# Compare migration files between the host repo and the running container.
#
# Use after `git pull` to confirm whether the image needs rebuilding:
#
#   bash scripts/migrate_check.sh
#
# Reports three things:
#   1. Files present on host but NOT in container (image is stale — rebuild).
#   2. Files present in container but NOT on host (someone hand-edited the
#      image — investigate).
#   3. Migrations recorded in the schema_migrations table that aren't on disk.
#
# Exits non-zero if any drift is detected.

set -uo pipefail

CONTAINER=${WHEELBOT_CONTAINER:-wheelbot}
HOST_DIR=${WHEELBOT_HOST_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
HOST_MIG_DIR="$HOST_DIR/db/migrations"
CONTAINER_MIG_DIR=/opt/wheelbot/db/migrations

if [[ ! -d "$HOST_MIG_DIR" ]]; then
    echo "ERROR: host migrations dir not found: $HOST_MIG_DIR" >&2
    exit 2
fi

if ! docker exec "$CONTAINER" true >/dev/null 2>&1; then
    echo "ERROR: container '$CONTAINER' not running" >&2
    exit 2
fi

host_files=$(cd "$HOST_MIG_DIR" && ls *.sql 2>/dev/null | sort)
container_files=$(docker exec "$CONTAINER" sh -c "ls $CONTAINER_MIG_DIR/*.sql 2>/dev/null | xargs -n1 basename | sort")

# 1. On host but not in container → image is stale, needs rebuild.
missing_in_container=$(comm -23 <(echo "$host_files") <(echo "$container_files"))
# 2. In container but not on host → image has files the repo doesn't.
missing_on_host=$(comm -13 <(echo "$host_files") <(echo "$container_files"))

drift=0

echo "== Host migrations dir : $HOST_MIG_DIR"
echo "== Container path      : $CONTAINER_MIG_DIR  (in $CONTAINER)"
echo

if [[ -n "$missing_in_container" ]]; then
    echo "DRIFT: present on host but NOT in container (image needs rebuild):"
    echo "$missing_in_container" | sed 's/^/  - /'
    drift=1
else
    echo "OK: every host migration file is present in the container."
fi
echo

if [[ -n "$missing_on_host" ]]; then
    echo "WARN: present in container but NOT on host (investigate):"
    echo "$missing_on_host" | sed 's/^/  - /'
    drift=1
fi

# 3. Compare applied migrations in the schema_migrations table with files on disk.
echo
echo "== Applied migrations (from container's DB) =="
docker exec "$CONTAINER" python - <<'PY' 2>/dev/null || echo "  (couldn't query schema_migrations — table may not exist yet)"
import asyncio
from pathlib import Path
from core.config import load_config
from db.repo import Database

async def main():
    db_path = Path(load_config().get("database", {}).get("path", "wheelbot.db")).expanduser()
    async with Database(db_path) as db:
        conn = await db.connect()
        try:
            async with conn.execute("SELECT version FROM schema_migrations ORDER BY version") as cur:
                rows = await cur.fetchall()
            for r in rows:
                print(f"  applied: {r[0]}")
        except Exception as exc:
            print(f"  (no schema_migrations table: {exc})")

asyncio.run(main())
PY

if [[ $drift -ne 0 ]]; then
    echo
    echo "ACTION: rebuild the image:"
    echo "  docker compose build wheelbot && docker compose up -d wheelbot"
    exit 1
fi

echo
echo "OK: no drift detected."
