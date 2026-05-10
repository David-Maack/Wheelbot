-- Migration 004 — Sub-sprint 1 (multi-strategy framework)
--
-- Adds strategy_id to positions, orders, wheel_cycles, chain_snapshots so
-- multiple strategies (monthly_wheel, weekly_wheel, put_spread) can
-- co-exist in the same account.
--
-- positions had UNIQUE(account_id, symbol). The new constraint is
-- UNIQUE(account_id, symbol, strategy_id) so a single ticker can be held
-- by multiple strategies simultaneously. SQLite can't drop a UNIQUE
-- constraint via ALTER TABLE, so we recreate the table.
--
-- Run via: docker exec wheelbot python -m scripts.run_migration --version 004
--
-- Idempotent: the runner skips if migration 004 is already in
-- schema_migrations. Safe to re-run on already-migrated DBs.

-- 0. Use legacy ALTER TABLE semantics for the rename so that FK
--    references in OTHER tables (state_log → positions) are NOT auto-
--    rewritten to point at positions_old. We want them to keep referring
--    to "positions" by name so when the new positions table is created,
--    the FKs resolve to it cleanly. The runner has FK enforcement
--    disabled around the migration body; foreign_key_check verifies
--    integrity before committing.
PRAGMA legacy_alter_table = ON;

-- 1. Rename old positions to a temp name. SQLite renames the implicit
--    UNIQUE index alongside the table, and DROP TABLE later cleans it up.
ALTER TABLE positions RENAME TO positions_old;

-- 2. Create new positions with strategy_id + new UNIQUE constraint.
CREATE TABLE positions (
    id                    INTEGER PRIMARY KEY,
    account_id            TEXT    NOT NULL,
    symbol                TEXT    NOT NULL,
    strategy_id           TEXT    NOT NULL DEFAULT 'monthly_wheel',
    state                 TEXT    NOT NULL,
    shares                INTEGER NOT NULL DEFAULT 0,
    cost_basis            REAL,
    current_cycle_id      INTEGER,
    state_changed_at      DATETIME NOT NULL,
    state_change_reason   TEXT,
    UNIQUE(account_id, symbol, strategy_id),
    FOREIGN KEY (current_cycle_id) REFERENCES wheel_cycles(id)
);

-- 4. Copy data, defaulting existing rows to monthly_wheel.
INSERT INTO positions
    (id, account_id, symbol, strategy_id, state, shares, cost_basis,
     current_cycle_id, state_changed_at, state_change_reason)
SELECT id, account_id, symbol, 'monthly_wheel', state, shares, cost_basis,
       current_cycle_id, state_changed_at, state_change_reason
FROM positions_old;

-- 5. Drop the temp table.
DROP TABLE positions_old;

-- 6. Recreate indexes.
CREATE INDEX IF NOT EXISTS idx_positions_state ON positions(state);
CREATE INDEX IF NOT EXISTS idx_positions_account_state ON positions(account_id, state);
CREATE INDEX IF NOT EXISTS idx_positions_strategy ON positions(strategy_id);

-- 7. Add strategy_id to the other tables (these don't change UNIQUE
--    constraints so simple ALTER TABLE works).
ALTER TABLE orders          ADD COLUMN strategy_id TEXT;
ALTER TABLE wheel_cycles    ADD COLUMN strategy_id TEXT;
ALTER TABLE chain_snapshots ADD COLUMN strategy_id TEXT;

-- 8. Backfill existing rows so they're attributed to monthly_wheel.
UPDATE orders        SET strategy_id = 'monthly_wheel' WHERE strategy_id IS NULL;
UPDATE wheel_cycles  SET strategy_id = 'monthly_wheel' WHERE strategy_id IS NULL;
UPDATE chain_snapshots SET strategy_id = 'monthly_wheel' WHERE strategy_id IS NULL;

CREATE INDEX IF NOT EXISTS idx_orders_strategy ON orders(strategy_id);
CREATE INDEX IF NOT EXISTS idx_cycles_strategy ON wheel_cycles(strategy_id);

-- Restore default ALTER TABLE behavior for any subsequent migrations
-- run on the same connection.
PRAGMA legacy_alter_table = OFF;
