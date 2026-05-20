-- Migration 006 — Sprint 13 sub-sprint 1 (auto-disable on drawdown)
--
-- Tracks runtime-only "this strategy is paused" state, distinct from the
-- static `enabled` flag in config.yaml. Lets the auto-disable circuit
-- breaker pause a strategy after a drawdown threshold is breached without
-- requiring a config edit + redeploy.
--
-- disabled_until: NULL means strategy is currently enabled (or never
-- disabled). A future datetime means strategy is paused until that time;
-- past datetime auto-clears on next is_currently_disabled() check.
--
-- Run via: docker exec wheelbot python -m scripts.run_migration --version 006
--
-- Idempotent — runner skips if 006 is already applied.

CREATE TABLE IF NOT EXISTS strategy_runtime_state (
    strategy_id      TEXT PRIMARY KEY,
    disabled_at      DATETIME,
    disabled_until   DATETIME,
    disabled_reason  TEXT
);
