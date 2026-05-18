-- Migration 005 — Sprint 10 (bear_call_spread)
--
-- Adds `bear_calls_allowed` to regime_snapshots so the risk gate can
-- evaluate bear call credit spreads independently of CSPs.
--
-- Decision rule (computed in risk/regime.py::classify_regime):
--   bear_calls_allowed = False  if  HIGH_VOL  or  BULL_TREND
--                       = True   if  BEAR_TREND  or  NEUTRAL
--
-- Backfill for existing rows derives the value from the stored `regime`
-- column so historical snapshots have a self-consistent flag.
--
-- Run via: docker exec wheelbot python -m scripts.run_migration --version 005
--
-- Idempotent: the runner skips if migration 005 is already in
-- schema_migrations. Safe to re-run on already-migrated DBs.

ALTER TABLE regime_snapshots ADD COLUMN bear_calls_allowed BOOLEAN;

-- Backfill: derive from existing regime values so historical rows are
-- consistent with the live classifier.
UPDATE regime_snapshots
SET bear_calls_allowed = CASE
    WHEN regime = 'BEAR_TREND' THEN 1
    WHEN regime = 'NEUTRAL'    THEN 1
    WHEN regime = 'BULL_TREND' THEN 0
    WHEN regime = 'HIGH_VOL'   THEN 0
    ELSE NULL
END
WHERE bear_calls_allowed IS NULL;
