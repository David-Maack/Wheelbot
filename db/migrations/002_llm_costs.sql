-- Migration 002 — Sprint 7
-- Adds token-usage and dollar-cost columns to llm_decisions so the daily
-- budget gate (intelligence/budget.py) can sum spend per day.
--
-- SQLite ALTER TABLE only supports adding columns one-at-a-time. These three
-- statements are idempotent for fresh schemas (where schema.sql already
-- includes them) — running this against a fresh DB is a no-op when the
-- columns exist; SQLite raises if they do, hence the workaround in the
-- bootstrap step (try/except per column).

ALTER TABLE llm_decisions ADD COLUMN tokens_in  INTEGER;
ALTER TABLE llm_decisions ADD COLUMN tokens_out INTEGER;
ALTER TABLE llm_decisions ADD COLUMN cost_usd   REAL;
