-- Migration 001 — Sprint 4
-- Adds daily_state table used by execution/kill_switch.py to anchor the
-- daily-P&L gate (rule 8 of §8) across mid-day restarts and to track
-- consecutive losing cycles (rule 9). schema.sql holds the canonical copy
-- so a fresh bootstrap also creates this table; this file exists for the
-- migration audit trail.

CREATE TABLE IF NOT EXISTS daily_state (
    id                    INTEGER PRIMARY KEY,
    account_id            TEXT    NOT NULL,
    snapshot_date         DATE    NOT NULL,
    session_open_equity   REAL,
    consecutive_losses    INTEGER NOT NULL DEFAULT 0,
    kill_switch_armed     BOOLEAN NOT NULL DEFAULT 0,
    kill_switch_reason    TEXT,
    UNIQUE(account_id, snapshot_date)
);

CREATE INDEX IF NOT EXISTS idx_daily_state_date ON daily_state(snapshot_date);
