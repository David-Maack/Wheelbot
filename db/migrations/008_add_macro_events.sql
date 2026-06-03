-- Migration 008 — TICKET-007 (macro event blackout overlay)
--
-- Stores upcoming high-impact macro events (FOMC, CPI, NFP, ...) that the
-- risk gate uses to block new short-premium entries whose lifespan would
-- cross the event. Populated by scripts/refresh_macro_calendar.py (daily
-- cron) from Finnhub /calendar/economic, with a hand-curated YAML fallback.
--
-- event_type values written by the refresh script:
--   FOMC | CPI | NFP | PPI | GDP | JOLTS | OTHER
-- "OTHER" is the bucket for Finnhub event names that don't match any
-- canonical type — see data/macro_calendar.py::_FINNHUB_TYPE_MAP.
--
-- Run via: docker exec wheelbot python -m scripts.run_migration --version 008
-- Idempotent — runner skips if 008 is already applied.

CREATE TABLE IF NOT EXISTS macro_events (
    id           INTEGER PRIMARY KEY,
    event_date   DATE     NOT NULL,
    event_type   TEXT     NOT NULL,           -- canonical type per _FINNHUB_TYPE_MAP
    impact       TEXT     NOT NULL,           -- high | medium | low
    description  TEXT,
    fetched_at   DATETIME NOT NULL,           -- when the refresh script wrote this row
    created_at   DATETIME NOT NULL,
    UNIQUE(event_date, event_type)
);

CREATE INDEX IF NOT EXISTS idx_macro_events_date ON macro_events(event_date);
