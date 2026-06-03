-- Migration 009 — TICKET-007 (reusable Discord/notify rate-limiter)
--
-- Generic key→timestamp store for "only fire this alert once every N hours".
-- Survives process restarts so a flaky cron doesn't keep silencing itself by
-- restarting; survives container rebuilds so deploy churn doesn't reset.
--
-- Usage pattern (data/alert_rate_limits.py::AlertRateLimitsRepo.try_fire):
--
--     if await repos.alert_rate_limits.try_fire("macro_calendar_stale", cooldown_hours=20):
--         await notify(...)
--
-- alert_key naming convention: snake_case, scoped by feature.
-- Examples: macro_calendar_stale, macro_calendar_empty,
--           earnings_recheck_repeat_<symbol>, weekly_review_unack.
--
-- Run via: docker exec wheelbot python -m scripts.run_migration --version 009
-- Idempotent — runner skips if 009 is already applied.

CREATE TABLE IF NOT EXISTS alert_rate_limits (
    alert_key      TEXT PRIMARY KEY,
    last_fired_at  DATETIME NOT NULL
);
