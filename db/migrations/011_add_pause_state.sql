-- TICKET-009: separate per-strategy pause columns for the win-rate floor.
-- See risk/win_rate_floor.py.
--
-- pause_state independent of drawdown_state — a strategy can be in BOTH
-- drawdown WARNING (sizes halved) AND pause LOW_WIN_RATE (skip entirely)
-- on the same tick. The bot's runtime gate applies the more restrictive
-- (PAUSE supersedes WARNING).
--
-- paused_reenable_eligible_at is advisory only — the bot does NOT auto-clear
-- when this passes. Operator must run scripts/reenable_strategy.py.
-- Drawdown auto-resets at 30d (current_state in auto_disable.py); win-rate
-- pause requires human judgment because the underlying signal is evidence
-- of a losing pattern, not a transient drawdown.
--
-- SQLite caveat (same as migration 010): no DEFAULT clauses — adding a
-- DEFAULT to a new column would rewrite every existing row.

ALTER TABLE strategy_runtime_state ADD COLUMN pause_state TEXT;
ALTER TABLE strategy_runtime_state ADD COLUMN paused_at DATETIME;
ALTER TABLE strategy_runtime_state ADD COLUMN paused_reason TEXT;
ALTER TABLE strategy_runtime_state ADD COLUMN paused_reenable_eligible_at DATETIME;
