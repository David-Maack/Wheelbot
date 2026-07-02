-- Sprint: dynamic universe refresh (two-tier: quant pre-filter + weekly LLM rank).
--
-- A refresh run proposes per-strategy watchlists (add/keep/drop per symbol).
-- Runs are PROPOSED by default — spec §6 says never auto-add tickers without
-- human review, so a human applies via the MCP approve_watchlist tool (or the
-- universe_refresh.auto_apply config flag, default false). Exactly one run is
-- APPLIED at a time; applying a new run marks the previous one SUPERSEDED.
-- The bot overlays the applied run's membership onto universe.yaml at tick
-- time (core/watchlists.py); universe.yaml stays the fallback and the source
-- of tier/override metadata.

CREATE TABLE IF NOT EXISTS watchlist_runs (
    id               INTEGER PRIMARY KEY,
    run_date         DATE    NOT NULL,
    status           TEXT    NOT NULL DEFAULT 'proposed',  -- proposed | applied | rejected | superseded | failed
    llm_decision_id  INTEGER REFERENCES llm_decisions(id),
    cost_usd         REAL,
    summary          TEXT,                                 -- model's one-paragraph refresh summary
    created_at       DATETIME NOT NULL,
    applied_at       DATETIME,
    applied_by       TEXT                                  -- 'auto' | 'mcp' | 'manual'
);

CREATE INDEX IF NOT EXISTS idx_watchlist_runs_status ON watchlist_runs(status);
CREATE INDEX IF NOT EXISTS idx_watchlist_runs_run_date ON watchlist_runs(run_date);

CREATE TABLE IF NOT EXISTS watchlist_entries (
    id           INTEGER PRIMARY KEY,
    run_id       INTEGER NOT NULL REFERENCES watchlist_runs(id),
    strategy_id  TEXT    NOT NULL,
    symbol       TEXT    NOT NULL,
    action       TEXT    NOT NULL DEFAULT 'keep',          -- add | keep | drop
    score        REAL,                                     -- LLM conviction 0-100
    rationale    TEXT,
    UNIQUE(run_id, strategy_id, symbol)
);

CREATE INDEX IF NOT EXISTS idx_watchlist_entries_run ON watchlist_entries(run_id);
