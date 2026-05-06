-- Migration 003 — Sprint 8
-- Captured option chains at decision points so scripts/backtest_cycle.py can
-- replay decisions against historical chains instead of re-fetching live data
-- that has moved on. The cycle backtester reads from this table when present
-- and falls back to Order.raw_request for cycles that predate it.

CREATE TABLE IF NOT EXISTS chain_snapshots (
    id              INTEGER PRIMARY KEY,
    captured_at     DATETIME NOT NULL,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,
    underlying_price REAL,
    contracts       JSON NOT NULL,
    decision_id     INTEGER,
    cycle_id        INTEGER,
    notes           TEXT,
    FOREIGN KEY (decision_id) REFERENCES llm_decisions(id),
    FOREIGN KEY (cycle_id) REFERENCES wheel_cycles(id)
);

CREATE INDEX IF NOT EXISTS idx_chain_snapshots_symbol_date ON chain_snapshots(symbol, captured_at);
CREATE INDEX IF NOT EXISTS idx_chain_snapshots_cycle ON chain_snapshots(cycle_id);
