-- 0DTE paper-test sprint (docs/research/0dte_research_2026-07.md §4-5).
--
-- Dual-ledger realism: Alpaca's paper fill simulator fills marketable orders
-- against NBBO with no size constraint and no slippage — systematically
-- OPTIMISTIC for a 0DTE premium seller (the research doc's §4.1). Every
-- zero_dte entry/exit therefore records BOTH a raw figure (what the paper
-- account will roughly show) and a PENALIZED figure (half the summed leg
-- bid-ask spreads conceded on each side, plus a fixed adder on forced
-- flatten exits). Go/no-go decisions read pnl_penalized ONLY.
--
-- Rows are written from the strategy's propose paths using quotes at
-- proposal time; reconciling against actual broker fills is phase 2.

CREATE TABLE IF NOT EXISTS zero_dte_ledger (
    id                          INTEGER PRIMARY KEY,
    cycle_id                    INTEGER REFERENCES wheel_cycles(id),  -- NULL until phase-2 fill reconciliation links it
    symbol                      TEXT    NOT NULL,
    structure                   TEXT    NOT NULL,   -- narrow_vertical | iron_condor
    quantity                    INTEGER NOT NULL DEFAULT 1,           -- packages; pnl_* are dollars across all of them
    entry_ts                    DATETIME NOT NULL,
    entry_credit_raw            REAL,               -- $/share net credit at proposal-time quotes
    entry_spread_width_quotes   REAL,               -- sum of leg (ask - bid) at entry, $/share
    entry_credit_penalized      REAL,               -- raw - entry_spread_width_quotes / 2
    exit_ts                     DATETIME,
    exit_debit_raw              REAL,               -- $/share debit-to-close at proposal-time mids
    exit_debit_penalized        REAL,               -- raw + exit half-spread sum (+ stop_slippage_adder on flatten)
    pnl_raw                     REAL,               -- (entry_credit_raw - exit_debit_raw) * 100 * quantity
    pnl_penalized               REAL,               -- penalized equivalent; feeds the zero_dte daily loss cap
    exit_reason                 TEXT                -- flatten | profit_target
);

CREATE INDEX IF NOT EXISTS idx_zero_dte_ledger_symbol_entry
    ON zero_dte_ledger(symbol, entry_ts);
CREATE INDEX IF NOT EXISTS idx_zero_dte_ledger_exit_ts
    ON zero_dte_ledger(exit_ts);
