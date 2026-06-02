-- WheelBot SQLite schema. Source of truth — see spec §7.
-- All DATETIME columns are stored as ISO-8601 UTC strings.

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

-- Current state per (account, symbol, strategy). A single ticker can be
-- traded by multiple strategies simultaneously (e.g. monthly_wheel on F at
-- one strike + weekly_wheel on F at another).
CREATE TABLE IF NOT EXISTS positions (
    id                    INTEGER PRIMARY KEY,
    account_id            TEXT    NOT NULL,
    symbol                TEXT    NOT NULL,
    strategy_id           TEXT    NOT NULL DEFAULT 'monthly_wheel',
    state                 TEXT    NOT NULL,           -- IDLE, CSP_OPEN, SHARES_HELD, CC_OPEN, SPREAD_OPEN, ...
    shares                INTEGER NOT NULL DEFAULT 0,
    cost_basis            REAL,                       -- avg cost per share, premium-adjusted
    current_cycle_id      INTEGER,                    -- FK to wheel_cycles
    state_changed_at      DATETIME NOT NULL,
    state_change_reason   TEXT,
    UNIQUE(account_id, symbol, strategy_id),
    FOREIGN KEY (current_cycle_id) REFERENCES wheel_cycles(id)
);

CREATE INDEX IF NOT EXISTS idx_positions_state ON positions(state);
CREATE INDEX IF NOT EXISTS idx_positions_account_state ON positions(account_id, state);
CREATE INDEX IF NOT EXISTS idx_positions_strategy ON positions(strategy_id);

-- Every order ever submitted (CSP, CC, BTC, BTO)
CREATE TABLE IF NOT EXISTS orders (
    id                INTEGER PRIMARY KEY,
    account_id        TEXT    NOT NULL,
    symbol            TEXT    NOT NULL,
    strategy_id       TEXT,                           -- which strategy submitted this order; nullable for legacy rows
    cycle_id          INTEGER,
    broker_order_id   TEXT    UNIQUE,                 -- broker's ID (idempotency)
    client_order_id   TEXT    UNIQUE,                 -- our UUID (dedupe)
    order_type        TEXT    NOT NULL,               -- SELL_TO_OPEN, BUY_TO_CLOSE, MULTI_LEG_OPEN, etc.
    contract_symbol   TEXT,                           -- OCC option symbol or stock ticker
    strike            REAL,
    expiration        DATE,
    option_type       TEXT,                           -- PUT, CALL, NULL for stock or multi-leg
    quantity          INTEGER NOT NULL,
    limit_price       REAL,
    fill_price        REAL,
    status            TEXT    NOT NULL,               -- PENDING, FILLED, PARTIAL, CANCELLED, REJECTED
    placed_at         DATETIME NOT NULL,
    filled_at         DATETIME,
    raw_request       JSON,                           -- full broker request body
    raw_response      JSON,                           -- full broker response
    trigger_reason    TEXT,                           -- TICKET-005 (migration 007) — close-reason tag
    FOREIGN KEY (cycle_id) REFERENCES wheel_cycles(id)
);

CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol);
CREATE INDEX IF NOT EXISTS idx_orders_placed_at ON orders(placed_at);
CREATE INDEX IF NOT EXISTS idx_orders_cycle_id ON orders(cycle_id);
CREATE INDEX IF NOT EXISTS idx_orders_strategy ON orders(strategy_id);

-- A wheel cycle = CSP open → ... → final close. One symbol can have many cycles over time.
CREATE TABLE IF NOT EXISTS wheel_cycles (
    id                       INTEGER PRIMARY KEY,
    account_id               TEXT    NOT NULL,
    symbol                   TEXT    NOT NULL,
    strategy_id              TEXT,                    -- which strategy owns this cycle; nullable for legacy rows
    started_at               DATETIME NOT NULL,
    ended_at                 DATETIME,
    initial_csp_strike       REAL,
    initial_csp_premium      REAL,
    initial_capital_at_risk  REAL,
    final_pnl                REAL,                    -- realized at cycle end
    final_pnl_pct            REAL,
    cycle_outcome            TEXT,                    -- CSP_EXPIRED, CC_CALLED_AWAY, SPREAD_EXPIRED_PROFIT, ...
    days_held                INTEGER,
    n_orders                 INTEGER
);

CREATE INDEX IF NOT EXISTS idx_cycles_symbol ON wheel_cycles(symbol);
CREATE INDEX IF NOT EXISTS idx_cycles_ended_at ON wheel_cycles(ended_at);
CREATE INDEX IF NOT EXISTS idx_cycles_strategy ON wheel_cycles(strategy_id);

-- State transition log (full audit)
CREATE TABLE IF NOT EXISTS state_log (
    id            INTEGER PRIMARY KEY,
    position_id   INTEGER NOT NULL,
    from_state    TEXT,
    to_state      TEXT    NOT NULL,
    reason        TEXT,
    triggered_by  TEXT,                                -- ORDER_FILL, RECONCILER, MANUAL, KILL_SWITCH
    metadata      JSON,
    created_at    DATETIME NOT NULL,
    FOREIGN KEY (position_id) REFERENCES positions(id)
);

CREATE INDEX IF NOT EXISTS idx_state_log_position ON state_log(position_id);
CREATE INDEX IF NOT EXISTS idx_state_log_created_at ON state_log(created_at);

-- Daily LLM screener output, even if not acted upon
CREATE TABLE IF NOT EXISTS candidates (
    id                INTEGER PRIMARY KEY,
    run_date          DATE    NOT NULL,
    symbol            TEXT    NOT NULL,
    score             REAL,
    rank              INTEGER,
    rationale         TEXT,
    ivr               REAL,
    iv_pct            REAL,
    price             REAL,
    bp_required       REAL,
    suggested_strike  REAL,
    suggested_dte     INTEGER,
    raw_llm_response  JSON,
    acted_on          BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_candidates_run_date ON candidates(run_date);
CREATE INDEX IF NOT EXISTS idx_candidates_symbol ON candidates(symbol);

-- Daily regime snapshot
CREATE TABLE IF NOT EXISTS regime_snapshots (
    id              INTEGER PRIMARY KEY,
    snapshot_date   DATE    UNIQUE NOT NULL,
    spy_close       REAL,
    spy_sma_200     REAL,
    spy_above_sma   BOOLEAN,
    vix_close       REAL,
    vix_change_pct  REAL,
    choppiness      REAL,
    regime          TEXT,                              -- BULL_TREND, NEUTRAL, BEAR_TREND, HIGH_VOL
    csps_allowed    BOOLEAN,
    bear_calls_allowed BOOLEAN,                         -- mirrored gate for bear_call_spread (Sprint 10)
    notes           TEXT
);

-- Audit trail of LLM decisions (so we can backtest "would the LLM have helped?")
CREATE TABLE IF NOT EXISTS llm_decisions (
    id             INTEGER PRIMARY KEY,
    decision_type  TEXT    NOT NULL,                   -- SCREEN, NEWS_CHECK, ROLL_ADVISE
    context        JSON,
    model          TEXT,
    response       JSON,
    decision       TEXT,
    confidence     REAL,
    acted_on       BOOLEAN,
    outcome        TEXT,                               -- backfilled later when result known
    created_at     DATETIME NOT NULL,
    tokens_in      INTEGER,
    tokens_out     INTEGER,
    cost_usd       REAL                                -- per-call cost; sum per day for budget gate
);

CREATE INDEX IF NOT EXISTS idx_llm_decisions_type ON llm_decisions(decision_type);
CREATE INDEX IF NOT EXISTS idx_llm_decisions_created_at ON llm_decisions(created_at);

-- IV history for IV Rank/Percentile calc (rolling 52w per symbol)
CREATE TABLE IF NOT EXISTS iv_history (
    id             INTEGER PRIMARY KEY,
    symbol         TEXT    NOT NULL,
    snapshot_date  DATE    NOT NULL,
    iv_30d         REAL,                               -- ATM 30-day IV
    UNIQUE(symbol, snapshot_date)
);

CREATE INDEX IF NOT EXISTS idx_iv_history_symbol_date ON iv_history(symbol, snapshot_date);

-- Per-day per-account anchor row used by the kill switch (rules 8-9 of §8).
-- session_open_equity is captured at the first reconcile-tick of the day so a
-- mid-day restart doesn't lose the daily-P&L baseline.
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

-- Captured option chain at each strategy decision point. The cycle backtester
-- replays decisions against the stored chain instead of re-fetching live data
-- (which would have moved). One row per decision; `contracts` holds the
-- post-filter list the selector evaluated.
CREATE TABLE IF NOT EXISTS chain_snapshots (
    id              INTEGER PRIMARY KEY,
    captured_at     DATETIME NOT NULL,
    symbol          TEXT NOT NULL,
    strategy_id     TEXT,                    -- which strategy was evaluating; nullable for legacy rows
    side            TEXT NOT NULL,           -- "put" | "call"
    underlying_price REAL,
    contracts       JSON NOT NULL,           -- list of OptionContract dicts
    decision_id     INTEGER,                 -- FK to llm_decisions when LLM-driven
    cycle_id        INTEGER,                 -- FK to wheel_cycles when known
    notes           TEXT,
    FOREIGN KEY (decision_id) REFERENCES llm_decisions(id),
    FOREIGN KEY (cycle_id) REFERENCES wheel_cycles(id)
);

CREATE INDEX IF NOT EXISTS idx_chain_snapshots_symbol_date ON chain_snapshots(symbol, captured_at);
CREATE INDEX IF NOT EXISTS idx_chain_snapshots_cycle ON chain_snapshots(cycle_id);

-- Runtime state for strategies (Sprint 13). Tracks auto-disable from the
-- drawdown circuit breaker. Disabled_until=NULL means active; future date
-- means paused-until; past date auto-clears on next check.
CREATE TABLE IF NOT EXISTS strategy_runtime_state (
    strategy_id      TEXT PRIMARY KEY,
    disabled_at      DATETIME,
    disabled_until   DATETIME,
    disabled_reason  TEXT
);
