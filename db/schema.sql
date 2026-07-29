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
    notes           TEXT,
    llm_risk_veto   BOOLEAN,                            -- migration 013: daily reduce-only LLM exception veto
    llm_veto_reason TEXT
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
    strategy_id                  TEXT PRIMARY KEY,
    disabled_at                  DATETIME,
    disabled_until               DATETIME,
    disabled_reason              TEXT,
    drawdown_state               TEXT,        -- TICKET-008: NULL/NORMAL/WARNING/DISABLED
    -- TICKET-009 (migration 011): independent pause columns. drawdown_state
    -- and pause_state can both be set on the same row — bot's runtime gate
    -- applies the more restrictive. Pause never auto-clears.
    pause_state                  TEXT,        -- NULL | 'LOW_WIN_RATE'
    paused_at                    DATETIME,
    paused_reason                TEXT,
    paused_reenable_eligible_at  DATETIME     -- advisory only; not enforced
);

-- TICKET-022 (migration 012): per-contract pricing-parity log between
-- Alpaca paper and Tastytrade sandbox. Populated by scripts/parity_run.py
-- on a cron schedule; consumed by /parity dashboard + daily markdown
-- report. contract_symbol is canonical OCC dense form on both sides.
CREATE TABLE IF NOT EXISTS broker_parity_log (
    id                       INTEGER PRIMARY KEY,
    fetched_at               DATETIME NOT NULL,
    symbol                   TEXT NOT NULL,
    contract_symbol          TEXT NOT NULL,
    alpaca_mid               REAL,
    tasty_mid                REAL,
    mid_diff_pct             REAL,
    alpaca_bid               REAL,
    alpaca_ask               REAL,
    tasty_bid                REAL,
    tasty_ask                REAL,
    asymmetric_liquidity     INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_broker_parity_log_fetched_at
    ON broker_parity_log(fetched_at);

CREATE INDEX IF NOT EXISTS idx_broker_parity_log_symbol_fetched
    ON broker_parity_log(symbol, fetched_at);

-- Macro event blackout (TICKET-007, migration 008). High-impact events
-- (FOMC/CPI/NFP/...) used by risk/limits.py::_rule_macro_blackout to block
-- new short premium whose lifespan crosses them. Populated by the daily
-- cron scripts/refresh_macro_calendar.py from Finnhub + YAML fallback.
CREATE TABLE IF NOT EXISTS macro_events (
    id           INTEGER PRIMARY KEY,
    event_date   DATE     NOT NULL,
    event_type   TEXT     NOT NULL,
    impact       TEXT     NOT NULL,
    description  TEXT,
    fetched_at   DATETIME NOT NULL,
    created_at   DATETIME NOT NULL,
    UNIQUE(event_date, event_type)
);
CREATE INDEX IF NOT EXISTS idx_macro_events_date ON macro_events(event_date);

-- Reusable Discord/notify rate-limit primitive (TICKET-007, migration 009).
-- AlertRateLimitsRepo.try_fire(alert_key, cooldown_hours) returns True iff
-- the alert hasn't fired within the cooldown — and persists the timestamp
-- so process restarts don't reset it.
CREATE TABLE IF NOT EXISTS alert_rate_limits (
    alert_key      TEXT PRIMARY KEY,
    last_fired_at  DATETIME NOT NULL
);

-- Dynamic universe refresh (migration 014). A refresh run proposes per-strategy
-- watchlists; exactly one run is APPLIED at a time and the bot overlays its
-- membership onto universe.yaml at tick time (core/watchlists.py). Runs default
-- to PROPOSED — spec §6: never auto-add tickers without human review.
CREATE TABLE IF NOT EXISTS watchlist_runs (
    id               INTEGER PRIMARY KEY,
    run_date         DATE    NOT NULL,
    status           TEXT    NOT NULL DEFAULT 'proposed',  -- proposed | applied | rejected | superseded | failed
    llm_decision_id  INTEGER REFERENCES llm_decisions(id),
    cost_usd         REAL,
    summary          TEXT,
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

-- 0DTE dual-ledger (migration 015). Alpaca paper fills are systematically
-- optimistic for a 0DTE premium seller (no slippage, NBBO touch fills — see
-- docs/research/0dte_research_2026-07.md §4.1), so every zero_dte trade
-- records BOTH raw and penalized figures: penalized concedes half the summed
-- leg bid-ask spreads on each side, plus stop_slippage_adder on forced
-- flatten exits. Go/no-go reads pnl_penalized only. Rows are written at
-- proposal time; reconciling against actual fills is phase 2.
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
