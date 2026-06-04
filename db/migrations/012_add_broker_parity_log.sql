-- TICKET-022: per-contract pricing-parity comparison between Alpaca paper
-- and Tastytrade sandbox. Populated by scripts/parity_run.py on a 6-times-
-- daily cron during NYSE hours. Consumed by the /parity dashboard page and
-- by the daily markdown report under /mnt/wheelbot-storage/parity_reports/.
--
-- Acceptance thresholds (per docs/GO_LIVE_RUNBOOK.md Stage 2):
--   mean mid_diff_pct per symbol  < 2%
--   worst mid_diff_pct per symbol < 5%
--   zero rows where Alpaca bid+ask > 0 but Tastytrade missing/zero
--
-- contract_symbol is the canonical OCC dense form (21 char). Both feeds
-- are normalized through the same helper before insert so the join
-- between brokers is deterministic — see scripts.parity_run._normalize_occ.

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
    asymmetric_liquidity     INTEGER NOT NULL DEFAULT 0  -- 1 if Alpaca has bid+ask>0 but Tastytrade missing/zero
);

CREATE INDEX IF NOT EXISTS idx_broker_parity_log_fetched_at
    ON broker_parity_log(fetched_at);

CREATE INDEX IF NOT EXISTS idx_broker_parity_log_symbol_fetched
    ON broker_parity_log(symbol, fetched_at);
