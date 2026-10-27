-- 2026-10-27 mid-fill test (docs/MID_FILL_TEST.md).
--
-- The monthly_wheel rule changes rest entries at mid and keep a profit-take
-- resting at the 50% target. Whether such orders fill in a real market is
-- the open question: Alpaca's paper simulator fills against the quote with
-- no queue, so a paper fill alone proves little. Each tick, the bot records
-- the live quote for every working single-leg order (phase 'working') and
-- for 30 minutes after a fill (phase 'post_fill'). scripts/fill_report.py
-- turns this into touch rates (did the market ever trade through our
-- limit?), fills-without-touch (paper artefacts) and post-fill drift.

CREATE TABLE IF NOT EXISTS order_quote_log (
    id              INTEGER PRIMARY KEY,
    order_id        INTEGER NOT NULL,
    logged_at       DATETIME NOT NULL,
    phase           TEXT NOT NULL,          -- working | post_fill
    bid             REAL,
    ask             REAL,
    limit_price     REAL,
    FOREIGN KEY (order_id) REFERENCES orders(id)
);

CREATE INDEX IF NOT EXISTS idx_order_quote_log_order
    ON order_quote_log(order_id, logged_at);
