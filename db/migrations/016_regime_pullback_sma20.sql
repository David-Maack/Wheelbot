-- 2026-08-26 regime-sensitivity fix: PULLBACK regime (defensive-only).
--
-- The 200d-SMA-only ladder stamped BULL_TREND for 64 straight days through
-- the entire Aug-2026 decline from the ATH — SPY would have needed another
-- -7.5% before BEAR_TREND fired. classify_regime gains an intermediate gear:
-- SPY below its 20d SMA (with a small hysteresis margin) while above the
-- 200d => PULLBACK, which blocks NEW bullish-premium entries (csps_allowed
-- = 0) and — per the defensive-only decision — keeps bear_calls_allowed = 0
-- until that side earns its way in with paper evidence. Over 5y this state
-- fires on ~19% of days; forward returns there are two-sided, so it is a
-- risk-state, not a direction forecast.
--
-- This migration only adds the observability column; the classification
-- logic lives in risk/regime.py.

ALTER TABLE regime_snapshots ADD COLUMN spy_sma_20 REAL;
