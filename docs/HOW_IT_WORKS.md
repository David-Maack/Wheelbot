# How WheelBot Trades

A plain-English map of what the bot does and when it does it. Rendered live
at `/how-it-works` on the dashboard.

> **Maintenance rule:** any commit that changes trading behavior — a strategy,
> a gate, an exit rule, a cron — must update this document in the same commit
> (same convention as the go-live runbook). If this page and the code
> disagree, the code won; fix the page.

---

## 1. The big picture

WheelBot runs **two defined-risk options income strategies** on an Alpaca
paper account (virtual book re-based to $12k assumed live funding): a monthly
wheel on a handful of cheap, ownable names, and a same-day SPY/QQQ bull put
spread. It is autonomous tick-to-tick: it picks strikes, places orders,
manages exits, and defends itself with two automatic circuits — no human in
the loop for individual trades.

**2026-10-27 two-strategy cleanup.** A six-month paper audit found that ten strategies, an AI layer and
seven breakers had produced −5.3% equity and no measured edge. The bot now
keeps only the two strategies with any evidence. Each one is judged by one
scorecard (§5), which is the human's go-live / retire rule. Retired:
weekly_wheel, put_spread, narrow_put_spread, bear_call_spread, iron_condor,
pmcc, calendar, spy_swing_opt. Removed: every AI call (screener, universe
refresh, news check, position-news sentry, regime veto, roll advisor), plus
the win-rate floor, expectancy floor, drawdown WARNING tier and the
consecutive-losses pause. Git history keeps all of it.

Humans keep three touchpoints: the dashboard, Discord notifications, and the
Ops MCP (pause / kill / flatten / unflag from Claude Code).

## 2. The tick loop

The bot wakes every **~5 minutes during market hours** (30 min off-hours) and
runs the same sequence:

1. **Reconcile** — read broker truth (orders, positions) and sync the DB:
   record fills, detect expiries and assignment, cancel orphaned resting
   orders, self-heal positions stuck in `*_PENDING`, and run the rule-based
   roll evaluator on challenged wheel shorts.
2. **Kill switch** — check the daily-loss limit (5% of session-open equity)
   and the manual STOP file (`/mnt/wheelbot-storage/STOP`). Tripped = **no
   new orders bot-wide**.
3. **Stale-entry sweep** — cancel entry orders resting > 15 minutes so the
   next tick re-prices them.
4. **Per strategy — manage, then maybe enter** (each strategy's pass is
   exception-isolated: one strategy's data failure logs + notifies and the
   loop moves on, so it can never skip another strategy's closes):
   - **Closes, stops, and rolls run on EVERY tick**, even for a disabled or
     breaker-tripped strategy. Disabling a strategy only stops new entries;
     it never orphans open positions.
   - **New entries** run only if the strategy is enabled, not drawdown-
     DISABLED, not operator-paused, and every gate passes (§4).

## 3. The strategies

| Strategy | Structure | Wants | Key exits |
|---|---|---|---|
| `monthly_wheel` | 30–45 DTE cash-secured puts → assignment → covered calls, on F / NOK / T / VZ / KMI | Cheap, ownable names; IVR ≥ 20; regime allows bullish premium (not PULLBACK / BEAR / HIGH_VOL) | 50% profit, DTE-21, 2× stop, Δ0.55 stop→roll (rule-based roll at Δ0.50, credit-only). **Earnings close (2026-10-27):** a CSP whose remaining hold comes to overlap an earnings window — the date was unknown or moved after entry — is bought back through the normal close path (trigger `earnings`) instead of being flagged and left unmanaged. Covered calls are exempt: the earnings risk sits in the shares |
| `zero_dte` | Same-day-expiry **bull put** spread on SPY and QQQ, entered 10:00–10:15 ET only (rebuilt 2026-10 from the spread-paper-logger delta ladder): $2 wide, Δ0.10–0.15 short, credit ≥ 3% of width, and only while **live spot is above the 20-day SMA of prior closes** (`trend_sma_days`; read at entry — not the regime snapshot's PULLBACK, which uses the prior close and missed 2026-10-16). No bear-call arm: the call side carries no skew premium. **The wing is the stop** (no stop orders — kills the sim-to-live stop-slippage gap) | One round-trip per symbol per day, ≤$200 defined risk each; own daily loss cap ($250 penalized); macro blackout skips event days only — `macro_same_day_only` (a position flat by 15:00 can't see tomorrow's event), with `macro_event_types` still FOMC/CPI/NFP (the ladder has one CPI day, no NFP day) | 50% profit take, then **HARD FLATTEN at 15:00 ET regardless of P&L** (Alpaca rejects expiry-day orders ~15:15/15:30 and force-liquidates 15:30/15:45). Same-day contracts are priced on hours left to the 16:00 ET close. Close limits are **hard-clamped at the wing width**. Every trade writes a dual raw/**penalized** ledger row (`zero_dte_ledger` — penalized concedes half the leg spreads each way + $0.05/share on flatten); its scorecard reads penalized P&L only |

## 4. The entry gauntlet

Every proposed entry passes, in order:

1. **Universe membership** — symbol must be tagged for the strategy in
   `universe.yaml`.
2. **Strategy selector** — finds a contract in the delta/DTE band with
   acceptable bid-ask spread and enough credit. The selector-level IVR floor
   is **regime-aware**: when the latest VIX close is ≥ 25, `ivr_min` relaxes
   by 10 points (IVR reads "low" in sustained high-vol because the rank
   compares against an elevated 52-week high).
3. **Risk gate** (deterministic, `risk/limits.py`). Params = the shared
   `wheel:` block, overlaid by the strategy's own params, overlaid by
   per-ticker overrides (before 2026-10-27 the strategy's params were
   ignored here). Rules: **banned symbols** (GME, AMC — enforced here since
   2026-10-27); buying-power floor; per-position capital cap; global (6) and
   per-strategy concurrent caps; earnings blackout (no opens within 7 days of
   earnings, nor any whose planned hold — expiry minus the strategy's
   time-close — reaches 5 days before earnings); CC strike floor; liquidity;
   regime (CSPs and bull put spreads need `csps_allowed`); macro blackout
   (no opens within 2 days of FOMC/CPI/NFP, from the hand-kept official
   calendar; a strategy may override the event list (`macro_event_types`) or
   narrow it to the event day itself (`macro_same_day_only` — zero_dte
   does)).
4. **Router** — concedes $0.05 slippage on spreads so they fill, refuses
   entries in the last 15 minutes before the close, and cancels/re-places
   limits stale > 15 min. **Close** orders never move a position out of its
   OPEN state, so a close that doesn't cross is re-proposed every tick and
   re-priced through the same stale-replace path.

Exits skip most of this on purpose: a bad regime or blackout must never trap
an open position.

## 5. Self-defense circuits and the scorecard

| Circuit | Trigger | Effect |
|---|---|---|
| Kill switch | −5% equity in a day, or the STOP file | All new orders halt — including roll-generated new shorts (the risk-reducing buyback leg still places). The drawdown trip **latches for the rest of the session**; only the MCP `release_kill_switch` (or the day rolling over) clears it. STOP-file trips release when the file is removed. (The consecutive-losses rule was removed 2026-10-27: a losing streak halted entries, and with no entries the streak could never end.) |
| Drawdown breaker | Strategy's 7-day realized P&L ≤ `auto_disable_drawdown_usd` (−$300) | Strategy disabled 30 days (auto-clears; manual reenable available). A crashing breaker check posts a `risk.monitor_check_failed` Discord alert at most once per 24h |
| Operator pause | MCP `pause_strategy` | No new entries until `reenable_strategy` |

**Scorecard** (`risk/scorecard.py`; `/risk` panel + daily Discord summary).
Never pauses anything — it is the human's rule. Per strategy: closed trades
N, $/trade (zero_dte: **penalized** ledger; wheel: realized cycle P&L), max
drawdown, and fill rate per entry attempt (symbol-day, so cancel/re-price
churn isn't counted as failures). Verdicts: `COLLECTING (n/30)` →
`GO_LIVE_BAR` at N ≥ 30, $/trade > 0, fill rate ≥ 70%; `RETIRE` at N ≥ 30
with $/trade ≤ 0; `FILL_LOW` otherwise. Thresholds: `risk.scorecard`.
Optional per-strategy `scorecard_since` starts the count at a date.

## 6. AI

None in the trading path since 2026-10-27. Over six months the screener,
universe refresh, news check, position-news sentry and regime veto cost ~$46
and changed no trade (3,428 screener candidates, `acted_on = 0`; 18
watchlist proposals, none applied; the regime veto halved sizes that were
already one contract). The `llm_decisions`, `candidates` and `watchlist_*`
tables keep their history.

## 7. Scheduled jobs (LXC host crontab, times in MDT)

The host crontab on CT 105 is authoritative — `crontab -l` to verify.

- **Regime classifier** — daily (SPY vs 200/20-day SMA, VIX → regime flags)
- **Macro calendar refresh** — daily 06:00. Loads `config/macro_calendar.yaml` (official FOMC/CPI/NFP dates; the only source since Finnhub's economic calendar started returning 403) and replaces every future row with it. When any event type has nothing within 60 days the gate keeps blocking on what it has and Discord gets a daily "extend the YAML" alert. Before 2026-10-27 the staleness check looked at refresh time, so the calendar ran out on 09-16 without anyone noticing.
- **Daily summary** — weekday evenings → Discord, including the scorecard
- **History ingest** — weekday evenings (fills `iv_history`; every IVR gate reads it)
- **DB backup** — nightly; keeps 7 daily + 4 weekly backups, and first deletes `chain_snapshots` older than 14 days unless they belong to a cycle or a filled trade (`database.chain_snapshot_retention_days`). `--vacuum` compacts the file (run once with the containers stopped).
- **Parity check** (`wb_parity_check.py`) — weekday 20:45 UTC broker-vs-DB position diff
- **Tastytrade parity harness** (`scripts/parity_run.py`) — only during GO_LIVE_RUNBOOK Stage 2

The screener (weekday 07:00) and universe-refresh (Saturday 07:00) cron lines
must be removed from the host crontab when this version deploys — their
scripts are gone.

## 8. Human touchpoints

- **Dashboard** (port 8889): positions, cycles, orders, performance, risk
  (breakers + scorecard), macro calendar, broker parity, runbook, this page.
- **Discord**: fills on state changes, assignments, MANUAL_INTERVENTION,
  kill-switch, drawdown trips, macro-calendar coverage, daily summary.
- **Ops MCP** (port 8890, bearer-token): read tools (positions, risk,
  performance, regime + calendar, diagnose_symbol) + guarded controls
  (pause/reenable strategy, kill switch, flatten position [dry-run first],
  unflag a MANUAL_INTERVENTION position back to its pre-flag state, refresh
  macro calendar). Every control is audit-logged.

## 9. Data stores in one breath

SQLite (WAL) on the shared volume: `positions` + `orders` + `wheel_cycles`
(the trading truth), `zero_dte_ledger` (raw + penalized 0DTE P&L),
`regime_snapshots`, `macro_events`, `iv_history`, `strategy_runtime_state`
(breaker / pause states), `chain_snapshots` (chains seen at decisions),
`state_log` (position-state audit trail). History-only since 2026-10-27:
`llm_decisions`, `candidates`, `watchlist_runs` / `watchlist_entries`. The
broker is the source of truth for fills; the reconciler keeps the DB honest.
