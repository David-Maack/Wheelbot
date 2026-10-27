# Mid-fill test — monthly_wheel

**Question:** do monthly_wheel's patient orders fill in a real market, how fast, and what does waiting cost?

- **Entries:** a sell at mid, re-pegged to the new mid every 15 minutes.
- **Resting profit-take:** a buy-to-close sitting at 50% of the premium from the moment the put fills.

## Why it matters

A 2018–2026 backtest of the exact rules (2026-10-27, kept outside the repo) found that execution is most of the edge:

| monthly_wheel, 7-DTE close, PULLBACK off | Fills pay half the spread | Patient fills (entries and profit-takes at mid) |
|---|---|---|
| 20 cheap names, $/trade | +$5.83 | +$9.53 |
| The bot's 5 names, max 2 open, $/trade | +$1.55 | +$3.02 |

The patient column assumes every resting order at mid fills. That is the backtest's most optimistic assumption, and this test checks it.

## What the branch changed (feat/wheel-rule-changes)

**Rules (config.yaml, monthly_wheel params):**

- `time_close_dte: 7` (was 21).
- `pullback_allows_csps: true`: PULLBACK no longer blocks the wheel's CSPs; HIGH_VOL and BEAR_TREND still do.
- `resting_profit_take: true`: after a short fills, the bot keeps a BUY_TO_CLOSE resting at the target.
  - Alpaca option orders are DAY-only, so it is re-placed every session.
  - Its client id starts with `wb-rest-`, and it is never stale-replaced.

**Urgent closes:**

- Stops, time, earnings and delta closes, and rolls cancel the resting order and wait for the cancel to confirm. Only then do they place their own close.
- They now pay the **ask**. Before this change every wheel close, stops included, was priced at mid, because the close path set bid = ask = mid. A mid-priced stop on a rising option can sit unfilled, and an in-flight close is not re-proposed.
- Closes are no longer blocked by the liquidity (spread-width) rule.
- A sweep cancels resting orders whose position is no longer open (assigned, closed elsewhere, flagged).

**Data for this test:**

- `orders.raw_request.wb_quote`: the bid, ask and mid each single-leg order was priced from, and whether it was resting.
- `order_quote_log` (migration 017): every tick in market hours, the live quote of each working monthly_wheel order (`working`), and of each order filled in the last 30 minutes (`post_fill`).
- `scripts/fill_report.py`: the summary below.

## What paper can and cannot tell us

- **Alpaca paper fills against the quote with no queue and no size limit.** A resting order "fills" as soon as the quote reaches it, possibly without that. A real resting order fills only when someone trades with it. So a paper fill rate is an upper bound, and a paper fill with no recorded touch is a simulator artefact.
- **The quotes may not be the real market.** Alpaca's free options data is the *indicative* feed, not the OPRA NBBO, and the bot requests the account default. Phase 0 checks which one we have.
- **So the real fill rate needs real orders.** Paper (Phase 1) proves the mechanics and gives an upper bound. The live probe (Phase 2) measures the fill rate.

## Metrics (`python -m scripts.fill_report`)

The report splits orders into three classes:

- **entries:** SELL_TO_OPEN.
- **resting:** `wb-rest-` buy-to-closes.
- **urgent:** every other buy-to-close.

| Metric | Meaning |
|---|---|
| `fill_rate_per_order`; entries also `fill_rate_per_symbol_day` | Share of orders that filled. A re-pegged entry is several orders but one symbol-day. |
| `minutes_to_fill_median` / `_p90` | Placed → filled. |
| `spread_position_median` | Where fills landed in the spread they were priced from. 0 = paid the whole spread, 0.5 = mid. |
| `touch_rate` | Share of taped orders whose limit the quote reached while they worked (sell: bid ≥ limit; buy: ask ≤ limit). |
| `fills_without_touch` | Paper fills the quote never justified. |
| `post_fill_drift_usd_mean` | How far the mid moved against us within 30 min of a fill, in $/contract. This is the price of being filled by someone who knew better. |
| resting: `positions_closed_by_resting_fill`, `cancelled_for_urgent_close` | How profit-takes ended. |
| urgent: `cost_vs_mid_usd_mean`, `by_trigger` | What paying the ask costs, by close reason. |

## Phase 0 — before deploying (one day)

1. **Deploy order.** PRs #7 and #8 merge first, after HOOD and INTC are closed. This branch is based on #8 and deploys after them through the same steps (`git checkout main && git pull`, `docker compose up -d --build`). Migration 017 applies on start.
2. **Feed probe:**
   ```
   docker exec wheelbot python -m scripts.fill_report --probe-feed --contract <any wheel OCC symbol>
   ```
   - **OPRA refused:** the tape's touch rates are indicative only. Either accept that for Phase 1 and say so in the write-up, or subscribe to OPRA for the test window.
   - **OPRA accepted:** point the bot's quote requests at OPRA (a one-line change in `platforms/alpaca_broker.py`) before Phase 1.

## Phase 1 — paper (Alpaca, 4–6 weeks)

**Correctness checks.** Each must hold, read from the logs and the orders table:

- **Never two filled closes on one short.** Query: FILLED BUY_TO_CLOSE grouped by `contract_symbol` and cycle, count > 1, should return no rows.
- **Every urgent close on a position with a resting order logs `router_resting_cleared` first.** `router_resting_not_cleared` may delay a close by at most one tick. If it happens twice in a row on one position, stop and investigate.
- **Resting orders reappear each session.** Each morning there is one working `wb-rest-` order per open short.
- **The orphan sweep works.** After an assignment or manual close, `router_orphan_resting_cancelled` fires and no `wb-rest-` order stays working on a non-open position.
- **Stops fill at once.** Urgent closes are priced at the ask, so `minutes_to_fill_median` for urgent closes should be about 0 on paper.

**Measurements.** Run `fill_report` weekly and note the paper fill rate, touch rate and fills without touch for entries and resting orders.

**Sample size.** At the backtest's pace the bot's own setup (5 names, max 2 open) opens about 3 trades a month. Phase 1 will not reach 20 resting outcomes in 6 weeks, and it doesn't need to: its job is mechanics plus a touch-rate baseline.

**Faster alternative (separate change).** To get statistics from paper, add a paper-only `wheel_fill_probe` strategy: the same params on the 15 extra backtest names, its own scorecard, `max_concurrent` around 8, and `account.max_concurrent_total` raised to match. The backtest pace for that set is about 14 entries a month. It touches `universe.yaml` and the global cap, so it is not in this branch.

**Exit to Phase 2:**

- every correctness check clean for the last 4 weeks;
- at least 5 resting profit-take outcomes and 5 entry symbol-days observed.

## Phase 2 — live probe (first weeks of GO_LIVE_RUNBOOK Stage 3, Tastytrade)

**Setup:**

- 1 contract per trade on the cheapest wheel names: F, NOK and T, about $900–$2,500 of collateral each.
- At most 2 open, with the Phase 1 rules unchanged.
- Counts begin at the Stage 3 start date.

**Blocking issue.** Stage 3's `max_position_pct_of_account: 5` allows about $150 a position on a $3k quarter-size account, which blocks every wheel contract. Decide this before Stage 3: fund the account to at least $3k and give monthly_wheel 30%, or keep the wheel out of Stage 3 entirely.

**Sample.** The first 20 entry symbol-days and the first 20 positions whose quote reached the profit target. With 20 observations, a measured 70% has a 95% interval of roughly 46–88%. That's enough to tell "usually fills" from "rarely fills", nothing finer.

**Decision rules:**

| Result after 20 | Decision |
|---|---|
| Entries: ≥ 70% of symbol-days fill by the close, at a median spread position ≥ 0.4 | Keep mid entries |
| Entries: < 70% fill | Add a per-strategy concession (sell at mid − 1 tick; new knob, small change) and re-measure 20 |
| Resting: ≥ 60% of the positions that touched the target filled at it within that session or the next | Keep `resting_profit_take: true` |
| Resting: < 60% | `resting_profit_take: false`: profit-takes cross at the ask as before. Expect the backtest's "half the spread" rows |
| Post-fill drift (either class) > half the median quoted spread | Fills are being picked off. Treat as a fail for that class even if the fill rate passed |
| Any double fill, or an urgent close held back more than one tick | Stop: pause the strategy, flatten if needed, fix before resuming (runbook §3) |

**Rollback is config-only:** `resting_profit_take: false`, `time_close_dte: 21`, `pullback_allows_csps: false`. The quote tape can stay on; it only reads.
