# How WheelBot Trades

A plain-English map of what the bot does, when it does it, and exactly where
the AI is (and is not) involved. Rendered live at `/how-it-works` on the
dashboard.

> **Maintenance rule:** any commit that changes trading behavior — a strategy,
> a gate, an exit rule, an AI touchpoint, a cron — must update this document in
> the same commit (same convention as the go-live runbook). If this page and
> the code disagree, the code won; fix the page.

---

## 1. The big picture

WheelBot runs a portfolio of **defined-risk options income strategies** on an
Alpaca paper account (virtual book re-based to $12k assumed live funding). It
is autonomous tick-to-tick: it picks strikes, places orders, manages exits,
and defends itself with layered risk circuits — no human in the loop for
individual trades.

The AI (Anthropic models) acts as **analyst and gatekeeper, never trigger**:
it ranks candidates, sniffs news, flags regime risk, and proposes watchlist
changes — but every order still has to pass deterministic, code-enforced risk
rules, and the AI has no tool that places an order. Humans keep three
touchpoints: the dashboard, Discord notifications, and the Ops MCP (pause /
kill / flatten / approve-watchlist from Claude Code).

## 2. The tick loop

The bot wakes every **~5 minutes during market hours** (30 min off-hours) and
runs the same sequence:

1. **Reconcile** — read broker truth (orders, positions) and sync the DB:
   record fills, detect expiries and assignment, cancel orphaned resting
   orders, self-heal positions stuck in `*_PENDING`.
2. **Kill switch** — check the daily-loss limit (5% of session-open equity),
   the consecutive-losses counter, and the manual STOP file
   (`/mnt/wheelbot-storage/STOP`). Tripped = **no new orders bot-wide**.
3. **Universe overlay** — refresh each strategy's watchlist membership from
   the currently-APPLIED universe-refresh run (§6); falls back to
   `universe.yaml` when none is applied.
4. **Earnings recheck** — companies confirm earnings dates late; every open
   short-premium position is periodically re-checked and flagged
   `MANUAL_INTERVENTION` (+ Discord) if earnings moved inside its window.
5. **Per strategy — manage, then maybe enter:**
   - **Closes, stops, and rolls run on EVERY tick for EVERY strategy**, even
     disabled or drawdown-paused ones. Disabling a strategy only stops new
     entries; it never orphans open positions.
   - **New entries** run only if every gate passes (§4).

## 3. The strategies

Nine strategies are registered; which are enabled lives in `config.yaml`
(`strategies:` block) and shows live on the dashboard. Universe membership per
strategy comes from `universe.yaml` plus the weekly watchlist overlay.

| Strategy | Structure | Wants | Key exits |
|---|---|---|---|
| `monthly_wheel` | 30–45 DTE cash-secured puts → assignment → covered calls | Cheap, ownable names; IVR ≥ 20 | 50% profit, DTE-21, 2× stop, Δ0.55 stop→roll |
| `weekly_wheel` | 7–14 DTE CSPs, tighter deltas | High-IV liquid weeklies | 50% profit, 2× stop, Δ0.55 stop→close |
| `put_spread` | 30–45 DTE bull put spreads, $5 wide, ≤$500 risk | Mega-caps, IVR ≥ 20 | 50% profit, DTE-21, 2× stop |
| `narrow_put_spread` | Same, $2 wide, ≤$250 risk | Low-price names ($15–40) | Same as put_spread |
| `bear_call_spread` | 30–45 DTE bear call spreads | Weak tape (regime-gated) | 35% profit, DTE-21, 1.5× stop |
| `iron_condor` | Both wings, Δ0.16 shorts, $5 wings | Range-bound, credit ≥ 30% of width | 25% profit, DTE-21 |
| `pmcc` | Deep-ITM LEAP + short OTM calls | Cheap names with liquid LEAPs (≤$1,500) | Short: 50%/1-DTE; long rolls at 60 DTE or Δ < 0.70 (below that it stops acting like stock) |
| `calendar` | Same-strike front/back calls, net debit | **LOW** IV (IVR ≤ 35 — inverted gate) | 25% of debit, force-close front ≤ 2 DTE |
| `spy_swing_opt` | Directional deep-ITM SPY calls/puts (Δ0.90, ~60 DTE) | MTF VWAP/EMA9 signal + 200-SMA gate | Prior-day-level stop, 1.5R target, 7-day time stop |

## 4. The entry gauntlet

Every proposed entry passes, in order:

1. **Universe membership** — symbol must be on the strategy's watchlist and
   not banned (GME, AMC, tier-3 names — hard-coded, AI cannot override).
2. **Strategy selector** — finds a contract in the delta/DTE band with
   acceptable bid-ask spread and enough credit (spreads: ≥25% of width).
3. **Risk gate** (deterministic, `risk/limits.py`) — regime allows the
   direction; global (6) and per-strategy concurrent caps; buying-power floor;
   per-position capital caps; IVR floor/ceiling; earnings blackout (no opens
   within 7 days of earnings, or spanning it); macro blackout (no opens within
   2 days of FOMC/CPI/NFP); **tier-2 screen** — tier-2 symbols need TODAY's
   LLM screener score ≥ 50.
4. **News check** (AI, §6) — headline sniff on wheel CSP entries.
5. **Router** — sizes the order (halved under drawdown WARNING or regime
   veto), concedes $0.05 slippage so it fills, refuses entries in the last 15
   minutes before the close, and cancels/re-places limits stale > 15 min.

Exits skip most of this on purpose: a bad regime or blackout must never trap
an open position.

## 5. Self-defense circuits

| Circuit | Trigger | Effect |
|---|---|---|
| Kill switch | −5% equity in a day, or losses streak, or STOP file | All new orders halt |
| Drawdown breaker | Strategy's 7-day realized P&L ≤ −$300 | Strategy disabled 30 days (auto-clears; manual reenable available) |
| Drawdown WARNING | 7-day P&L ≤ −$150 | New spreads at half size |
| Win-rate floor | < 60% over last 10+ cycles | Strategy paused, **manual** reenable only |
| Regime veto (AI) | LLM flags headline risk the numbers can't see | All entry sizes halved for the day — reduce-only |
| Earnings recheck | Earnings date moves inside an open position's window | Flag MANUAL_INTERVENTION + Discord |

## 6. Where the AI is involved

All calls go through one client with a **$2.00/day budget cap**; every call is
logged to the `llm_decisions` table (dashboard `/decisions`) with cost. Design
stance: **the AI can make the bot more careful, never more aggressive** — and
every AI failure "fails open" to the bot's plain rule-based behavior.

| Job | Model | Cadence | Power | On failure |
|---|---|---|---|---|
| **Screener** | Opus | Daily pre-market (cron) | Scores tier-1/2 universe 0–100; tier-2 names need ≥ 50 to trade that day | Tier-2 entries blocked (safe-closed) |
| **Adversarial screen** | Haiku | Same run | Bull/bear second read of top-3, adjusts scores ±15 | Keeps single-pass scores |
| **News check** | Haiku | Per wheel-CSP entry | proceed / caution / block. Currently **advisory**: only hard "block" cancels (flip `news_check_advisory: false` before live) | Proceeds (fail-open) |
| **Regime veto** | Haiku | Daily (regime cron) | Halves ALL entry sizes for the day — reduce-only by construction | No veto |
| **Universe refresh** | Opus | Weekly, Sat 07:00 MDT (cron) | Proposes watchlist add/keep/drop per strategy, matching candidates to each strategy's profile (rich IV → put spreads, low IV → calendars, cheap LEAPs → PMCC, …). Candidates = the curated universe **plus a market-wide discovery scan**: the top ~100 most-active US stocks (Alpaca screener), quant-gated, chain-tradability-checked, capped at 25 new names per run. **Human must approve** (MCP `approve_watchlist`); code-enforced guardrails: open/pinned symbols undroppable, ≤2 adds + ≤2 drops per strategy, min 3 symbols, banned/tier-3 never re-addable, discovered adds enter at tier 2 (need a daily screener score to trade) | Last-good watchlist stands; a broken screener just shrinks the pool back to hand-curated |
| **Roll advisor** | Opus+Haiku ensemble | — | **Disabled** (`llm_roll_advisor_enabled: false`) until 3+ months of data | — |

What the AI can **not** do: place or size an order upward, override the risk
gate, touch banned/tier-3 names, apply its own watchlist (unless
`auto_apply: true`, which ships false), or spend past the daily budget.

## 7. Scheduled jobs (LXC host crontab, times in MDT)

The host crontab on CT 105 is authoritative — `crontab -l` to verify.

- **Screener** — weekday pre-market (Opus ranking → `candidates` table)
- **Regime classifier + LLM veto** — daily (SPY vs 200-SMA, VIX → regime flags)
- **Macro calendar refresh** — daily 06:00 (Finnhub, falls back to the YAML)
- **Daily summary** — weekday evenings → Discord
- **Universe refresh** — Saturday 07:00 → proposal → Discord + MCP approval
- **DB backup / history ingest** — per crontab

## 8. Human touchpoints

- **Dashboard** (port 8889): positions, cycles, candidates, orders, LLM
  decisions, performance, risk circuits, macro calendar, runbook, this page.
- **Discord**: fills on state changes, assignments, MANUAL_INTERVENTION,
  kill-switch, drawdown trips, budget exceeded, daily summary, weekly
  universe-refresh proposals.
- **Ops MCP** (port 8890, bearer-token): read tools (positions, risk,
  performance, decisions, regime, watchlists, diagnose_symbol) + guarded
  controls (pause/reenable strategy, kill switch, flatten position [dry-run
  first], refresh macro calendar, approve/reject watchlist). Every control is
  audit-logged.

## 9. Data stores in one breath

SQLite (WAL) on the shared volume: `positions` + `orders` + `wheel_cycles`
(the trading truth), `candidates` (screener output), `llm_decisions` (every AI
call + cost), `regime_snapshots`, `macro_events`, `iv_history`,
`strategy_runtime_state` (breaker states), `watchlist_runs`/`watchlist_entries`
(universe refresh), `state_log` (position-state audit trail). The broker is
the source of truth for fills; the reconciler keeps the DB honest.
