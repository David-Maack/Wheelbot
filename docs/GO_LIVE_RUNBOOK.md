# WheelBot Go-Live Runbook

This runbook is the sequenced path from Alpaca paper to full live capital. It defines five stages (0 through 4), the entry criteria for each, what must run and what must not, daily cadence, stop conditions, and exit criteria. It exists because shipping all eight sprints is necessary but not sufficient: the bot needs observed, dated evidence under each progressively stricter regime before the next regime is unlocked.

Writing this runbook is one job. Following it is a different, separate job done by you, the operator, day by day, decision by decision. Treat the two as distinct: the runbook does not press its own buttons.

The minimum elapsed time from Stage 1 entry to Stage 4 entry is approximately 90 days by design (30d Stage 1 + 60d Stage 3, with Stage 2 parity running in parallel with Stage 1). Expect 6 months realistically before Stage 4. The numbers are floors that let each stage accumulate enough closed cycles, P&L history, and incident-free days for the next stage's entry criteria to be honestly satisfied. If a stage takes longer, that is the runbook working. The only acceptable way to move faster is to discover that an entry criterion was wrong and fix the criterion in writing before relaxing it.

## Stage 0 — Current state (Alpaca paper)

Stage 0 is where the bot lives today: ticking on Alpaca paper, all Tier-1 safety features shipped (TICKET-001 through TICKET-009 and TICKET-020), `intelligence.news_check_advisory: true` (decisions logged but not enforced at the router), and every migration through 011 applied.

Nothing in Stage 0 is aspirational. Before Stage 1 can begin, the bot must be demonstrably clean against the checklist below. "Clean" means observed for a continuous window, not a single green snapshot.

### Stage 0 clean checklist

- Bot is ticking healthily — `scripts/run_bot.py` running under the service manager, no restart loops, heartbeat current.
- Dashboard reachable — `/risk`, `/orders`, `/decisions`, `/positions`, `/performance`, `/macro` all render and refresh.
- Zero unresolved `MANUAL_INTERVENTION` rows. Open the reconciler view, confirm count is zero. If non-zero, resolve before proceeding; do not enter Stage 1 with an open intervention.
- `config/config.local.yaml` exists with intended overrides (see below). The file does not currently exist — only `config.local.yaml.example` ships in the repo. Stage 0 creates it.
- `python -m scripts.preflight_live` exits 0 with no warnings you have not explicitly decided to accept.

### `config/config.local.yaml` entries to create

Create the file with the following overrides:

- `risk.consecutive_losses_pause: 5` — raise from the enforced live default of 3 to the target value.
- `wheel.open_interest_min: 0` — Alpaca paper returns null for this field on less-liquid wheel underlyings; the live default of 500 starves the wheel of CSP candidates on F/T/VZ/KMI. Restore to 500 at Stage 3 broker swap.
- `wheel.volume_min: 0` — same Alpaca-paper data-gap rationale; restore to 100 at Stage 3.
- `risk.daily_loss_kill_switch_pct: 5` — pin explicitly at 5% so a future edit to `config.yaml` cannot silently change it. (The `.example` file shows 3%; that value is not what is enforced.)
- `intelligence.llm_screener_enabled: true` — pin enabled. (The `.example` shows `false`; that is also not what is enforced.)

Do not add any other overrides at Stage 0. New overrides arrive at the stage that requires them.

## Stage 1 — news_check enforcement flip (30 days minimum, on Alpaca paper)

Stage 1 flips `news_check` from advisory to enforcing at the router. The bot stays on Alpaca paper. No capital change, no broker change, no strategy change. The single variable under test is whether enforcement produces a sane proceed/caution/block mix and does not over-block live order flow.

### Config changes

- `intelligence.news_check_advisory: true` → `false` in `config/config.local.yaml`.
- No other changes.

### What's running

- All four strategies as configured in Stage 0.
- Screener (`intelligence.llm_screener_enabled: true`), `intelligence.daily_budget_usd: 2.00`.
- All Tier-1 safety features.
- `news_check` now enforcing: `proceed` passes, `caution` halves size, `block` cancels.

### What's NOT running

- No move off Alpaca paper.
- No Tastytrade parity work yet (that is Stage 2; see TICKET-022).
- No capital changes.

### Entry criteria

Stage 0 clean checklist fully satisfied, and the 30 days of Stage 0 observation include at least one full week where `news_check` advisory output looked sane in `/decisions`. Strategies still in `insufficient_data` (fewer than 10 closed cycles) are allowed through this gate — the asymmetry between 1→2, 2→3, and 3→4 is documented at the top of §2.

### Daily cadence

- Open `/decisions`, filter to `news_check`. Review the proceed/caution/block mix for the last 24 hours. Expect proceed-dominant with occasional caution and rare block.
- Confirm fail-open paths logged as `skipped:*` are not silently dominating the decision stream.
- Check `/risk` for any new MANUAL_INTERVENTION rows; resolve before next tick.
- Spot-check one blocked symbol per day against headlines to confirm the model's call.

### Stop conditions (stage-1-specific)

- Over-blocking: `block` rate exceeds 20% of decisions over any rolling 7-day window, or any single trading day shows zero placed orders attributable to `news_check` blocks. Revert `news_check_advisory: true`, investigate, do not advance.
- Fail-open detection: `skipped:*` outcomes (no_source, news_unavailable, budget, llm_error) exceed 30% of decisions over any rolling 3-day window. This means enforcement is nominal but the gate is structurally bypassed. Fix the upstream cause before continuing to count Stage 1 days.

Generic stop conditions (kill-switch trip, consecutive losses, STOP file, MANUAL_INTERVENTION backlog) live in §3 Table A and apply at every stage.

### Exit criteria

- Minimum 30 calendar days with `news_check_advisory: false` and neither stop condition triggered.
- No unresolved MANUAL_INTERVENTION rows at the moment of advancing.
- `/decisions` shows a plausible proceed/caution/block distribution sustained, not a single clean week followed by drift.

## Stage 2 — Tastytrade sandbox parity (30 days, parallel with Stage 1)

**Purpose.** Confirm Tastytrade sandbox quotes and (where the sandbox supports it) fills track Alpaca paper closely enough to trust the broker swap planned at Stage 3. This stage runs in parallel with Stage 1 — Alpaca paper continues to be the live execution path; Tastytrade is read-only/sandbox during this window.

**Depends on.** TICKET-022 (parity diff harness) is shipped as of 2026-06-04. The cron line below must be installed on the LXC. If Tastytrade sandbox credentials have not been provisioned (see §5 Part A), the harness exits with an actionable error message and writes no rows.

**Operating procedure.**
- Install the cron: `35 14-19 * * 1-5 cd /opt/wheelbot && docker exec wheelbot python -m scripts.parity_run` (six samples/day during NYSE hours).
- Each run fetches option chains for every ticker with a non-empty `strategies:` list in `universe.yaml`, joins by canonical OCC symbol on both sides, computes per-contract mid-price diff, and writes rows to `broker_parity_log`.
- Review the daily markdown report at `/mnt/wheelbot-storage/parity_reports/YYYY-MM-DD.md` — header, per-symbol summary, acceptance check, top-10 outliers.
- Watch the `/parity` dashboard page for the rolling 7-day trend chart + per-symbol summary table + Stage 2 acceptance pass/fail.

**Acceptance criteria (all must hold across the 30-day window).**

- Mean mid-price diff per symbol < 2%.
- Worst-case mid-price diff per symbol < 5%.
- Zero contracts where Alpaca shows liquidity (both bid > 0 and ask > 0) but Tastytrade sandbox does not.

**Stop condition.** Any one of the three acceptance criteria missed on any day inside the 30-day window. Stop means: do not advance to Stage 3, open a ticket against the parity gap, and restart the 30-day clock once the gap is resolved.

## Stage 3 — Tastytrade live with 1/4 capital (60 days)

This is the highest-stakes transition in the runbook: first real money on a new broker. The §5 pre-flight checklist is mandatory before flipping the broker. Do not skip it because Stage 1 and Stage 2 went clean.

### Entry criteria

- Stage 1 has completed its 30-day window with zero unresolved stop conditions.
- Stage 2 has completed its 30-day window and all three acceptance criteria held.
- Every active strategy has ≥10 closed cycles at ≥60% win rate, **or** is explicitly disabled in config. Strategies still in `insufficient_data` are not allowed through this gate — disable them or wait.
- §5 Part B manual pre-flight checklist signed off within the last 24 hours.
- TICKET-022 parity harness still green for the trailing 7 days.

### Config changes

| Key | From | To |
|---|---|---|
| `account.broker` | `alpaca_paper` | `tastytrade` |
| `account.max_concurrent_total` | 14 | 4 |
| per-strategy `max_position_pct_of_account` | base (30) | 5 |
| `wheel.open_interest_min` | 0 (paper override) | 500 (remove override) |
| `wheel.volume_min` | 0 (paper override) | 100 (remove override) |

Leave `risk.daily_loss_kill_switch_pct` at 5%, `risk.consecutive_losses_pause` at 5 (already set in `config/config.local.yaml` at Stage 0), `intelligence.daily_budget_usd` at 2.00, and `intelligence.news_check_advisory` at `false`. Do not relax any other risk gate for this stage.

### Daily cadence

- Discord daily summary verified received each morning. If missing, treat as a stop-and-investigate event before market open.
- Review `/positions`, `/orders`, and `/risk` daily.
- Review `/decisions` weekly, or same-day on any unusual fill.
- `/performance` reviewed weekly against Stage 1 baseline.

### Stage-3-specific stop conditions

Generic stop conditions live in §3. The following are specific to this stage:

- First 5 live Tastytrade fills: `fill_price` vs `limit_price` divergence outside the band observed during Stage 2 parity. Flatten the affected position, pause the strategy, investigate before resuming.
- Any `MANUAL_INTERVENTION` flag not cleared within 24 hours of being raised.
- Two strategies entering `paused` state within the same calendar week, for any reason (drawdown, win-rate floor, consecutive-losses, or operator action).

### Rollback

Use the §4 rollback decision matrix. Default behavior on doubt is flatten. Specifically for Stage 3, any position on a strategy being rolled back due to a suspected logic error in that strategy is flattened, not held. Revert `account.broker` to `alpaca_paper` and restart from Stage 2 once the cause is understood.

### Exit criteria (advance to Stage 4)

- 60 calendar days elapsed with zero unresolved stop conditions.
- No `MANUAL_INTERVENTION` outstanding.
- No strategy in `paused` state at the moment of transition; if any is paused, either re-enable it via `python -m scripts.reenable_strategy --strategy <name> --reason "<reason>"` or leave it disabled and document that choice.
- Realized win rate and drawdown per strategy within the bounds enforced by `min_win_rate_pct: 60`, `auto_disable_drawdown_usd: -300`, `drawdown_warning_usd: -150`.

## Stage 4 — Tastytrade live at full size

Steady state. Caps come back up to the values the system was designed around. Review cadence does **not** relax — same daily Discord summary, same daily `/positions` `/orders` `/risk` review, same weekly `/decisions` `/performance` review as Stage 3.

### Entry criteria

- Stage 3 exit criteria all met.
- §5 Part B manual pre-flight re-run within the last 24 hours before lifting caps.
- No open `MANUAL_INTERVENTION`.

### Config changes

| Key | From | To |
|---|---|---|
| `account.max_concurrent_total` | 4 | 14 |
| per-strategy `max_position_pct_of_account` | 5 | base (30) |
| `account.broker` | `tastytrade` | `tastytrade` (unchanged) |

All other risk values stay where they were at Stage 3: `risk.daily_loss_kill_switch_pct` 5%, `risk.consecutive_losses_pause` 5, `wheel.open_interest_min` 500, `wheel.volume_min` 100, `auto_disable_drawdown_usd` -300, `drawdown_warning_usd` -150, `min_win_rate_pct` 60, `min_closed_cycles` 10, `pause_duration_days` 14.

### Stop conditions

Generic stop conditions from §3 apply unchanged. There are no Stage-4-specific stop conditions — the only thing that changed from Stage 3 is the cap. Lifting the cap is not a license to ease off monitoring.

### Rollback

First rollback step in Stage 4 is to drop `account.max_concurrent_total` back to 4 and per-strategy `max_position_pct_of_account` back to 5 — i.e. return to Stage 3 posture — before considering a broker revert or a full flatten. Use the §4 matrix from there.

### Steady-state notes

There is no Stage 5. This is steady state. Re-run the runbook from Stage 0 only after a major code change — new strategy added, broker swap, reconciler rewrite, sizing logic change, kill-switch logic change, or any change to the gates enforced in `scripts/preflight_live.py`. Minor bug fixes and dependency bumps do not require a full re-run; document them in the change log and continue.

## §2 — Entry criteria per stage

Entry-criterion strictness is asymmetric across the four stage transitions, on purpose. The 1→2 and 2→3 gates are lenient: strategies with fewer than 10 closed cycles (the `insufficient_data` branch of `risk/win_rate_floor.py`) are allowed through. They are not yet eligible for the win-rate floor, but the drawdown breaker in `risk/auto_disable.py` (warning at -$150, disable at -$300 rolling 7d) keeps a leash on dollar bleed in the meantime.

The 3→4 gate is strict. Every active strategy must show ≥10 closed cycles AND ≥60% win rate, OR be explicitly marked `enabled: false` in `config/config.yaml` before you advance. "Active but data-thin" is not acceptable at the live boundary — the strategy either has proven itself or it is turned off. There is no third option.

Every checklist below is and-gated. Every item must be true. Do not reason "four of five is good enough" — the checks are written to gate failure modes that have already bitten this bot in paper. If one item is red, the answer is "not yet", not "close enough".

### Enter Stage 1

- [ ] `config/config.local.yaml` exists, committed to the LXC host at `/opt/wheelbot/config/config.local.yaml`, and overrides `account.broker: alpaca_paper`.
- [ ] `risk.stop_file_path` resolves to `/opt/wheelbot/STOP` and the parent dir is writable by the container user.
- [ ] `scripts/preflight_live.py` exits 0 (or any non-zero is acknowledged in the Stage 0 follow-up ticket).
- [ ] `docker compose up -d --build wheelbot` succeeded on the LXC and `docker logs wheelbot --tail 50` shows the `bot_migrations` checkpoint at status=ok.
- [ ] `docker exec wheelbot python -m scripts.db_health` exits 0.
- [ ] `bash scripts/migrate_check.sh` reports no drift between host `db/migrations/` and the container.
- [ ] Dashboard `/risk` is reachable on `127.0.0.1:8889` behind basic auth and renders the kill-switch panel without errors.
- [ ] `intelligence.daily_budget_usd` is `2.00` and `intelligence.news_check_advisory` is `true` (paper defaults).
- [ ] `account.max_concurrent_total` is `14` (testing) — the per-strategy caps will bind.
- [ ] Discord webhook posts a test message from the bot's first heartbeat.

### Enter Stage 2

- [ ] Stage 1 has been live for at least one full trading week (5 sessions) with no `MANUAL_INTERVENTION` flags older than 24h. (Stage 2 is read-only sandbox work and may begin overlapped with Stage 1 — it does not require Stage 1's 30-day window to be complete.)
- [ ] No `reconciler_on_cancel` storms, no broker-divergence spikes — `/orders` rejection rate stays below ambient noise.
- [ ] At least one closed cycle has appeared in `/performance` for at least one strategy (proves the round trip).
- [ ] Strategies still in `insufficient_data` (<10 closed cycles) ARE allowed through — the drawdown breaker continues to protect them; do not block on win-rate sample size at this transition.
- [ ] Daily summary cron (`scripts.daily_summary`) has posted to Discord on every trading day since Stage 1 entry.
- [ ] `kill_switch_armed` has not flipped to True from a real cause (not test-engaged) at any point during Stage 1.
- [ ] You have read TICKET-022's parity harness contract and have an account at the Tastytrade sandbox provisioned.

### Enter Stage 3

- [ ] Stage 1 has completed its full 30-day window with zero unresolved stop conditions.
- [ ] Stage 2 parity harness (TICKET-022) has run for at least its full 30-day window with mean mid-price diff <2% and worst-case <5% across the active universe. No Alpaca-liquid / Tastytrade-illiquid contracts remain on the live universe.
- [ ] Zero new `MANUAL_INTERVENTION` flags during the Stage 2 window.
- [ ] No reconciler-mismatch events above one per day; none unresolved by the next session.
- [ ] Strategies in `insufficient_data` ARE still allowed through at this transition — Stage 3 is a parity confidence gate, not a win-rate gate.
- [ ] `intelligence.daily_budget_usd: 2.00` has not been hit on any day in the Stage 2 window (no LLM budget exhaustion).
- [ ] `account.broker` flip plan is written and reviewed (which adapter, which secrets, which kill-switch dry-run first).

### Enter Stage 4

- [ ] Stage 3 has been live on `tastytrade` for at least 60 calendar days.
- [ ] Every strategy in `config.yaml` with `enabled: true` has ≥10 closed cycles at ≥60% win rate as reported by `risk/win_rate_floor.py`. No `insufficient_data` strategies are allowed through this gate.
- [ ] Any strategy that does not meet the win-rate bar has been explicitly toggled to `enabled: false` in `config.yaml` and the change is committed.
- [ ] No strategy is in `DISABLED` (drawdown) or `PAUSED_LOW_WIN_RATE` state in `strategy_runtime_state`. Run `python -m scripts.reenable_strategy --list` and confirm the output is "No strategies are currently auto-disabled or paused."
- [ ] `risk.consecutive_losses_pause` is set to `5` in `config/config.local.yaml`.
- [ ] `wheel.open_interest_min: 500` and `wheel.volume_min: 100` confirmed in the effective merged config (paper override removed at Stage 3 broker swap).
- [ ] `intelligence.news_check_advisory` has been `false` continuously since Stage 1.
- [ ] `intelligence.llm_screener_enabled: true` confirmed in the effective merged config.
- [ ] No `kill_switch_armed=True` event during the last 5 Stage 3 sessions from a real cause.
- [ ] You can describe, from memory and without checking, what the rollback procedure (§4) does at each step.

## §3 — Stop conditions

Stops fall into two categories. **Automated breakers** are wired into code paths the bot executes every tick; they trip without you. **Manual judgment thresholds** are tighter than the automated bar and require you to engage the stop file yourself. The automated breakers are the floor, not the ceiling — if any of the manual triggers in Table B fires, you engage `risk.stop_file_path` (default `/opt/wheelbot/STOP`) immediately, even if no automated breaker has tripped.

The distinction matters because the automated breakers are calibrated to be reluctant: they wait for confirmed signal so they don't thrash. Your job is to stop earlier when the *pattern* is clear before the threshold is met.

### Table A — Automated breakers

| Trigger | Configured threshold | Where wired | Auto-recovery? |
|---|---|---|---|
| Per-strategy rolling 7d realized P&L ≤ -$300 | `auto_disable_drawdown_usd: -300` (per strategy in `config.yaml`) | `risk/auto_disable.py` `check_and_apply` (called from `scripts/run_bot.py` after each strategy iteration) | 30-day auto-reset on the `disabled_until` column, OR earlier via `python -m scripts.reenable_strategy --strategy <id> --reason "..."`. P&L recovery alone does NOT downgrade DISABLED. |
| Per-strategy rolling 7d realized P&L in (-$300, -$150] | `drawdown_warning_usd: -150` (per strategy) | `risk/auto_disable.py` same path; sets `DrawdownState.WARNING` (spread entries halved, wheels stay qty=1) | Yes — clears silently to NORMAL when 7d P&L crosses back above -$150. No alert on recovery. |
| Win rate <60% over last 10 closed cycles | `risk.win_rate_floor.min_win_rate_pct: 60`, `min_closed_cycles: 10` | `risk/win_rate_floor.py` `check_and_apply` | Manual only — `python -m scripts.reenable_strategy --strategy <id> --reason "..."`. `pause_duration_days: 14` is advisory display only; the bot never reads it. |
| Daily account drawdown >5% from session-open equity | `risk.daily_loss_kill_switch_pct: 5` (pinned in `config/config.local.yaml`) | `execution/kill_switch.py` Rule 8, anchored against `daily_state.session_open_equity` | Manual — delete `risk.stop_file_path` via `rm /opt/wheelbot/STOP` or POST `/risk/manual_stop` action=release from the dashboard. The anchor rolls forward at the next session prime. |
| Consecutive losing cycles | `risk.consecutive_losses_pause: 5` (in `config/config.local.yaml`; base `config.yaml` value is 3) | `execution/kill_switch.py` Rule 9, counted via `_consecutive_losses` over `wheel_cycles.final_pnl < 0` | Manual reenable — same path as the daily drawdown trip (release the STOP file once you have decided what to do). |

### Table B — Manual judgment thresholds

| Trigger | Manual threshold | Why tighter than the automated breaker |
|---|---|---|
| `MANUAL_INTERVENTION` flag not cleared within 24h | Any single position stuck in `MANUAL_INTERVENTION` for >24h | No automated equivalent. The bot flags but never auto-resolves; a stuck flag means the operator queue is broken, not the bot. |
| Two strategies entering PAUSE (drawdown DISABLED or win-rate LOW_WIN_RATE) within the same week | 2 strategies, 7 calendar days, any combination of breaker types | Systemic signal — one strategy bleeding is local; two in a week says the regime has turned or a shared dependency (universe, data feed, screener) is broken. The automated breakers fire per-strategy and never look across the portfolio. |
| Reconciler mismatch events ≥3 in a single day | 3 mismatches between broker view and local view in 24h | No automated stop. `reconciler_on_cancel` and divergence checkpoints are logged but never trigger a halt. Three in a day means the local state has lost coherence with the broker. |
| Daily summary Discord post missing 2 days in a row | 2 consecutive trading days without `scripts.daily_summary` posting | Observability failure — if the summary is silent, you cannot see what the bot did. You stop until you can see. |
| Single-position realized loss >50% of `max_capital_per_spread_usd` (-$250 on a $500 cap) | -$250 realized on one spread package | Earlier signal than the -$300 rolling 7d breaker. A single position taking half the per-spread cap means the strategy's stop/exit logic underperformed; investigate before another one stacks on top. |

## §4 — Rollback procedure

When in doubt, flatten. The decision matrix below is the tiebreaker, but the default tilts toward flatten for a reason — leaving open positions on a misbehaving bot is the most expensive mistake on this list. Follow the steps in order; do not reorder them.

### Decision matrix

**Default: when in doubt, flatten.**

**Flatten if:**

1. The trigger is a broker-side issue (auth, feed, order-routing).
2. The reconciler shows divergence between broker view and local view.
3. The root cause is unknown.
4. Any leg of a spread is exposed (one leg filled, the other did not).
5. Any position belongs to a strategy being rolled back due to a suspected logic error in that strategy.

**Let ride if:**

1. The strategy is paused for a benign reason (e.g. win-rate floor on a known-good strategy you intend to reenable after review) AND the open positions are unaffected by the cause of the pause.
2. The position has DTE ≥7 with no near-term gamma exposure and no leg risk.

### Steps

1. `touch /opt/wheelbot/STOP` (the path in `risk.stop_file_path`). Equivalent: POST `/risk/manual_stop` action=engage from the dashboard. This trips Rule 10 in `execution/kill_switch.py` and halts new-order placement at the top of the next loop tick. The reconciler keeps running.
2. Confirm in `/positions` that no new orders are being placed. Give it 5 minutes. Watch `/orders` for any new rows; if you see one after the STOP file is in place, escalate — the kill switch is not catching.
3. Decide flatten vs let ride using the matrix above. Write the decision down before acting on it.
4. If flatten: `docker exec wheelbot python -m scripts.manual_close --all --dry-run` first to preview, then `docker exec wheelbot python -m scripts.manual_close --all` for real. For a single ticker, use `--symbol <SYM>`. Pass `--force` only if the risk gate (BP floor, position cap) is blocking the close-out under stress.
5. Wait for fills. `manual_close` routes through `OrderRouter`; the reconciler resolves each position when the broker confirms the close. Do not proceed until `/positions` shows the targeted positions in a terminal state.
6. Revert `account.broker` in `config/config.local.yaml` to the prior stage's broker if relevant: from Stage 3 or Stage 4 back to Stage 1 or 2, revert to `alpaca_paper`. Rolling back FROM Stage 2 is just "stop running the parity harness" — Stage 2 does not change the live broker, so no broker config change is needed there.
7. Restore the position caps appropriate for the stage you are returning to (e.g. `account.max_concurrent_total: 14` for paper testing).
8. `cd /opt/wheelbot && git pull && docker compose up -d --build wheelbot`. Watch `docker logs wheelbot --tail 20` for the `bot_migrations` line; verify `docker exec wheelbot python -m scripts.db_health` exits 0 and `bash scripts/migrate_check.sh` shows no drift.
9. Release the kill switch: `rm /opt/wheelbot/STOP` or POST `/risk/manual_stop` action=release from `/risk`. The next loop tick will resume new-order placement.

Do NOT skip forward to the failed stage. Restart from the prior stage's entry criteria in §2 and re-clear every checkbox. The point of the staged runbook is that each gate is independently meaningful; treating a rollback as a brief pause defeats it.

## §5 — Pre-flight checklist before Stage 3

Stage 3 is the first transition where real money is at risk. Treat the entry to Stage 3 as the highest-stakes gate in this runbook: an undetected misconfiguration here goes straight to the account ledger. Do not skip either part of the checklist, do not run them out of order, and do not begin Stage 3 with any FAIL outstanding from Part A or any unchecked item in Part B.

### Part A — Automated checks (run `python -m scripts.preflight_live`)

Run the script from inside the deployed container so it reads the same config and DB the bot will use:

```
docker exec wheelbot python -m scripts.preflight_live
docker exec wheelbot python -m scripts.preflight_live --json   # machine-readable
```

Exit code is non-zero if any required check fails. The script runs the following 11 checks:

| # | Check | Severity | What it verifies |
|---|---|---|---|
| 1 | `broker_selection` | FAIL | `account.broker` is `tastytrade` or `tastytrade_sandbox`; refuses to proceed against `alpaca_paper`. |
| 2 | `broker_secrets` | FAIL | `TASTYTRADE_PROVIDER_SECRET` and `TASTYTRADE_REMEMBER_TOKEN` are both present in `config/secrets.env`. |
| 3 | `broker_auth` | FAIL | Actually instantiates the broker, calls `get_account()`, and prints equity + buying power. |
| 4 | `universe` | FAIL | `load_universe()` returns at least one tier-1 ticker. |
| 5 | `storage_volume` | FAIL | The parent directory of `database.path` exists on the mounted volume. |
| 6 | `db_writable` | FAIL | Opens the SQLite file and runs `SELECT COUNT(*) FROM positions` end-to-end. |
| 7 | `manual_intervention` | WARN | No active positions are in `MANUAL_INTERVENTION` (flagged but not blocking). |
| 8 | `kill_switch` | WARN | Daily-state `kill_switch_armed` is false (an armed kill switch is fine for a preflight, but you must know). |
| 9 | `stop_file` | WARN | `risk.stop_file_path` (default `/opt/wheelbot/STOP`) is not present. |
| 10 | `regime_snapshots` | WARN | `regime_snapshots` table has at least one row; otherwise the regime gate fails open. |
| 11 | `iv_history` | WARN | `iv_history` table has at least 20 rows; otherwise IVR gating fails open. |

Treat WARN rows as items to investigate, not items to skip; for Stage 3 specifically, every WARN should have a written one-line justification before you proceed.

### Part B — Manual checks (TICKET-024 will fold these into `preflight_live` later)

These are not yet automated. Until TICKET-024 lands, walk this list by hand and tick each item:

- DB backed up to a dated file on the host (e.g. `wheelbot-YYYYMMDD-pre-stage3.db`) and the backup is readable.
- `docker exec wheelbot bash scripts/migrate_check.sh` shows no drift between `db/migrations/` in the repo, the migrations baked into the image, and the rows in `schema_migrations`.
- Discord webhook is reachable from the container — send a test message and confirm it arrives in the operator channel.
- Anthropic API key is valid and the day's spend leaves at least one full daily budget (`intelligence.daily_budget_usd: 2.00`) of headroom.
- `intelligence.news_check_advisory: false` is committed in `config/config.local.yaml` (advisory mode never blocks; it must be off before Stage 3).
- Caps reduced for the quarter-size live ramp: `account.max_concurrent_total: 4` and the active strategies' `max_position_pct_of_account: 5`.
- Paper-phase OI/volume overrides removed: `wheel.open_interest_min: 500` and `wheel.volume_min: 100` are effective (override pulled when broker swap to Tastytrade lands).
- Kill switch is released — `/opt/wheelbot/STOP` is absent and the `/risk` page shows the switch disarmed.
- Dashboard reachable from the operator's actual phone and laptop, not just from localhost on the LXC; confirm `/positions`, `/orders`, `/risk`, `/decisions`, `/performance`, `/macro` all render.
- Calendar entry exists for the daily review cadence (operator looks at `/risk` and `/decisions` once per trading day for the first two weeks of live).

## §6 — Manual override procedures

The bot is designed to run hands-off; these tools exist for the cases where it cannot. Reach for them when the bot is wrong, when the bot is right but slower than the situation, or when you need a clean account before an upgrade or rollback. Every command here writes a checkpoint or `state_log` row so the audit trail captures both *what* and *why*.

### Reenable a strategy

A strategy can land in runtime state for two reasons: the drawdown circuit breaker (`auto_disable_drawdown_usd: -300`) or the win-rate floor (`min_win_rate_pct: 60`, `min_closed_cycles: 10`, `pause_duration_days: 14`). The same script clears both gates.

```
docker exec wheelbot python -m scripts.reenable_strategy --list
docker exec wheelbot python -m scripts.reenable_strategy --strategy weekly_wheel
docker exec wheelbot python -m scripts.reenable_strategy --strategy weekly_wheel \
    --reason "reviewed last 10 cycles; 3 losses on KMI during a sector rotation; tightened universe"
```

`--reason` is optional but strongly encouraged for win-rate pauses, where the reenable is a human judgment call; it is redundant for drawdown auto-resets, which are mechanical. The script deletes the entire `strategy_runtime_state` row, which clears BOTH the drawdown gate (`DISABLED` / `WARNING`) AND the pause gate (`LOW_WIN_RATE`) in one shot. The prior state is printed before the clear runs, so if the row had both gates set and you only meant to clear one, you can Ctrl-C. A `strategy_manual_reenable` checkpoint is emitted with the prior gate states and the supplied reason. The static `enabled` flag in `config.yaml` is not touched.

### Close positions manually

Use when the bot won't close a position it should, or when you need a clean account before a halt, upgrade, or rollback. Always preview with `--dry-run` first.

```
docker exec wheelbot python -m scripts.manual_close --symbol F --dry-run
docker exec wheelbot python -m scripts.manual_close --symbol F
docker exec wheelbot python -m scripts.manual_close --all
```

Orders go through `OrderRouter` so `client_order_id` idempotency, retry, and DB writes stay consistent with the live path. State transitions: `CSP_OPEN` / `CSP_PENDING` buy-to-close the short put; `CC_OPEN` / `CC_PENDING` buy-to-close the short call; `SHARES_HELD` sell-to-close the shares. Every action writes a `state_log` row with `triggered_by=MANUAL`. `--force` bypasses the risk gates (`buying_power_floor_pct`, `max_position_pct_of_account`, `max_concurrent_positions`); use it only when the gates themselves are malfunctioning — for example when the BP floor is blocking a close-out under stress and you've decided the close-out is correct.

### Kill switch — halts NEW orders, does NOT close existing positions

The kill switch is a halt, not a flatten. It stops the router from placing new orders; in-flight orders continue, fills continue to reconcile, and existing positions remain on the book.

Engage:

```
touch /opt/wheelbot/STOP
```

or click Engage on the `/risk` page (POSTs to `/risk/manual_stop`).

Release:

```
rm /opt/wheelbot/STOP
```

or click Release on the `/risk` page.

Important: an emergency shutdown is three steps, in this order:

1. Engage the kill switch (`touch /opt/wheelbot/STOP`).
2. Run `python -m scripts.manual_close --all`.
3. Verify `/positions` is empty before walking away.

## §7 — Config diffs by stage

Each stage flips a small, named set of keys. Everything not listed stays at its Stage 0 value. Apply the diff before the stage starts; do not roll multiple stage diffs into one deploy.

| Key | Stage 0 (paper) | Stage 1 (enforced) | Stage 3 (live 1/4) | Stage 4 (full) |
|---|---|---|---|---|
| `account.broker` | `alpaca_paper` | `alpaca_paper` | `tastytrade` | `tastytrade` |
| `account.max_concurrent_total` | `14` | `14` | `4` | `14` |
| `intelligence.news_check_advisory` | `true` | `false` | `false` | `false` |
| per-strategy `max_position_pct_of_account` | current | current | `5` | current |
| `wheel.open_interest_min` | `0` (paper) | `0` (paper) | `500` (drop override) | `500` |
| `wheel.volume_min` | `0` (paper) | `0` (paper) | `100` (drop override) | `100` |

Stage 2 runs in parallel with Stage 1 — no config diff vs Stage 1.

## Appendix — What could go wrong at each stage

A brief look at the most likely failure mode at each stage and what its early signal looks like. Use this as a triage starting point, not as a definitive list — anything new gets investigated on its merits.

### Stage 1

After flipping `intelligence.news_check_advisory` to `false` in `config/config.local.yaml` and rebuilding, expect the most likely failure to be over-blocking: Haiku's caution bias plus the qty=1-caution-becomes-block rule chokes off Tier-1 single-lot CSPs. Watch `/decisions?decision_type=news_check` for the proceed/caution/block mix versus the prior 30 days, and grep container logs for `router_news_check decision=block` and `router_news_caution_block` checkpoints. If entries dry up entirely with no caution/block decisions logged, the gate is fail-opening — check for `news_check_source_skip` (Finnhub down/rate-limited), `news_check_no_headlines`, `news_check_budget_skip`, or `news_check_llm_fail` first. Confirm Tier-1 safety nets (TICKET-005 through 009) each fire at least once before promoting to live capital; revert the flag if caution rate exceeds ~40%.

### Stage 2

The most likely break at this stage is the parity harness itself, not either broker: `scripts/parity_run.py` joins Alpaca and Tastytrade quotes by `contract_symbol`, and if it stores Alpaca's raw symbol (which may carry the rstrip-friendly padded prefix per `_parse_occ` in `platforms/alpaca_broker.py`) against Tastytrade's `_occ_dense`-normalized form, the join silently misses. You will see this on the `/parity` page as a Sample Count cliff with Avg Mid Diff % pinned near zero, or Worst Diff % punching past the 5% acceptance threshold on names you know are liquid. Spot-check the daily markdown under `/mnt/wheelbot-storage/parity_reports/` for rows with one side NULL. Fix: route both feeds through `platforms.tastytrade_broker._occ_dense` before insert into `broker_parity_log`, then re-run the day.

### Stage 3

Most likely failure: your first real Tastytrade fills come in materially worse than the sandbox parity test predicted, meaning cert.tastyworks.com did not represent live routing. On the `/orders` dashboard panel, watch the first 5 filled rows — if `fill_price` consistently diverges from your submitted `limit_price` by more than expected slippage, the parity assumption is broken. Note that `place_multi_leg_order` returns `fill_price=None` (`platforms/tastytrade_broker.py`), so the reconciler back-fills from `limit_price` at `reconciler.py:288` and will mask the gap; check broker order history directly. Second failure mode: a `risk.kill_switch_armed` notification on a drawdown number you didn't set — Stage 0 forgot to create `config/config.local.yaml`, so `daily_loss_kill_switch_pct` still reads the committed default. Pause the loop, write the missing overrides, restart.

### Stage 4

At Stage 4 the most likely break is a USD-absolute threshold tripping on noise that was invisible at quarter size — `auto_disable_drawdown_usd` (-$300) and `drawdown_warning_usd` (-$150) are per-strategy dollar floors, not percent-of-equity, and a normal red day across 14 concurrent positions can clear -$150 before lunch. You will see the `drawdown.warning` Discord alert on a strategy that was quiet through Stages 1-3, the `/risk` page badge flipping to WARNING with size_multiplier 0.5, and a `drawdown_warning_triggered` checkpoint in the log. Cross-check with `risk_gate` failures clustering on `per_position_cap` or `concurrent_total_cap` for correlated names. If the trip looks like variance not regime, raise the thresholds in config and rebuild — do not manually clear and re-arm the same number.

---

*Last reviewed: 2026-06-04.*
