# CLAUDE.md — project context for Claude Code

This file is loaded into context whenever Claude Code works in this repo. Keep
it short and stable; longer-lived design docs live in `docs/`.

## Repo layout (top level)

- `scripts/run_bot.py`     — the long-running trading loop (Docker entrypoint)
- `scripts/run_*.py`       — cron entrypoints (screener, regime, daily summary)
- `core/`                  — broker abstraction, config, notifier, models
- `strategies/`            — wheel + spreads orchestrators and selectors
- `risk/`                  — risk gate, regime classifier, auto-disable
- `execution/`             — router and reconciler (single source of truth)
- `intelligence/`          — Anthropic client, screener, news_check, budget
- `data/`                  — chain, IVR, earnings, yfinance helpers
- `platforms/`             — Alpaca, Tastytrade, PaperBroker adapters
- `dashboard/`             — FastAPI + Jinja UI
- `db/`                    — repos, models, migrations (SQLite)
- `tests/unit/`            — pytest suite

## Deployment Gotchas

### DB migrations are baked into the Docker image AND auto-apply on startup

Migrations in `db/migrations/` are baked into the Docker image at build time.
**As of TICKET-007 follow-up, `scripts/run_bot.py` calls
`scripts.run_migration.apply_pending()` at startup**, so a fresh container
self-heals onto the latest schema — no extra step. Per-migration writes are
transactional; if any fails, the bot exits non-zero and the container is
marked unhealthy by Docker rather than running on a partial schema.

A `git pull` on the LXC host alone does NOT update the running container.
You still need to rebuild the image.

**To deploy:**
```
cd /opt/wheelbot
git pull
docker compose up -d --build      # NO service name → rebuilds BOTH wheelbot + dashboard
# auto-applies any pending migrations during startup; watch the log:
docker logs wheelbot --tail 20 | grep -E 'bot_migrations'
# Verify:
docker exec wheelbot python -m scripts.db_health
docker exec wheelbot bash scripts/migrate_check.sh
```

`migrate_check.sh` diffs the host's `db/migrations/` directory against the
files inside the running container AND lists applied versions from
`schema_migrations` — drift detector.

If a deploy somehow misses the auto-apply (e.g. you ran the bot under a
different entrypoint), the manual fallback is:
```
docker exec wheelbot python -m scripts.run_migration --all-pending
```

### Config + code changes both need `--build` — and rebuild BOTH services

Same trap applies to any code change: `git pull` alone leaves the container
running the old image. Always rebuild — and **never name a single service**.
`up -d --build wheelbot` rebuilds only the bot and silently leaves the
`dashboard` container frozen on old code; once the DB schema drifts the
dashboard 500s (it bit us 2026-06-17 — a 3-week-stale dashboard rejected the
`trigger_reason` column on /orders and /positions). Name no service so compose
rebuilds the bot AND the dashboard together:

```
docker compose up -d --build      # bot + dashboard, kept in sync
```

You can tell the deployed image is stale when:
- Order limit prices on Alpaca match `net_credit` exactly (no slippage applied).
- `/decisions` rows show no recent activity but `git log` shows new fixes.
- Position rows show stale `current_cycle_id` that newer reconciler code clears.

### Cron timezone

The LXC runs on **MDT (UTC-6)** local time — verified via `date` on CT 105
(2026-06-23). An earlier note here claimed the box had been switched to UTC;
that did not hold, so treat the LXC as MDT. The installed host crontab is
MDT-aware (its header comment and the times line up, e.g. `0 7` = 07:00 MDT =
09:00 ET pre-market), so jobs fire at the intended ET moments. Keep any new
cron entries in LXC-local MDT (e.g. the daily macro-calendar refresh at
`0 6 * * *` = 06:00 MDT).

## Workflow conventions

- **One sprint / ticket at a time.** Propose scope before scaffolding, wait
  for explicit ack before commit/push.
- Every new module that mutates state writes a `core.checkpoint.log_checkpoint(...)`.
- All Anthropic calls go through `intelligence.anthropic_client.AnthropicClient`.
- All DB migrations go in `db/migrations/NNNN_description.sql`.
- New config knobs default to **off** or **conservative**.
- Tests live in `tests/unit/test_<module>.py`.

## Live state knobs that matter

- `intelligence.news_check_advisory`: `true` (paper testing — caution
  proceeds at full size; only `block` cancels). Flip to `false` before live.
- `intelligence.llm_screener_enabled`: `true`. Opus runs daily via cron.
- `intelligence.llm_roll_advisor_enabled`: `false`. Re-enable after 3+ months
  of paper data.
- `account.max_concurrent_total`: `14` during testing (= sum of per-strategy
  caps so the per-strategy limits actually bind). Tighten to ~4 before live.

## Live-deployment authority

`docs/GO_LIVE_RUNBOOK.md` is the authoritative sequence for taking WheelBot
live; any change to `account.broker`, position caps (`max_concurrent_total`,
per-strategy `max_position_pct_of_account`), or risk thresholds
(`daily_loss_kill_switch_pct`, `consecutive_losses_pause`,
`auto_disable_drawdown_usd`, `drawdown_warning_usd`) that affects
live-trading readiness must update the runbook in the same commit. The
runbook is rendered at `/runbook` on the dashboard; the markdown file in
`docs/` is the source of truth.
