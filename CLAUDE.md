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
docker compose up -d --build wheelbot
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

### Config + code changes both need `--build`

Same trap applies to any code change: `git pull` alone leaves the container
running the old image. Always rebuild:

```
docker compose up -d --build wheelbot
```

You can tell the deployed image is stale when:
- Order limit prices on Alpaca match `net_credit` exactly (no slippage applied).
- `/decisions` rows show no recent activity but `git log` shows new fixes.
- Position rows show stale `current_cycle_id` that newer reconciler code clears.

### Cron timezone

The LXC was historically MDT, which made the screener cron fire 6 hours off.
TZ is now **UTC** on the LXC. Cron entries in this repo (`scripts/run_*.py`)
assume UTC; if you re-host on a non-UTC machine, adjust the crontab.

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
