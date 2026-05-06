# WheelBot

Automated options-wheel trading bot. Sells cash-secured puts on a curated universe; if
assigned, sells covered calls; collects premium continuously. Companion to PolyTrader.

See `wheelbot_spec.md.pdf` for the full technical specification.

## Status

Sprint 8 — Polish (the last sprint per spec §13). Rule-based roll advisor
(`strategies/roll_advisor.py`) decides `ROLL` / `LET_ASSIGN` / `CLOSE` for
ITM short options; LLM ensemble (`intelligence/roll_advisor_llm.py`,
default off) provides a second pair of eyes. The roll orchestrator halts
the position for human review on disagreement. Reconciler runs the
roll-trigger scan each tick. New `chain_snapshots` table captures the
post-filter chain at each strategy decision so the new
`scripts/backtest_cycle.py` can replay decisions through current strategy
code and report divergences. Discord webhook notifier
(`core/notify.py`) emits state-change and risk events: position assigned,
called away, MANUAL_INTERVENTION, broker down, kill switch armed, daily
LLM budget exceeded, roll disagreement, cycle closed at a loss.
Notifier defaults to a NullNotifier when no `DISCORD_WEBHOOK_URL` is set.

## Quick start (local dev)

```bash
python -m venv .venv
source .venv/bin/activate                              # Windows: .venv\Scripts\activate
pip install -e ".[dev,broker,dashboard,intelligence,data]"

cp config/secrets.env.example      config/secrets.env
cp config/config.local.yaml.example config/config.local.yaml
# edit config/secrets.env with your keys
# adjust config/config.local.yaml for your phase

python scripts/bootstrap_db.py
python -m scripts.run_bot --once                       # smoke: one tick + exit
```

## Tests

```bash
pytest tests/unit                  # always run, no creds needed
pytest tests/integration -v        # auto-skips if ALPACA_API_KEY/SECRET unset
```

## Layout

```
core/           Broker ABC, lifecycle, models, config, checkpoint logger
platforms/      Broker implementations (Tastytrade, Alpaca, paper)
data/           Option chain, Greeks, IVR, earnings, universe
strategies/     Wheel orchestrator, CSP/CC selectors, roll/exit managers
execution/      Order router, reconciler, kill switch
intelligence/   LLM screener, news check, roll advisor, ensemble
risk/           Sizing, exposure, regime, hard limits
dashboard/      FastAPI + HTMX views on :8889
db/             schema.sql, migrations, repository pattern
config/         config.yaml, config.local.yaml, universe.yaml, secrets.env
tests/          unit, integration, fixtures
scripts/        bootstrap_db, run_bot, run_screener, run_regime, ingest_history,
                manual_close, replay_cycle, backup_db, backtest_cycle,
                bootstrap_tastytrade, preflight_live
```

## Configuration

Three layers, each overlays the previous:

1. `config/config.yaml`        — defaults, committed
2. `config/config.local.yaml`  — local overrides, gitignored
3. `config/secrets.env`        — API keys, gitignored

## Operations

- Manual stop file: `touch /opt/wheelbot/STOP` halts all new orders. Reconciler keeps running.
- Dashboard: <http://127.0.0.1:8889> — run with `uvicorn dashboard.app:create_app --factory --host 127.0.0.1 --port 8889`. HTTP Basic auth: username from `config.dashboard.basic_auth_user`, password from `WHEELBOT_DASHBOARD_PASSWORD`. Reach via Tailscale or SSH tunnel.
- Manual close: `python -m scripts.manual_close --symbol F` (or `--all`); `--dry-run` previews, `--force` bypasses risk gates.
- Replay cycle: `python -m scripts.replay_cycle --cycle-id 42` (or `--latest [--symbol F]`).
- IV history ingest: `0 17 * * 1-5 /opt/wheelbot/.venv/bin/python -m scripts.ingest_history --source=broker` writes one row per universe ticker to `iv_history`. Without this, the IVR gate stays fail-open forever (insufficient history).
- DB backups: `30 23 * * * /opt/wheelbot/.venv/bin/python -m scripts.backup_db` writes `wheelbot-YYYY-MM-DD.sql.gz` to `<db-dir>/backups/` and keeps 30 days.
- Tastytrade bootstrap: register an app at <https://developer.tastytrade.com>, write `TASTYTRADE_PROVIDER_SECRET=...` into `config/secrets.env`, then `python -m scripts.bootstrap_tastytrade --sandbox` (interactive) or `python -m scripts.bootstrap_tastytrade --sandbox --username you@example.com --password-stdin <pw.txt`. Add `--prod` later for production. The script writes `TASTYTRADE_REMEMBER_TOKEN`, `TASTYTRADE_USE_SANDBOX`, optionally `TASTYTRADE_ACCOUNT_NUMBER` — leaves all other lines untouched.
- Pre-live checks: `python -m scripts.preflight_live` (text) or `--json`. Exits non-zero if any required check fails. Run before flipping `account.broker` to `tastytrade`.
- Daily LLM screener: `0 8 * * 1-5 /opt/wheelbot/.venv/bin/python -m scripts.run_screener` writes top-N rows to `candidates`. Visible at `/candidates` in the dashboard.
- Daily regime snapshot: `30 16 * * 1-5 /opt/wheelbot/.venv/bin/python -m scripts.run_regime` writes one row to `regime_snapshots`. Once present the §8 #7 gate enforces (CSPs blocked when `csps_allowed=false`).
- LLM spend cap: configured at `intelligence.daily_budget_usd` (default $1). Tracked at `/decisions` with today's spend vs cap. When cap is hit, screener and news_check fail-open (skip + log); router still places orders.
- News source backups: when Finnhub rate-limits, news_check fails-open (treats as `proceed`). Recommended backups to obtain: `NEWSAPI_API_KEY` (newsapi.org, 100 req/day free), Marketaux, Polygon.io, or Alpaca News (uses your existing Alpaca key). Add a single adapter file in `intelligence/news.py` to plug in.
- Cycle backtester: `python -m scripts.backtest_cycle --cycle-id 42` (or `--json`) replays the cycle's decisions through the *current* strategy code using the chain captured at decision time. Useful for measuring parameter drift after you tune wheel params.
- Discord notifications: paste a webhook URL into `DISCORD_WEBHOOK_URL` (Discord → Server Settings → Integrations → Webhooks). The bot posts state-change events (assignments, called-away, MANUAL_INTERVENTION) and risk events (kill switch armed, budget exhausted, broker down, roll disagreement, cycle closed at a loss). Disable via `notifications.enabled: false` in config.

## LXC deploy (Proxmox)

The bot runs as two `docker-compose` services on a fresh LXC:

```bash
# 1. On the LXC host, clone the repo to your storage volume.
mkdir -p /mnt/wheelbot-storage
git clone https://github.com/David-Maack/Wheelbot.git /opt/wheelbot
cd /opt/wheelbot

# 2. Configure secrets and local overrides (gitignored — create per LXC).
cp config/secrets.env.example      config/secrets.env
cp config/config.local.yaml.example config/config.local.yaml
$EDITOR config/secrets.env          # paste API keys
$EDITOR config/config.local.yaml    # adjust phase / position sizing

# 3. First-time DB bootstrap (creates schema in /mnt/wheelbot-storage/wheelbot.db).
docker compose run --rm wheelbot python -m scripts.bootstrap_db

# 4. Bring up bot + dashboard.
docker compose up -d
docker compose logs -f wheelbot      # tail bot logs
```

Reach the dashboard at `http://127.0.0.1:8889/` from the LXC host (or via
Tailscale / SSH tunnel). Healthcheck endpoint: `GET /healthz` — both services
have Docker healthchecks wired.

### Cron pattern (host crontab → docker exec)

Schedule the periodic jobs from the LXC host's crontab. Each one runs inside
the bot container so it shares config + DB + creds:

```cron
# Daily IV ingest (after market close).
0 17 * * 1-5  docker exec wheelbot python -m scripts.ingest_history --source=broker

# Daily regime snapshot.
30 16 * * 1-5  docker exec wheelbot python -m scripts.run_regime

# Daily LLM screener (pre-market).
0 8 * * 1-5  docker exec wheelbot python -m scripts.run_screener

# Nightly DB backup.
30 23 * * *  docker exec wheelbot python -m scripts.backup_db
```

Append `>> /var/log/wheelbot-cron.log 2>&1` to each line if you want a
durable cron log on the LXC host.

### Updating

```bash
cd /opt/wheelbot
git pull
docker compose build --no-cache
docker compose up -d
```

The schema migrations in `db/migrations/` are idempotent against
`schema.sql`; running `python -m scripts.bootstrap_db` after a pull is
safe.
