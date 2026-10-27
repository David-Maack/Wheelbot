# WheelBot

Automated options-income bot on Alpaca paper. Since the 2026-10-27 two-strategy
cleanup it runs a monthly wheel (cash-secured puts → covered calls on a few cheap,
ownable names) and `zero_dte` (same-day SPY/QQQ bull put spreads). See
`docs/HOW_IT_WORKS.md` for the full picture and `docs/GO_LIVE_RUNBOOK.md` for the
path to live capital.

See `wheelbot_spec.md.pdf` for the full technical specification.

## Status

Two strategies, each judged by one scorecard (`risk/scorecard.py`: closed
trades, $/trade, max drawdown, fill rate — go-live bar 30 trades, $/trade > 0,
fill ≥ 70%). The rule-based roll advisor (`strategies/roll_advisor.py`)
decides `ROLL` / `LET_ASSIGN` / `CLOSE` for challenged wheel shorts. The
AI layer, eight other strategies and three breakers were removed in the
cleanup; git history keeps them. Discord webhook notifier (`core/notify.py`)
emits state-change and risk events; it defaults to a NullNotifier when no
`DISCORD_WEBHOOK_URL` is set.

## Quick start (local dev)

```bash
python -m venv .venv
source .venv/bin/activate                              # Windows: .venv\Scripts\activate
pip install -e ".[dev,broker,dashboard,data]"

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
strategies/     Wheel + zero_dte orchestrators, CSP/CC selectors, roll/exit managers
execution/      Order router, reconciler, kill switch
risk/           Risk gate, regime, drawdown breaker, scorecard
dashboard/      FastAPI + HTMX views on :8889
db/             schema.sql, migrations, repository pattern
config/         config.yaml, config.local.yaml, universe.yaml, secrets.env
tests/          unit, integration, fixtures
scripts/        bootstrap_db, run_bot, run_regime, refresh_macro_calendar, ingest_history,
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
- Daily regime snapshot: `30 16 * * 1-5 /opt/wheelbot/.venv/bin/python -m scripts.run_regime` writes one row to `regime_snapshots`. Once present the §8 #7 gate enforces (CSPs blocked when `csps_allowed=false`).
- Cycle backtester: `python -m scripts.backtest_cycle --cycle-id 42` (or `--json`) replays the cycle's decisions through the *current* strategy code using the chain captured at decision time. Useful for measuring parameter drift after you tune wheel params.
- Discord notifications: paste a webhook URL into `DISCORD_WEBHOOK_URL` (Discord → Server Settings → Integrations → Webhooks). The bot posts state-change events (assignments, called-away, MANUAL_INTERVENTION) and risk events (kill switch armed, strategy auto-disabled, broker down, macro calendar running out, cycle closed at a loss). Disable via `notifications.enabled: false` in config.

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

# Daily macro-calendar refresh (loads config/macro_calendar.yaml).
0 6 * * *  docker exec wheelbot python -m scripts.refresh_macro_calendar

# Daily summary + scorecard to Discord.
35 15 * * 1-5  docker exec wheelbot python -m scripts.daily_summary

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
