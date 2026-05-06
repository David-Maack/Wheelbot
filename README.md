# WheelBot

Automated options-wheel trading bot. Sells cash-secured puts on a curated universe; if
assigned, sells covered calls; collects premium continuously. Companion to PolyTrader.

See `wheelbot_spec.md.pdf` for the full technical specification.

## Status

Sprint 6 — Tastytrade & Live Prep. `platforms/tastytrade_broker.py` against
the tastyware/tastytrade SDK 12.x (fully async, OAuth2). Broker factory wires
`tastytrade` → prod, `tastytrade_sandbox` → cert.tastyworks.com. One-time
OAuth bootstrap (`scripts/bootstrap_tastytrade.py`) supports both interactive
and `--password-stdin` headless flows; refuses to clobber an existing prod
token with sandbox credentials (or vice versa) without `--force`.
`scripts/preflight_live.py` runs read-only readiness checks (broker auth,
universe, DB, MANUAL_INTERVENTION queue, kill switch, stop file, regime/IV
history thinness) before flipping config to production. The reconciler's
mismatch coverage is reinforced by a dedicated test suite per spec §13 #29.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev,broker]"     # 'broker' brings alpaca-py + tastytrade

cp config/secrets.env.example config/secrets.env
# edit config/secrets.env with your keys

python scripts/bootstrap_db.py
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
scripts/        bootstrap_db, ingest_history, manual_close, replay_cycle
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
- DB backups: `30 23 * * * /opt/wheelbot/.venv/bin/python -m scripts.backup_db` writes `wheelbot-YYYY-MM-DD.sql.gz` to `<db-dir>/backups/` and keeps 30 days.
- Tastytrade bootstrap: register an app at <https://developer.tastytrade.com>, write `TASTYTRADE_PROVIDER_SECRET=...` into `config/secrets.env`, then `python -m scripts.bootstrap_tastytrade --sandbox` (interactive) or `python -m scripts.bootstrap_tastytrade --sandbox --username you@example.com --password-stdin <pw.txt`. Add `--prod` later for production. The script writes `TASTYTRADE_REMEMBER_TOKEN`, `TASTYTRADE_USE_SANDBOX`, optionally `TASTYTRADE_ACCOUNT_NUMBER` — leaves all other lines untouched.
- Pre-live checks: `python -m scripts.preflight_live` (text) or `--json`. Exits non-zero if any required check fails. Run before flipping `account.broker` to `tastytrade`.
