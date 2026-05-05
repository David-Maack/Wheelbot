# WheelBot

Automated options-wheel trading bot. Sells cash-secured puts on a curated universe; if
assigned, sells covered calls; collects premium continuously. Companion to PolyTrader.

See `wheelbot_spec.md.pdf` for the full technical specification.

## Status

Sprint 1 — Foundation. Repo skeleton, DB schema, Pydantic models, repo layer, config loader,
and checkpoint logger. No broker integration, no live orders.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

cp config/secrets.env.example config/secrets.env
# edit config/secrets.env with your keys

python scripts/bootstrap_db.py
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
- Dashboard: <http://127.0.0.1:8889> (after Sprint 5).
- DB backups: nightly cron writes to `/mnt/wheelbot-storage/backups/` (after Sprint 5).
