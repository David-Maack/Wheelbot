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
- Daily LLM screener: `0 8 * * 1-5 /opt/wheelbot/.venv/bin/python -m scripts.run_screener` writes top-N rows to `candidates`. Visible at `/candidates` in the dashboard.
- Daily regime snapshot: `30 16 * * 1-5 /opt/wheelbot/.venv/bin/python -m scripts.run_regime` writes one row to `regime_snapshots`. Once present the §8 #7 gate enforces (CSPs blocked when `csps_allowed=false`).
- LLM spend cap: configured at `intelligence.daily_budget_usd` (default $1). Tracked at `/decisions` with today's spend vs cap. When cap is hit, screener and news_check fail-open (skip + log); router still places orders.
- News source backups: when Finnhub rate-limits, news_check fails-open (treats as `proceed`). Recommended backups to obtain: `NEWSAPI_API_KEY` (newsapi.org, 100 req/day free), Marketaux, Polygon.io, or Alpaca News (uses your existing Alpaca key). Add a single adapter file in `intelligence/news.py` to plug in.
- Cycle backtester: `python -m scripts.backtest_cycle --cycle-id 42` (or `--json`) replays the cycle's decisions through the *current* strategy code using the chain captured at decision time. Useful for measuring parameter drift after you tune wheel params.
- Discord notifications: paste a webhook URL into `DISCORD_WEBHOOK_URL` (Discord → Server Settings → Integrations → Webhooks). The bot posts state-change events (assignments, called-away, MANUAL_INTERVENTION) and risk events (kill switch armed, budget exhausted, broker down, roll disagreement, cycle closed at a loss). Disable via `notifications.enabled: false` in config.
