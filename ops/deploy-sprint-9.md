# Sprint 9 deploy plan — multi-strategy bot

Bringing `monthly_wheel`, `weekly_wheel`, and `put_spread` live in parallel on
LXC 105 for the 6–8 week comparison run.

Target commit: `021d688` (or later) on `main`.

## 1. Pre-flight (dev machine)

```powershell
cd "C:\Users\david\Documents\Wheel options bot"
git pull origin main
git log -1 --oneline                              # confirm 021d688 or later
.venv\Scripts\python -m pytest tests/unit -q      # expect 269/269
```

Optional repo edit to make `put_spread` the default-on (skip if you want it
opt-in only):

- `config/config.yaml` ~line 166: `enabled: false` → `enabled: true`
- Commit + push as `Enable put_spread strategy by default`

## 2. Capital-allocation pre-flight (run on LXC)

Sanity check that the three strategies won't immediately fight each other for
buying power, plus a tuning flag for `put_spread` on the mega-cap universe.

### 2.1 Current account state

```bash
docker exec wheelbot-dashboard curl -s -u wheelbot:$WHEELBOT_DASHBOARD_PASSWORD \
  http://127.0.0.1:8889/healthz

docker exec wheelbot-dashboard sqlite3 /mnt/wheelbot-storage/wheelbot.db \
  "SELECT state, strategy_id, symbol, current_cycle_id
   FROM positions
   WHERE state != 'IDLE'
   ORDER BY strategy_id, symbol;"
```

Record:
- **Equity** $E (Alpaca paper starts at $100k unless reset)
- **Free buying power** $BP
- **Already-open slots** by strategy_id

### 2.2 Worst-case footprint per strategy

| Strategy       | Per-position BP (worst case)             | Slots | Footprint                        |
| -------------- | ---------------------------------------- | ----- | -------------------------------- |
| monthly_wheel  | `median_strike × 100` (CSP cash)         | 4     | strike-median × 400              |
| weekly_wheel   | `median_strike × 100` (CSP cash)         | 4     | strike-median × 400              |
| put_spread     | `max_loss_per_spread × qty`              | 4     | (width − credit) × 100 × qty × 4 |

Current-universe estimates on the paper account:

- monthly_wheel universe ≈ {F, BAC, NOK, SOFI, T, VZ, INTC, KMI} — median
  strike ~$15 → **~$6k**
- weekly_wheel universe ≈ {HOOD, PLTR, PLUG, RIVN, AMD, MARA, COIN, AAL} —
  median strike ~$25 → **~$10k**
- put_spread universe — see tuning warning below

On a $100k paper account that's comfortably under the 20% BP floor
($20k reserved). On a $25k account the wheel pair alone is tight.

### 2.3 put_spread tuning warning ⚠️

The default `spread_width_dollars: 1.0` was sized for $10–$20 underlyings. The
put_spread universe is mega-caps ($150–$700). A $1-wide put spread on a $500
stock at 25Δ yields a tiny credit — likely failing the `min_credit_pct_of_width: 25.0`
gate and producing zero proposals.

**Footgun**: `core/config.py:_merge_sections` is a one-level shallow merge.
For list-valued top-level keys (like `strategies`), an override **replaces**
the whole list. If you put a single-strategy `strategies:` block in
`config.local.yaml`, you wipe out monthly_wheel and weekly_wheel.

**Option A (recommended)** — edit the repo, not the overlay:

In `config/config.yaml` under the put_spread entry:

```yaml
  - id: put_spread
    display_name: "Bull Put Spread"
    type: vertical_spread
    enabled: true                        # was: false
    max_concurrent: 4
    params:
      dte_min: 30
      dte_max: 45
      short_delta_min: 0.20
      short_delta_max: 0.30
      spread_width_dollars: 5.0          # was: 1.0
      max_capital_per_spread_usd: 500    # new — caps qty to ~1 at width=5
      profit_close_pct: 50
```

Commit + push. On the LXC, the only `config.local.yaml` change is to remove
`- put_spread` from `disabled_strategies`.

Worst-case footprint then: `max_loss ≈ $400/package × 1 × 4 slots = $1,600`.

**Option B** — leave width at $1, accept that put_spread will likely propose
nothing for the first weeks (useful as a "control" arm; otherwise just disable).

**Option C** (if you must override via local config): mirror all three
strategies verbatim in `config.local.yaml` to work around the list-replacement
behavior. Verbose but explicit.

### 2.4 Sanity-check the loader

```bash
docker exec wheelbot python -c "
from core.config import load_config
c = load_config()
floor = c['wheel']['buying_power_floor_pct']
print(f'BP floor: {floor}% — at \$E equity, \$%s reserved' % (floor * 1000))
for s in c.get('strategies', []):
    p = s.get('params', {})
    print(f\"{s['id']}: max_concurrent={s['max_concurrent']}, \"
          f\"width={p.get('spread_width_dollars','-')}, \"
          f\"cap_usd={p.get('max_capital_per_spread_usd','-')}\")
"
```

### 2.5 Existing positions check

```bash
docker exec wheelbot-dashboard sqlite3 /mnt/wheelbot-storage/wheelbot.db \
  "SELECT COUNT(*) AS n, strategy_id FROM positions
   WHERE state IN ('CSP_OPEN','CSP_PENDING','SHARES_HELD','CC_OPEN','CC_PENDING',
                   'SPREAD_OPEN','SPREAD_PENDING')
   GROUP BY strategy_id;"
```

If a strategy already has 4 active slots the risk gate's
`concurrent_positions_cap` blocks new entries for that strategy regardless —
confirm you haven't drifted from intended capacity.

### 2.6 Decision gate

- [ ] Account equity ≥ $20k (or accept tighter operation)
- [ ] put_spread width tuned for mega-cap universe (Option A) OR accepted
      near-zero activity (Option B)
- [ ] Current open positions ≤ 4 per strategy
- [ ] BP floor not already breached

Only when all four are checked → proceed to Deploy.

## 3. Deploy

```bash
# SSH to host, pct enter 105, cd /opt/wheelbot
git pull
git log -1 --oneline                       # confirm target commit
```

Edit `config/config.local.yaml` on the LXC:

1. Remove or comment out the `- put_spread` line in `disabled_strategies`.
2. If you took Option A for tuning, add the strategies override block from
   §2.3.

Apply:

```bash
docker compose build wheelbot dashboard
docker compose up -d wheelbot dashboard
docker compose ps                          # both healthy?
docker logs wheelbot --tail 80 | grep -E 'bot_strategies_loaded|strategy='
```

Expected log line: `enabled=['monthly_wheel', 'weekly_wheel', 'put_spread']`.

## 4. Validation (first 30 min)

- Hit `/performance` in browser — three cards visible, `put_spread` shows 0
  cycles.
- One full reconcile tick:
  ```bash
  docker logs wheelbot --tail 200 | grep -E 'reconcile_once|bot_strategy_summary'
  ```
  `put_spread` summary line should show `n_proposals ≥ 0`.
- First spread fill (when it happens):
  ```bash
  docker logs wheelbot | grep -E 'router_submit_mleg|fill'
  ```
  Then `/performance` → put_spread `open_cycles` increments.

### Health markers

| Signal                   | Where                                              | Good                                              |
| ------------------------ | -------------------------------------------------- | ------------------------------------------------- |
| Bot heartbeat            | `/healthz`                                         | `status: ok`                                      |
| Spread orders submitted  | `docker logs wheelbot | grep router_submit_mleg`   | new entries during market hours                   |
| Cycles attributed        | `/performance` "By strategy" cards                 | `open_cycles` count grows per strategy            |
| No MANUAL_INTERVENTION   | `/` (Positions)                                    | no rows in MANUAL_INTERVENTION state              |

## 5. Rollback plan

### Soft rollback — kill put_spread only, keep wheel running

```bash
# On LXC, edit config/config.local.yaml — re-add the line:
disabled_strategies:
  - put_spread

docker compose restart wheelbot
```

Effect: no new spread proposals next tick. Open spreads keep being managed by
the reconciler + close orchestrator until they close naturally. **Do not** rip
out the code mid-cycle — the reconciler needs MULTI_LEG_OPEN/CLOSE handling to
wind down cleanly.

### Hard rollback — full code revert

```bash
# On LXC
cd /opt/wheelbot
git log --oneline -10              # find pre-sub-sprint-1 commit (b5382c4)
git checkout b5382c4
docker compose build wheelbot dashboard
docker compose up -d
```

⚠️ **Hard-rollback caveat**: pre-sub-sprint-1 code doesn't know about
`SPREAD_*` states or `MULTI_LEG_*` order_types. Any open spreads would become
invisible to the wheel bot and need to be manually closed at the Alpaca UI
before reverting. **Only hard-rollback when there are no SPREAD_OPEN
positions.** Check `/` for SPREAD_* rows first; if any exist, close them via
the dashboard or Alpaca before reverting.

### Emergency stop — instant halt, no rollback needed

```bash
# On LXC
touch /mnt/wheelbot-storage/STOP
```

The stop file halts new entries (all strategies) without unwinding open
positions. Existing reconciliation continues. Remove the file when ready to
resume.

## 6. Post-deploy cleanup (after 24h stable)

```bash
docker exec wheelbot-dashboard rm /mnt/wheelbot-storage/wheelbot.db.backup-pre-recovery
```

That's the leftover backup from the migration-004 recovery.
