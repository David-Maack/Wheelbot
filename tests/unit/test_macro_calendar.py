"""TICKET-007 — macro event blackout calendar.

Covers:
  - The predicate (lifespan overlap, reason string with blocking date)
  - NYSE timezone for `today`
  - Finnhub event-name mapping to canonical types
  - YAML fallback + valid_until stale-warning
  - AlertRateLimitsRepo daily-rate-limit pattern
  - Fail-open paths (empty, stale)
  - The risk-gate rule integration (single + multi-leg, bear_call NOT bypassed)
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml

from core.models import (
    MacroEvent,
    OptionContract,
    OptionType,
    OrderLeg,
    OrderType,
    UniverseEntry,
)
from data.macro_calendar import (
    BlackoutDecision,
    MacroCalendar,
    _today_nyse,
    load_yaml_calendar,
    map_finnhub_event_name,
    maybe_log_yaml_stale,
    parse_finnhub_events,
)
from strategies.spreads import MultiLegProposal
from strategies.wheel import Proposal


# -- helpers ----------------------------------------------------------------


def _today() -> date:
    return date(2026, 6, 3)


async def _seed_event(repos, *, day: date, etype: str = "FOMC", impact: str = "high", source: str = "yaml"):
    now = datetime.now(UTC).replace(tzinfo=None)
    await repos.macro_events.upsert_many([
        MacroEvent(
            event_date=day, event_type=etype, impact=impact,
            description=f"{etype} test event",
            fetched_at=now, created_at=now, source=source,
        )
    ])


def _config(**macro_overrides: Any) -> dict[str, Any]:
    base = {
        "account": {"id": "test"},
        "risk": {
            "macro_blackout": {
                "enabled": True,
                "event_types": ["FOMC", "CPI", "NFP"],
                "blackout_days_before": 1,
                "blackout_days_after": 0,
                "stale_threshold_hours": 48,
            },
        },
    }
    base["risk"]["macro_blackout"].update(macro_overrides)
    return base


# -- predicate (decision #1 — reason string with blocking date) -------------


@pytest.mark.asyncio
async def test_predicate_lifespan_overlap_blocks_when_event_inside(db_repos):
    """An event 10 days from today, with a 35-DTE position, is INSIDE the
    lifespan window — should block."""
    await _seed_event(db_repos, day=_today() + timedelta(days=10), etype="FOMC")
    cal = MacroCalendar(db_repos, _config())
    decision = await cal.is_blackout(
        today=_today(),
        short_expiration=_today() + timedelta(days=35),
        event_types=["FOMC", "CPI", "NFP"],
    )
    assert decision.in_blackout is True


@pytest.mark.asyncio
async def test_predicate_reason_string_includes_blocking_date(db_repos):
    """Reason encodes BOTH the event and the blocked expiration so post-hoc
    you can tell 'blocked because event tomorrow' from 'blocked because event
    30d into life'."""
    await _seed_event(db_repos, day=date(2026, 6, 17), etype="FOMC")
    cal = MacroCalendar(db_repos, _config())
    decision = await cal.is_blackout(
        today=date(2026, 6, 3),
        short_expiration=date(2026, 7, 18),
        event_types=["FOMC"],
    )
    assert decision.in_blackout is True
    assert decision.reason == "macro_blackout_FOMC_2026-06-17_blocks_expiration_2026-07-18"


@pytest.mark.asyncio
async def test_blackout_blocks_fomc_day(db_repos):
    """Per-ticket required case: event on the same day as `today`."""
    await _seed_event(db_repos, day=_today(), etype="FOMC")
    cal = MacroCalendar(db_repos, _config())
    decision = await cal.is_blackout(
        today=_today(), short_expiration=_today() + timedelta(days=14),
        event_types=["FOMC"],
    )
    assert decision.in_blackout is True


@pytest.mark.asyncio
async def test_blackout_blocks_day_before_cpi(db_repos):
    """Per-ticket required case: today is exactly one day BEFORE CPI →
    blackout window catches today."""
    await _seed_event(db_repos, day=_today() + timedelta(days=1), etype="CPI")
    cal = MacroCalendar(db_repos, _config(blackout_days_before=1, blackout_days_after=0))
    decision = await cal.is_blackout(
        today=_today(), short_expiration=_today() + timedelta(days=14),
        event_types=["CPI"],
    )
    assert decision.in_blackout is True


@pytest.mark.asyncio
async def test_event_outside_lifespan_does_not_block(db_repos):
    """Event 90 days out vs 35-DTE position → no overlap, no block."""
    await _seed_event(db_repos, day=_today() + timedelta(days=90), etype="FOMC")
    cal = MacroCalendar(db_repos, _config())
    decision = await cal.is_blackout(
        today=_today(), short_expiration=_today() + timedelta(days=35),
        event_types=["FOMC"],
    )
    assert decision.in_blackout is False
    assert decision.reason is None


@pytest.mark.asyncio
async def test_event_types_filter_respected(db_repos):
    """An event with event_type=GDP doesn't block when config only lists FOMC."""
    await _seed_event(db_repos, day=_today() + timedelta(days=5), etype="GDP")
    cal = MacroCalendar(db_repos, _config(event_types=["FOMC"]))
    decision = await cal.is_blackout(
        today=_today(), short_expiration=_today() + timedelta(days=14),
        event_types=["FOMC"],
    )
    assert decision.in_blackout is False


@pytest.mark.asyncio
async def test_blackout_disabled_does_not_block(db_repos):
    """Per-ticket required case: enabled=false in config → rule short-circuits
    (verified through the gate, not the predicate which is config-agnostic)."""
    # The predicate itself is config-agnostic; the rule reads `enabled`. Test
    # via the rule integration:
    from risk.limits import RiskGate
    from platforms.paper_broker import PaperBroker
    await _seed_event(db_repos, day=_today() + timedelta(days=5), etype="FOMC")
    cfg = _config(enabled=False)
    cfg["wheel"] = {"buying_power_floor_pct": 0, "max_position_pct_of_account": 100,
                    "open_interest_min": 0, "volume_min": 0, "bid_ask_spread_max_pct": 100}
    gate = RiskGate(PaperBroker(cash=20_000), db_repos, cfg,
                    {"tickers": [UniverseEntry(symbol="F", name="Ford", tier=1, overrides={})],
                     "banned": [], "banned_rules": []})
    proposal = Proposal(
        symbol="F",
        contract=OptionContract(
            underlying="F", occ_symbol="F260706P00010000", strike=10.0,
            expiration=_today() + timedelta(days=14), option_type=OptionType.PUT,
            bid=0.40, ask=0.42,
        ),
        order_type=OrderType.SELL_TO_OPEN, quantity=1, rationale="test",
    )
    result = await gate.evaluate(proposal, today=_today(), raise_on_fail=False)
    rule = next(r for r in result.results if r.rule == "macro_blackout")
    assert rule.status == "skip"
    assert "disabled" in rule.detail


# -- closes bypass + bear_call NOT bypassed (decision #2) -------------------


@pytest.mark.asyncio
async def test_closes_bypass_macro_blackout(db_repos):
    """Per-ticket required case: a BUY_TO_CLOSE proposal bypasses the rule
    even when an event is squarely in the lifespan window."""
    from risk.limits import RiskGate
    from platforms.paper_broker import PaperBroker
    await _seed_event(db_repos, day=_today(), etype="FOMC")
    cfg = _config()
    cfg["wheel"] = {"buying_power_floor_pct": 0, "max_position_pct_of_account": 100,
                    "open_interest_min": 0, "volume_min": 0, "bid_ask_spread_max_pct": 100}
    gate = RiskGate(PaperBroker(cash=20_000), db_repos, cfg,
                    {"tickers": [UniverseEntry(symbol="F", name="Ford", tier=1, overrides={})],
                     "banned": [], "banned_rules": []})
    close_proposal = Proposal(
        symbol="F",
        contract=OptionContract(
            underlying="F", occ_symbol="F260706P00010000", strike=10.0,
            expiration=_today() + timedelta(days=14), option_type=OptionType.PUT,
            bid=0.40, ask=0.42,
        ),
        order_type=OrderType.BUY_TO_CLOSE, quantity=1, rationale="test close",
    )
    result = await gate.evaluate(close_proposal, today=_today(), raise_on_fail=False)
    rule = next(r for r in result.results if r.rule == "macro_blackout")
    assert rule.status == "skip"
    assert "close" in rule.detail.lower()


@pytest.mark.asyncio
async def test_bear_call_proposals_are_blocked(db_repos):
    """Locks in decision #2: bear_call_spread does NOT bypass macro_blackout
    (unlike tier-2 LLM screen). FOMC gaps both directions."""
    from risk.limits import RiskGate
    from platforms.paper_broker import PaperBroker
    await _seed_event(db_repos, day=_today() + timedelta(days=5), etype="FOMC")
    cfg = _config()
    cfg["wheel"] = {"buying_power_floor_pct": 0, "max_position_pct_of_account": 100,
                    "open_interest_min": 0, "volume_min": 0, "bid_ask_spread_max_pct": 100}
    gate = RiskGate(
        PaperBroker(cash=20_000), db_repos, cfg,
        {"tickers": [UniverseEntry(symbol="SPY", name="SPDR S&P 500", tier=1, overrides={})],
         "banned": [], "banned_rules": []},
    )
    legs = [
        OrderLeg(
            contract_symbol="SPY260620C00500000", underlying="SPY",
            option_type=OptionType.CALL, strike=500.0,
            expiration=_today() + timedelta(days=14),
            action=OrderType.SELL_TO_OPEN,
        ),
        OrderLeg(
            contract_symbol="SPY260620C00505000", underlying="SPY",
            option_type=OptionType.CALL, strike=505.0,
            expiration=_today() + timedelta(days=14),
            action=OrderType.BUY_TO_OPEN,
        ),
    ]
    proposal = MultiLegProposal(
        symbol="SPY", legs=legs, net_credit_per_spread=0.30,
        max_loss_per_spread=470.0, width_dollars=5.0, quantity=1,
        rationale="bear_call test", strategy_id="bear_call_spread",
        direction="bear_call",
    )
    result = await gate.evaluate(proposal, today=_today(), raise_on_fail=False)
    rule = next(r for r in result.results if r.rule == "macro_blackout")
    assert rule.status == "fail", (
        f"bear_call must NOT bypass macro_blackout (outcome={rule.status}, detail={rule.detail})"
    )
    assert "FOMC" in rule.detail


# -- NYSE timezone (Issue 2) ------------------------------------------------


def test_macro_calendar_uses_nyse_timezone_for_today(monkeypatch):
    """At 04:00 UTC = 23:00 ET (previous day), _today_nyse must return the ET
    day (one BEFORE the UTC day). Catches the silent off-by-one a UTC-clock
    container would otherwise introduce."""
    from zoneinfo import ZoneInfo
    # June 2026 is in Eastern Daylight Time = UTC-4. 03:00 UTC on June 4
    # is 23:00 ET on June 3 (ET still on the 3rd, UTC already on the 4th).
    fake_utc_now = datetime(2026, 6, 4, 3, 0, 0, tzinfo=ZoneInfo("UTC"))
    nyse_now = fake_utc_now.astimezone(ZoneInfo("America/New_York"))
    assert nyse_now.date() == date(2026, 6, 3)  # ET is still on the 3rd

    class _FakeDatetime:
        @staticmethod
        def now(tz=None):
            return fake_utc_now if tz is None else fake_utc_now.astimezone(tz)
    monkeypatch.setattr("data.macro_calendar.datetime", _FakeDatetime)
    assert _today_nyse() == date(2026, 6, 3)


# -- Finnhub mapping (Issue 1) ----------------------------------------------


def test_finnhub_event_name_mapping_canonical():
    """Lock the canonical-mapping behaviour. When Finnhub renames events
    these will start writing OTHER and the dashboard will flag it."""
    assert map_finnhub_event_name("FOMC Statement") == "FOMC"
    assert map_finnhub_event_name("Fed Interest Rate Decision") == "FOMC"
    assert map_finnhub_event_name("CPI YoY") == "CPI"
    assert map_finnhub_event_name("Core CPI MoM") == "CPI"
    assert map_finnhub_event_name("Non Farm Payrolls") == "NFP"
    assert map_finnhub_event_name("Nonfarm Payrolls") == "NFP"  # punctuation variant
    assert map_finnhub_event_name("Unemployment Rate") == "NFP"
    assert map_finnhub_event_name("PPI MoM") == "PPI"
    assert map_finnhub_event_name("GDP QoQ") == "GDP"
    assert map_finnhub_event_name("JOLTS Job Openings") == "JOLTS"
    # Case-insensitive
    assert map_finnhub_event_name("fomc minutes") == "FOMC"
    # Unknown → OTHER (visible to operator on dashboard).
    assert map_finnhub_event_name("Random Speech By Senator") == "OTHER"
    assert map_finnhub_event_name("") == "OTHER"


def test_finnhub_core_cpi_does_not_match_plain_cpi(monkeypatch):
    """Substring matching would silently route 'Core CPI YoY' to the 'cpi
    yoy' key depending on dict iteration order; exact matching means an
    unknown variant correctly routes to OTHER instead of being mis-mapped.

    Proves the fix by removing the 'core cpi yoy' key and asserting the
    plain 'cpi yoy' key does NOT swallow it via substring fall-through.
    """
    monkeypatch.setattr(
        "data.macro_calendar._FINNHUB_TYPE_MAP",
        {"cpi yoy": "CPI"},  # only the plain key remains
    )
    # Plain CPI still maps correctly.
    assert map_finnhub_event_name("CPI YoY") == "CPI"
    # Core CPI must fall to OTHER — NOT be silently swallowed by the
    # substring "cpi yoy" inside "core cpi yoy".
    assert map_finnhub_event_name("Core CPI YoY") == "OTHER"


def test_finnhub_unknown_variant_falls_to_other():
    """Anything Finnhub renames or adds that the table doesn't cover routes
    to OTHER. The /macro page renders OTHER rows so the operator sees the
    drift and can extend `_FINNHUB_TYPE_MAP` accordingly."""
    # Geographic variants that share a substring with US series.
    assert map_finnhub_event_name("Mexico CPI YoY") == "OTHER"
    assert map_finnhub_event_name("EU Core CPI MoM") == "OTHER"
    # Brand-new event names Finnhub might add.
    assert map_finnhub_event_name("Some Brand New Indicator") == "OTHER"
    # Even a longer Powell-speaks variant doesn't false-positive — exact match.
    assert map_finnhub_event_name("Fed Chair Powell Speaks at Jackson Hole") == "OTHER"
    # Whitespace + case still normalize before the exact lookup.
    assert map_finnhub_event_name("  FOMC Statement  ") == "FOMC"


def test_parse_finnhub_events_filters_by_min_impact():
    """`impact_min=high` filters out medium/low events."""
    raw = [
        {"event": "FOMC Statement", "impact": "high", "time": "2026-06-17 14:00:00"},
        {"event": "Retail Sales MoM", "impact": "medium", "time": "2026-06-18"},
        {"event": "Random Index", "impact": "low", "date": "2026-06-19"},
    ]
    events = parse_finnhub_events(raw, min_impact="high")
    assert [e.event_type for e in events] == ["FOMC"]
    assert events[0].event_date == date(2026, 6, 17)


# -- YAML fallback + stale warning (decision #4) ----------------------------


def test_yaml_loader_parses_valid_file(tmp_path):
    yaml_path = tmp_path / "macro.yaml"
    yaml_path.write_text(yaml.safe_dump({
        "valid_until": "2026-12-31",
        "events": [
            {"date": "2026-06-17", "event_type": "FOMC", "impact": "high", "description": "Fed"},
            {"date": "2026-07-15", "event_type": "CPI", "impact": "high", "description": "CPI"},
        ],
    }))
    events, valid_until = load_yaml_calendar(yaml_path)
    assert len(events) == 2
    assert events[0].event_type == "FOMC"
    assert events[0].source == "yaml"
    assert valid_until == date(2026, 12, 31)


def test_yaml_loader_returns_empty_when_missing(tmp_path):
    events, valid_until = load_yaml_calendar(tmp_path / "does_not_exist.yaml")
    assert events == []
    assert valid_until is None


def test_yaml_stale_warning_fires_when_valid_until_within_30_days(monkeypatch, caplog):
    """Decision #4: maybe_log_yaml_stale fires when valid_until is within 30
    days (warning) and ramps to fail when past."""
    fake_today = date(2026, 6, 3)
    monkeypatch.setattr("data.macro_calendar._today_nyse", lambda: fake_today)
    captured = []
    def _capture(step, **fields):
        captured.append({"step": step, **fields})
    monkeypatch.setattr("data.macro_calendar.log_checkpoint", _capture)
    # Within 30 days but not past → warn
    maybe_log_yaml_stale(fake_today + timedelta(days=20))
    assert captured[-1]["step"] == "macro_yaml_stale_human_attention"
    assert captured[-1]["status"] == "ok"
    # Past → fail
    maybe_log_yaml_stale(fake_today - timedelta(days=1))
    assert captured[-1]["step"] == "macro_yaml_stale_human_attention"
    assert captured[-1]["status"] == "fail"
    # >30 days out → no log
    before = len(captured)
    maybe_log_yaml_stale(fake_today + timedelta(days=60))
    assert len(captured) == before


# -- AlertRateLimitsRepo (decision #3 + #5) ---------------------------------


@pytest.mark.asyncio
async def test_alert_rate_limit_first_fire_succeeds(db_repos):
    fired = await db_repos.alert_rate_limits.try_fire("test_key", cooldown_hours=20.0)
    assert fired is True
    last = await db_repos.alert_rate_limits.last_fired("test_key")
    assert last is not None


@pytest.mark.asyncio
async def test_stale_alert_rate_limited_to_daily(db_repos):
    """Within the cooldown the alert silently no-ops; second fire returns False."""
    assert await db_repos.alert_rate_limits.try_fire("macro_calendar_stale", 20.0) is True
    # Immediate retry within cooldown → blocked.
    assert await db_repos.alert_rate_limits.try_fire("macro_calendar_stale", 20.0) is False
    # Simulate 25h passing by manually backdating the row.
    c = await db_repos.db.connect()
    await c.execute(
        "UPDATE alert_rate_limits SET last_fired_at = ? WHERE alert_key = ?",
        ((datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=25)).isoformat(),
         "macro_calendar_stale"),
    )
    await c.commit()
    # Now cooldown elapsed → fires again.
    assert await db_repos.alert_rate_limits.try_fire("macro_calendar_stale", 20.0) is True


# -- Fail-open paths --------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_table_fails_open_with_rate_limited_alert(db_repos, monkeypatch):
    """`macro_events` empty → rule passes with skip + alert fires once."""
    captured: list[dict] = []
    async def _capture_notify(event, message, **fields):
        captured.append({"event": event, **fields})
    monkeypatch.setattr("data.macro_calendar.notify", _capture_notify, raising=False)
    # Import notify from where it's referenced inside maybe_alert_empty:
    monkeypatch.setattr("core.notify.notify", _capture_notify)
    cal = MacroCalendar(db_repos, _config())
    await cal.maybe_alert_empty()
    assert len(captured) == 1 and captured[0]["event"] == "macro.calendar_empty"
    # Second call within cooldown is silent.
    await cal.maybe_alert_empty()
    assert len(captured) == 1


@pytest.mark.asyncio
async def test_is_stale_when_never_refreshed(db_repos):
    cal = MacroCalendar(db_repos, _config())
    assert await cal.is_stale() is True


@pytest.mark.asyncio
async def test_is_stale_false_after_recent_refresh(db_repos):
    await _seed_event(db_repos, day=_today(), etype="FOMC")
    cal = MacroCalendar(db_repos, _config())
    assert await cal.is_stale() is False


# -- Idempotent refresh (Issue addendum) ------------------------------------


@pytest.mark.asyncio
async def test_idempotent_refresh_no_duplicates(db_repos):
    """Running the upsert twice with the same event yields 1 row in the table
    (UNIQUE(event_date, event_type) is the idempotency key)."""
    evt = MacroEvent(
        event_date=date(2026, 6, 17), event_type="FOMC", impact="high",
        description="FOMC test",
        fetched_at=datetime.now(UTC).replace(tzinfo=None),
        created_at=datetime.now(UTC).replace(tzinfo=None),
        source="yaml",
    )
    await db_repos.macro_events.upsert_many([evt])
    await db_repos.macro_events.upsert_many([evt])
    assert await db_repos.macro_events.count() == 1


@pytest.mark.asyncio
async def test_upsert_many_returns_attempts_and_distinct_rows(db_repos):
    """4 items mapping to 2 distinct (event_date, event_type) tuples should
    report (4, 2) — locks in the honest-logging fix for the misleading
    'rows_upserted=50' we hit in production with only 18 distinct rows."""
    now = datetime.now(UTC).replace(tzinfo=None)
    # Simulating Finnhub: FOMC Statement + Press Conference same day, then
    # CPI YoY + Core CPI MoM same day — all collapse via UNIQUE.
    events = [
        MacroEvent(event_date=date(2026, 6, 17), event_type="FOMC", impact="high",
                   description="FOMC Statement", fetched_at=now, created_at=now),
        MacroEvent(event_date=date(2026, 6, 17), event_type="FOMC", impact="high",
                   description="FOMC Press Conference", fetched_at=now, created_at=now),
        MacroEvent(event_date=date(2026, 7, 15), event_type="CPI", impact="high",
                   description="CPI YoY", fetched_at=now, created_at=now),
        MacroEvent(event_date=date(2026, 7, 15), event_type="CPI", impact="high",
                   description="Core CPI MoM", fetched_at=now, created_at=now),
    ]
    attempts, distinct = await db_repos.macro_events.upsert_many(events)
    assert attempts == 4
    assert distinct == 2
    assert await db_repos.macro_events.count() == 2


# -- apply_pending (auto-migrate on bot startup) ----------------------------


def test_apply_pending_runs_unapplied_migrations(tmp_path, monkeypatch):
    """The programmatic helper applies every unapplied migration and returns
    the list. Locks in the contract scripts/run_bot.py depends on."""
    import sqlite3
    from scripts import run_migration

    # Build a fresh DB with only schema_migrations populated for 001.
    db_path = tmp_path / "wb.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE schema_migrations ("
        " version TEXT PRIMARY KEY, name TEXT NOT NULL, applied_at DATETIME)"
    )
    conn.execute(
        "INSERT INTO schema_migrations (version, name, applied_at) "
        "VALUES ('001', 'daily_state', '2026-01-01 00:00:00')"
    )
    conn.commit()
    conn.close()

    # Stub config so load_config() inside apply_pending finds our temp DB.
    monkeypatch.setattr(
        "scripts.run_migration.load_config",
        lambda: {"database": {"path": str(db_path)}},
    )
    # Point at a tiny migrations dir with two synthetic migrations.
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    (migrations_dir / "002_two.sql").write_text(
        "CREATE TABLE IF NOT EXISTS t2 (id INTEGER PRIMARY KEY);"
    )
    (migrations_dir / "003_three.sql").write_text(
        "CREATE TABLE IF NOT EXISTS t3 (id INTEGER PRIMARY KEY);"
    )
    monkeypatch.setattr("scripts.run_migration.MIGRATIONS_DIR", migrations_dir)

    applied = run_migration.apply_pending()
    assert [m.version for m in applied] == ["002", "003"]

    # Re-running is a no-op.
    applied_again = run_migration.apply_pending()
    assert applied_again == []

    # Tables created.
    conn = sqlite3.connect(db_path)
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    assert "t2" in tables and "t3" in tables
    conn.close()


def test_apply_pending_raises_on_missing_db(tmp_path, monkeypatch):
    """If the DB file doesn't exist (fresh install — bootstrap_db not run),
    apply_pending raises FileNotFoundError so the bot caller refuses to
    start and the operator sees the actionable error instead of a stack
    trace from later code."""
    from scripts import run_migration
    monkeypatch.setattr(
        "scripts.run_migration.load_config",
        lambda: {"database": {"path": str(tmp_path / "does_not_exist.db")}},
    )
    with pytest.raises(FileNotFoundError):
        run_migration.apply_pending()
