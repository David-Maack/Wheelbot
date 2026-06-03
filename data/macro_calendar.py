"""Macro event blackout calendar — TICKET-007.

Two responsibilities:

  1. Populate `macro_events` from Finnhub `/calendar/economic` (or a
     hand-curated YAML fallback) — invoked by `scripts/refresh_macro_calendar`
     as a daily cron.
  2. Answer `is_blackout(today, short_expiration, event_types)` for the risk
     gate (`risk/limits.py::_rule_macro_blackout`) and the dashboard
     `/macro` page.

Both consumers share ONE predicate so the rule and the display can't disagree
(same source-of-truth pattern TICKET-006 established for earnings).

Timezone policy
---------------
Every date comparison uses **America/New_York** explicitly via `_today_nyse()`.
The container TZ is independent of the host TZ; if the container clock is in
UTC and you compare a naive `date.today()` against an NYSE event date, you
will silently shift by a day around midnight ET. This module never calls
`date.today()` directly — callers pass in the result of `_today_nyse()`.

YAML fallback
-------------
When `FINNHUB_API_KEY` is unset or Finnhub returns nothing, the loader falls
back to `config/macro_calendar.yaml`. The YAML must have a `valid_until` date
at the top; if it's within 30 days of today, the loader logs
`macro_yaml_stale_human_attention` so the operator knows the hand-curated
file needs refreshing.

Rate-limited Discord alerts
---------------------------
Stale-calendar and empty-table conditions both fire Discord events the FIRST
time per ~24h. Subsequent ticks within the cooldown silently no-op. This
prevents a flaky cron from spamming a channel while still surfacing if it
silently fails for days. See `db/migrations/009_add_alert_rate_limits.sql`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from core.checkpoint import log_checkpoint
from core.models import MacroEvent

_NYSE = ZoneInfo("America/New_York")


def _today_nyse() -> date:
    """Today in NYSE-local time. Use this everywhere instead of `date.today()`
    so a UTC-clock container doesn't silently shift by a day around midnight
    ET. Public so callers (risk gate, dashboard) can use it directly."""
    return datetime.now(_NYSE).date()


# Finnhub's `/calendar/economic` returns events with names like "FOMC Statement"
# or "Non Farm Payrolls" — we map case-insensitive substring matches to the
# canonical bucket the config uses ("FOMC" / "CPI" / "NFP" / ...). Anything
# that doesn't match becomes "OTHER", which is surfaced on the dashboard so
# operators notice when Finnhub renames an event and the mapping needs
# updating. Tested by `test_finnhub_event_name_mapping_canonical`.
_FINNHUB_TYPE_MAP: dict[str, str] = {
    "fomc statement": "FOMC",
    "fomc press conference": "FOMC",
    "fomc minutes": "FOMC",
    "fomc economic projections": "FOMC",
    "fed interest rate decision": "FOMC",
    "fed chair powell speaks": "FOMC",
    "cpi yoy": "CPI",
    "cpi mom": "CPI",
    "core cpi yoy": "CPI",
    "core cpi mom": "CPI",
    "non farm payrolls": "NFP",
    "nonfarm payrolls": "NFP",
    "unemployment rate": "NFP",
    "average hourly earnings mom": "NFP",
    "ppi yoy": "PPI",
    "ppi mom": "PPI",
    "core ppi yoy": "PPI",
    "core ppi mom": "PPI",
    "gdp qoq": "GDP",
    "gdp annualized": "GDP",
    "gdp annualized qoq": "GDP",
    "jolts job openings": "JOLTS",
}


def map_finnhub_event_name(raw_name: str) -> str:
    """Canonicalise a Finnhub event name. Returns 'OTHER' on no match so the
    operator can see (via the dashboard) when Finnhub renames an event."""
    if not raw_name:
        return "OTHER"
    needle = raw_name.lower().strip()
    for pattern, canonical in _FINNHUB_TYPE_MAP.items():
        if pattern in needle:
            return canonical
    return "OTHER"


@dataclass(frozen=True, slots=True)
class BlackoutDecision:
    """Returned by `MacroCalendar.is_blackout`. The reason string encodes both
    the blocking event AND the affected expiration so post-hoc analysis can
    distinguish 'blocked because event was tomorrow' from 'blocked because
    event was 30d into the position's life'."""

    in_blackout: bool
    reason: str | None             # None when in_blackout is False
    triggering_event: MacroEvent | None  # None when in_blackout is False


class MacroCalendar:
    """Loads macro events (Finnhub > YAML fallback) and answers blackout
    queries. One instance per process; takes repos for DB-backed queries.

    NOT thread-safe — async-safe (single asyncio loop assumption matches the
    rest of the bot)."""

    def __init__(self, repos: Any, config: dict[str, Any]) -> None:
        self._repos = repos
        self._config = config
        self._empty_warning_fired = False  # process-local backstop for the rate-limit table

    # -- predicate -----------------------------------------------------------

    async def is_blackout(
        self,
        *,
        today: date,
        short_expiration: date,
        event_types: list[str],
    ) -> BlackoutDecision:
        """True when ANY day in `[today, short_expiration]` falls within the
        configured `[event_date - days_before, event_date + days_after]`
        window for any event whose `event_type` is in `event_types`.

        Returns the FIRST matching event (events sorted by event_date), with a
        reason string encoding both the blocking event AND the blocked
        expiration:
            macro_blackout_FOMC_2026-06-17_blocks_expiration_2026-07-18
        """
        cfg = (self._config.get("risk", {}) or {}).get("macro_blackout", {}) or {}
        days_before = int(cfg.get("blackout_days_before", 1))
        days_after = int(cfg.get("blackout_days_after", 0))
        # Pull a slightly-wider window than [today, short_expiration] so any
        # event whose blackout window touches the position life is considered.
        events = await self._repos.macro_events.between(
            today - timedelta(days=days_after),
            short_expiration + timedelta(days=days_before),
        )
        wanted = {t.upper() for t in event_types}
        for evt in sorted(events, key=lambda e: e.event_date):
            if evt.event_type.upper() not in wanted:
                continue
            window_start = evt.event_date - timedelta(days=days_before)
            window_end = evt.event_date + timedelta(days=days_after)
            # Lifespan [today, short_expiration] overlaps [window_start, window_end]?
            if today <= window_end and short_expiration >= window_start:
                reason = (
                    f"macro_blackout_{evt.event_type}_{evt.event_date}"
                    f"_blocks_expiration_{short_expiration}"
                )
                return BlackoutDecision(in_blackout=True, reason=reason, triggering_event=evt)
        return BlackoutDecision(in_blackout=False, reason=None, triggering_event=None)

    # -- queries --------------------------------------------------------------

    async def upcoming_events(self, *, days_ahead: int) -> list[MacroEvent]:
        """Used by the dashboard /macro page."""
        today = _today_nyse()
        return await self._repos.macro_events.between(today, today + timedelta(days=days_ahead))

    async def last_refresh(self) -> datetime | None:
        return await self._repos.macro_events.last_fetched_at()

    async def is_stale(self) -> bool:
        """True when the cron hasn't refreshed within `stale_threshold_hours`."""
        cfg = (self._config.get("risk", {}) or {}).get("macro_blackout", {}) or {}
        threshold_hours = float(cfg.get("stale_threshold_hours", 48))
        last = await self.last_refresh()
        if last is None:
            return True  # never refreshed
        # Compare in UTC (DB stores naive UTC; last_fired is set with UTC now()).
        now = datetime.now(UTC).replace(tzinfo=None)
        return (now - last) > timedelta(hours=threshold_hours)

    async def maybe_alert_stale(self) -> None:
        """Fire a Discord event when the calendar is stale, rate-limited to
        once per ~20h via the alert_rate_limits table so a flaky cron doesn't
        spam and a process restart doesn't reset the silence."""
        if not await self.is_stale():
            return
        from core.notify import notify  # local import — keep module-import cheap
        if not await self._repos.alert_rate_limits.try_fire(
            "macro_calendar_stale", cooldown_hours=20.0,
        ):
            return
        last = await self.last_refresh()
        await notify(
            "macro.calendar_stale",
            "Macro calendar hasn't refreshed",
            last_refresh=str(last) if last else "never",
            stale_threshold_hours=float(
                (self._config.get("risk", {}) or {})
                .get("macro_blackout", {}).get("stale_threshold_hours", 48)
            ),
        )
        log_checkpoint(
            "macro_calendar_stale_fail_open",
            status="fail",
            last_refresh=str(last) if last else "never",
        )

    async def maybe_alert_empty(self) -> None:
        """Fire a Discord event when the table is empty (typically: cron was
        never set up). Rate-limited the same way."""
        count = await self._repos.macro_events.count()
        if count > 0:
            return
        from core.notify import notify
        if not await self._repos.alert_rate_limits.try_fire(
            "macro_calendar_empty", cooldown_hours=20.0,
        ):
            return
        await notify(
            "macro.calendar_empty",
            "Macro calendar is empty — refresh cron may not be running",
        )
        log_checkpoint("macro_calendar_empty", status="fail")


# -- ingest (called by scripts/refresh_macro_calendar) ----------------------


def _finnhub_fetch(*, days_ahead: int, api_key: str) -> list[dict[str, Any]] | None:
    """Pull raw events from Finnhub. Returns None on auth/network failure so
    the caller can fall back to YAML."""
    today = _today_nyse()
    end = today + timedelta(days=days_ahead)
    params = {"from": today.isoformat(), "to": end.isoformat(), "token": api_key}
    try:
        resp = httpx.get(
            "https://finnhub.io/api/v1/calendar/economic", params=params, timeout=10.0,
        )
        if resp.status_code in (401, 403):
            log_checkpoint("macro_finnhub_auth", status="fail")
            return None
        if resp.status_code == 429:
            log_checkpoint("macro_finnhub_rate_limit", status="skip")
            return None
        resp.raise_for_status()
        payload = resp.json()
    except httpx.HTTPError as exc:
        log_checkpoint("macro_finnhub_fail", status="fail", error=str(exc))
        return None
    return payload.get("economicCalendar") or []


def _coerce_event_date(raw: Any) -> date | None:
    if raw is None:
        return None
    if isinstance(raw, date) and not isinstance(raw, datetime):
        return raw
    if isinstance(raw, datetime):
        return raw.date()
    try:
        # Finnhub returns either "2026-06-17 14:00:00" or "2026-06-17".
        s = str(raw).split(" ")[0]
        return datetime.fromisoformat(s).date()
    except (ValueError, TypeError):
        return None


def parse_finnhub_events(
    raw_events: list[dict[str, Any]],
    *,
    min_impact: str = "high",
) -> list[MacroEvent]:
    """Filter, map, and convert raw Finnhub items to MacroEvent. Filters out
    items below `min_impact` (Finnhub uses 'high' | 'medium' | 'low')."""
    impact_order = {"low": 0, "medium": 1, "high": 2}
    min_level = impact_order.get(min_impact.lower(), 2)
    now = datetime.now(UTC).replace(tzinfo=None)
    out: list[MacroEvent] = []
    for item in raw_events:
        raw_impact = str(item.get("impact", "")).lower()
        if impact_order.get(raw_impact, 0) < min_level:
            continue
        evt_date = _coerce_event_date(item.get("time") or item.get("date"))
        if evt_date is None:
            continue
        raw_name = str(item.get("event") or "")
        event_type = map_finnhub_event_name(raw_name)
        out.append(MacroEvent(
            event_date=evt_date,
            event_type=event_type,
            impact=raw_impact or "high",
            description=raw_name or None,
            fetched_at=now,
            created_at=now,
            source="finnhub",
        ))
    return out


def load_yaml_calendar(path: Path | str | None = None) -> tuple[list[MacroEvent], date | None]:
    """Load events from `config/macro_calendar.yaml`. Returns (events,
    valid_until). The caller is expected to log
    `macro_yaml_stale_human_attention` when `valid_until < today + 30 days`.
    """
    import yaml  # local import — keeps module load cheap when YAML isn't used

    if path is None:
        path = Path(__file__).resolve().parent.parent / "config" / "macro_calendar.yaml"
    path = Path(path)
    if not path.exists():
        return [], None
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        log_checkpoint("macro_yaml_parse_fail", status="fail", error=str(exc))
        return [], None
    valid_until_raw = raw.get("valid_until")
    valid_until = _coerce_event_date(valid_until_raw)
    now = datetime.now(UTC).replace(tzinfo=None)
    out: list[MacroEvent] = []
    for item in raw.get("events") or []:
        evt_date = _coerce_event_date(item.get("date"))
        if evt_date is None:
            continue
        out.append(MacroEvent(
            event_date=evt_date,
            event_type=str(item.get("event_type", "OTHER")).upper(),
            impact=str(item.get("impact", "high")).lower(),
            description=item.get("description"),
            fetched_at=now,
            created_at=now,
            source="yaml",
        ))
    return out, valid_until


def maybe_log_yaml_stale(valid_until: date | None) -> None:
    """If the YAML's valid_until is within 30 days of today (or already past),
    log so the operator refreshes it before silently running on stale data."""
    if valid_until is None:
        return
    today = _today_nyse()
    days_remaining = (valid_until - today).days
    if days_remaining <= 30:
        log_checkpoint(
            "macro_yaml_stale_human_attention",
            status="fail" if days_remaining <= 0 else "ok",
            valid_until=str(valid_until),
            days_remaining=days_remaining,
        )


async def refresh_events(
    repos: Any,
    config: dict[str, Any],
    *,
    days_ahead: int = 60,
) -> tuple[int, str]:
    """Top-level refresh — used by scripts/refresh_macro_calendar.

    Returns (rows_upserted, source) where source is 'finnhub' or 'yaml'.
    Idempotent via the UNIQUE(event_date, event_type) constraint.
    """
    cfg = (config.get("risk", {}) or {}).get("macro_blackout", {}) or {}
    min_impact = str(cfg.get("finnhub_impact_min", "high"))

    events: list[MacroEvent] = []
    source = "yaml"
    api_key = os.environ.get("FINNHUB_API_KEY")
    if api_key:
        raw = _finnhub_fetch(days_ahead=days_ahead, api_key=api_key)
        if raw:
            events = parse_finnhub_events(raw, min_impact=min_impact)
            source = "finnhub"
    if not events:
        events, valid_until = load_yaml_calendar()
        maybe_log_yaml_stale(valid_until)

    attempts, distinct_rows = await repos.macro_events.upsert_many(events)
    log_checkpoint(
        "macro_refresh_done",
        status="ok",
        source=source,
        events_returned=len(events),
        upsert_attempts=attempts,
        distinct_rows_after=distinct_rows,
    )
    return distinct_rows, source
