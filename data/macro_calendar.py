"""Macro event blackout calendar — TICKET-007.

Two responsibilities:

  1. Populate `macro_events` from the hand-kept `config/macro_calendar.yaml`
     — invoked by `scripts/refresh_macro_calendar` as a daily cron. Every row
     dated today-or-later is REPLACED by the YAML's events, so a date moved
     or removed in the file can't leave a phantom blackout behind.
  2. Answer `is_blackout(today, short_expiration, event_types)` for the risk
     gate (`risk/limits.py::_rule_macro_blackout`) and the dashboard
     `/macro` page.

Finnhub's `/calendar/economic` was the primary source until 2026-10-27. It
returns 403 on our plan, so since June the refresh had silently been
re-stamping an expired YAML; the table held nothing after 2026-09-16 while
the fetched_at-based staleness check stayed green. The Finnhub path is
gone, and "stale" now means the calendar runs out (see `is_stale`).

Both consumers share ONE predicate so the rule and the display can't disagree
(same source-of-truth pattern TICKET-006 established for earnings).

Timezone policy
---------------
Every date comparison uses **America/New_York** explicitly via `_today_nyse()`.
The container TZ is independent of the host TZ; if the container clock is in
UTC and you compare a naive `date.today()` against an NYSE event date, you
will silently shift by a day around midnight ET. This module never calls
`date.today()` directly — callers pass in the result of `_today_nyse()`.

YAML upkeep
-----------
The YAML carries a `valid_until` date at the top; within 30 days of it the
refresh logs `macro_yaml_stale_human_attention` so the operator extends the
file from the official calendars listed in its header.

Rate-limited Discord alerts
---------------------------
Stale-calendar and empty-table conditions both fire Discord events the FIRST
time per ~24h. Subsequent ticks within the cooldown silently no-op. This
prevents a flaky cron from spamming a channel while still surfacing if it
silently fails for days. See `db/migrations/009_add_alert_rate_limits.sql`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from core.checkpoint import log_checkpoint
from core.models import MacroEvent

_NYSE = ZoneInfo("America/New_York")

# Default for risk.macro_blackout.min_coverage_days: the calendar counts as
# stale once any configured event type has no event within this many days.
# FOMC meets ~every 6-7 weeks, so 60 leaves a month of warning before the
# first FOMC could be missed; CPI/NFP are monthly and always fall inside it.
DEFAULT_MIN_COVERAGE_DAYS = 60


def _today_nyse() -> date:
    """Today in NYSE-local time. Use this everywhere instead of `date.today()`
    so a UTC-clock container doesn't silently shift by a day around midnight
    ET. Public so callers (risk gate, dashboard) can use it directly."""
    return datetime.now(_NYSE).date()


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
    """Answers blackout and coverage queries over `macro_events` (filled from
    the YAML by `refresh_events`). One instance per process; takes repos for
    DB-backed queries.

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
        entry_avoid_days: int | None = None,
        days_before: int | None = None,
        days_after: int | None = None,
    ) -> BlackoutDecision:
        """True when ANY day in the considered window falls within the configured
        `[event_date - days_before, event_date + days_after]` window for any event
        whose `event_type` is in `event_types`.

        The considered window is `[today, short_expiration]` (legacy: blocks if any
        macro event falls anywhere in the position's life). When `entry_avoid_days`
        is set, it shrinks to `[today, today + entry_avoid_days]` — entry-proximity:
        only avoid OPENING right before an imminent event, instead of refusing any
        30-45 DTE position whose life merely reaches a routine monthly macro event.
        Mirrors the earnings entry-proximity gate.

        `days_before` / `days_after` override the config's blackout_days_* when
        passed (a per-strategy override; all three at 0 = the event day only).

        Returns the FIRST matching event with a reason string encoding the event
        AND the blocked expiration:
            macro_blackout_FOMC_2026-06-17_blocks_expiration_2026-07-18
        """
        cfg = (self._config.get("risk", {}) or {}).get("macro_blackout", {}) or {}
        if days_before is None:
            days_before = int(cfg.get("blackout_days_before", 1))
        if days_after is None:
            days_after = int(cfg.get("blackout_days_after", 0))
        horizon = (
            today + timedelta(days=entry_avoid_days)
            if entry_avoid_days is not None else short_expiration
        )
        # Pull a slightly-wider window than [today, horizon] so any event whose
        # blackout window touches the considered window is considered.
        events = await self._repos.macro_events.between(
            today - timedelta(days=days_after),
            horizon + timedelta(days=days_before),
        )
        wanted = {t.upper() for t in event_types}
        for evt in sorted(events, key=lambda e: e.event_date):
            if evt.event_type.upper() not in wanted:
                continue
            window_start = evt.event_date - timedelta(days=days_before)
            window_end = evt.event_date + timedelta(days=days_after)
            # Considered window [today, horizon] overlaps [window_start, window_end]?
            if today <= window_end and horizon >= window_start:
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

    def _coverage_cfg(self) -> tuple[list[str], int]:
        cfg = (self._config.get("risk", {}) or {}).get("macro_blackout", {}) or {}
        types = [str(t).upper() for t in (cfg.get("event_types") or [])]
        days = int(cfg.get("min_coverage_days", DEFAULT_MIN_COVERAGE_DAYS))
        return types, days

    async def coverage_end(self) -> date | None:
        """The last day the calendar can speak for: the EARLIEST of each
        configured event type's latest stored date. None when any configured
        type has no event at all."""
        types, _ = self._coverage_cfg()
        latest = await self._repos.macro_events.latest_dates(types)
        if not types or any(latest.get(t) is None for t in types):
            return None
        return min(latest[t] for t in types)

    async def is_stale(self) -> bool:
        """True when the calendar runs out within `min_coverage_days`.

        Staleness is about COVERAGE, not refresh age: the source is a static
        hand-kept file, so a fresh fetched_at proves nothing — the old
        fetched_at check stayed green for six weeks while the table held no
        future event. A stale calendar still answers for the dates it has;
        it only alerts so the YAML gets extended."""
        _, days = self._coverage_cfg()
        end = await self.coverage_end()
        return end is None or end < _today_nyse() + timedelta(days=days)

    async def maybe_alert_stale(self) -> None:
        """Fire a Discord event when the calendar is stale, rate-limited to
        once per ~20h via the alert_rate_limits table so the alert repeats
        daily without spamming every tick."""
        if not await self.is_stale():
            return
        from core.notify import notify  # local import — keep module-import cheap
        if not await self._repos.alert_rate_limits.try_fire(
            "macro_calendar_stale", cooldown_hours=20.0,
        ):
            return
        end = await self.coverage_end()
        await notify(
            "macro.calendar_stale",
            "Macro calendar is running out — extend config/macro_calendar.yaml",
            coverage_end=str(end) if end else "none",
        )
        log_checkpoint(
            "macro_calendar_stale",
            status="fail",
            coverage_end=str(end) if end else "none",
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


def _coerce_event_date(raw: Any) -> date | None:
    if raw is None:
        return None
    if isinstance(raw, date) and not isinstance(raw, datetime):
        return raw
    if isinstance(raw, datetime):
        return raw.date()
    try:
        # Accept "2026-06-17" and "2026-06-17 14:00:00".
        s = str(raw).split(" ")[0]
        return datetime.fromisoformat(s).date()
    except (ValueError, TypeError):
        return None


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
) -> tuple[int, str]:
    """Top-level refresh — used by scripts/refresh_macro_calendar and the MCP
    `refresh_macro_calendar` tool.

    Replaces every row dated today-or-later with the YAML's events (past rows
    are kept as history). An empty or unreadable YAML changes nothing — it
    must never wipe the future calendar. Returns (rows_after, "yaml").
    """
    events, valid_until = load_yaml_calendar()
    maybe_log_yaml_stale(valid_until)
    if not events:
        log_checkpoint("macro_refresh_no_events", status="fail")
        return await repos.macro_events.count(), "yaml"
    today = _today_nyse()
    future = [e for e in events if e.event_date >= today]
    rows_after = await repos.macro_events.replace_from(today, future)
    log_checkpoint(
        "macro_refresh_done",
        status="ok",
        source="yaml",
        future_events=len(future),
        valid_until=str(valid_until) if valid_until else None,
        rows_after=rows_after,
    )
    return rows_after, "yaml"
