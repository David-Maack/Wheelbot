"""US equities market-hours helper.

Used by execution/loop.py to pick its tick cadence (5 min in market hours,
30 min off). Intentionally minimal:

- Regular session only — 09:30 to 16:00 America/New_York, weekdays.
- No half-day / holiday awareness in v1. The cost of being wrong is the loop
  ticks more (or less) frequently for one day; orders themselves are placed
  with TimeInForce.DAY so the broker rejects out-of-session attempts anyway.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
OPEN = time(9, 30)
CLOSE = time(16, 0)


def is_market_hours(now: datetime | None = None) -> bool:
    moment = now if now is not None else datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    et = moment.astimezone(ET)
    if et.weekday() >= 5:  # 5=Sat, 6=Sun
        return False
    return OPEN <= et.time() < CLOSE


def within_entry_window(
    now: datetime | None = None,
    *,
    minutes_before_close: int = 15,
) -> bool:
    """True when it's a weekday regular session AND at least
    `minutes_before_close` minutes remain before the 16:00 ET close.

    Used to gate NEW entries (CSPs, CCs, spread opens). Opening a position
    minutes before close is bad: little time for the limit to fill, and the
    bot can't manage it before the bell. Closes / rolls are NOT gated by
    this — exiting late is fine.
    """
    moment = now if now is not None else datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    et = moment.astimezone(ET)
    if et.weekday() >= 5:
        return False
    if et.time() < OPEN:
        return False
    close_dt = et.replace(
        hour=CLOSE.hour, minute=CLOSE.minute, second=0, microsecond=0
    )
    cutoff_dt = close_dt - timedelta(minutes=minutes_before_close)
    return et < cutoff_dt
