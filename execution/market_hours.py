"""US equities market-hours helper.

Used by execution/loop.py to pick its tick cadence (5 min in market hours,
30 min off). Intentionally minimal:

- Regular session only — 09:30 to 16:00 America/New_York, weekdays.
- No half-day / holiday awareness in v1. The cost of being wrong is the loop
  ticks more (or less) frequently for one day; orders themselves are placed
  with TimeInForce.DAY so the broker rejects out-of-session attempts anyway.
"""

from __future__ import annotations

from datetime import datetime, time, timezone
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
