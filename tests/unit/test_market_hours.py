"""execution/market_hours sanity checks."""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from execution.market_hours import is_market_hours

ET = ZoneInfo("America/New_York")


def _et(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=ET).astimezone(timezone.utc)


def test_weekday_during_session_is_true():
    # Mon Jun 2 2025, 10:00 ET — within 09:30-16:00.
    assert is_market_hours(_et(2025, 6, 2, 10, 0)) is True


def test_weekday_at_open_is_true():
    assert is_market_hours(_et(2025, 6, 2, 9, 30)) is True


def test_weekday_at_close_is_false():
    # 16:00 ET is the close — exclusive upper bound.
    assert is_market_hours(_et(2025, 6, 2, 16, 0)) is False


def test_weekday_after_hours_is_false():
    assert is_market_hours(_et(2025, 6, 2, 17, 0)) is False


def test_weekday_before_open_is_false():
    assert is_market_hours(_et(2025, 6, 2, 8, 0)) is False


def test_saturday_is_false():
    assert is_market_hours(_et(2025, 6, 7, 10, 0)) is False


def test_sunday_is_false():
    assert is_market_hours(_et(2025, 6, 8, 10, 0)) is False


def test_naive_datetime_treated_as_utc():
    naive = datetime(2025, 6, 2, 14, 0)  # 14:00 UTC = 10:00 ET → True
    assert is_market_hours(naive) is True
