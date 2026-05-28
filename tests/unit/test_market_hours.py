"""execution/market_hours sanity checks."""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from execution.market_hours import is_market_hours, within_entry_window

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


# -- within_entry_window (Sprint 14) ----------------------------------------


def test_entry_window_open_mid_session():
    # Mon Jun 2 2025, 10:00 ET — well before the 15-min cutoff.
    assert within_entry_window(_et(2025, 6, 2, 10, 0)) is True


def test_entry_window_closed_inside_cutoff():
    # 15:50 ET — inside the final 15 min before 16:00 close.
    assert within_entry_window(_et(2025, 6, 2, 15, 50)) is False


def test_entry_window_open_just_before_cutoff():
    # 15:44 ET — one minute before the 15:45 cutoff → still open.
    assert within_entry_window(_et(2025, 6, 2, 15, 44)) is True


def test_entry_window_closed_at_cutoff_boundary():
    # 15:45 ET exactly — at the cutoff → closed (strict <).
    assert within_entry_window(_et(2025, 6, 2, 15, 45)) is False


def test_entry_window_closed_after_close():
    # 16:01 ET — after the bell.
    assert within_entry_window(_et(2025, 6, 2, 16, 1)) is False


def test_entry_window_closed_premarket():
    # 09:00 ET — before open.
    assert within_entry_window(_et(2025, 6, 2, 9, 0)) is False


def test_entry_window_closed_weekend():
    assert within_entry_window(_et(2025, 6, 8, 12, 0)) is False


def test_entry_window_custom_cutoff():
    # 30-min cutoff → 15:30 boundary. 15:40 ET is inside the cutoff → closed.
    assert within_entry_window(_et(2025, 6, 2, 15, 40), minutes_before_close=30) is False
    # 15:20 ET is before the 15:30 cutoff → open.
    assert within_entry_window(_et(2025, 6, 2, 15, 20), minutes_before_close=30) is True
