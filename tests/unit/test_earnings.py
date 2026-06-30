"""data/earnings.in_blackout — entry earnings gate.

TICKET-030: the gate now also blocks a position that would be HELD THROUGH
earnings (the SOFI case: earnings inside the spread's life but expiry past the
old ±window), not just one whose expiration sits next to the event.
"""

from __future__ import annotations

from datetime import date

import pytest

from data import earnings as earnings_mod
from data.earnings import EarningsLookup, in_blackout

TODAY = date(2026, 6, 23)


def _stub_next_earnings(monkeypatch, d: date | None) -> None:
    monkeypatch.setattr(
        earnings_mod,
        "next_earnings",
        lambda *a, **k: EarningsLookup(symbol="X", next_date=d, source="test"),
    )


def test_spans_earnings_blocks_even_when_expiry_outside_window(monkeypatch):
    # SOFI shape: earnings 7/27 falls before a 7/31 expiry → held THROUGH
    # earnings → block, even though 7/31 is 4 days after (outside days_after=2).
    _stub_next_earnings(monkeypatch, date(2026, 7, 27))
    assert (
        in_blackout("X", date(2026, 7, 31), days_before=5, days_after=2, today=TODAY)
        is True
    )


def test_spans_disabled_falls_back_to_window_only(monkeypatch):
    # Same shape, block_if_spans=False → only the window rule → 4d after > 2 → pass.
    _stub_next_earnings(monkeypatch, date(2026, 7, 27))
    assert (
        in_blackout(
            "X",
            date(2026, 7, 31),
            days_before=5,
            days_after=2,
            today=TODAY,
            block_if_spans=False,
        )
        is False
    )


def test_earnings_after_expiry_passes(monkeypatch):
    # Earnings 8/15 is after the 7/31 expiry → not spanned, not near → pass.
    _stub_next_earnings(monkeypatch, date(2026, 8, 15))
    assert (
        in_blackout("X", date(2026, 7, 31), days_before=5, days_after=2, today=TODAY)
        is False
    )


def test_expiry_just_before_earnings_still_blocks_via_window(monkeypatch):
    # Earnings 7/27, expiry 7/24 (3d before, inside days_before=5). Not spanned
    # (earnings after expiry) but the ±window rule still blocks.
    _stub_next_earnings(monkeypatch, date(2026, 7, 27))
    assert (
        in_blackout("X", date(2026, 7, 24), days_before=5, days_after=2, today=TODAY)
        is True
    )


def test_no_earnings_data_returns_none(monkeypatch):
    _stub_next_earnings(monkeypatch, None)
    assert (
        in_blackout("X", date(2026, 7, 31), days_before=5, days_after=2, today=TODAY)
        is None
    )


def test_entry_proximity_blocks_imminent_earnings(monkeypatch):
    # entry_avoid_days mode: earnings 4 days out (within 7) → block the open.
    _stub_next_earnings(monkeypatch, date(2026, 6, 27))
    assert (
        in_blackout("X", date(2026, 8, 7), days_before=5, days_after=2,
                    today=TODAY, entry_avoid_days=7)
        is True
    )


def test_entry_proximity_allows_weeks_away_earnings_even_at_expiry(monkeypatch):
    # The case that sidelined the spreads: expiry == earnings (7/31), 38 days out.
    _stub_next_earnings(monkeypatch, date(2026, 7, 31))
    # Legacy expiry-based gate blocks (near_expiry).
    assert in_blackout("X", date(2026, 7, 31), days_before=5, days_after=2, today=TODAY) is True
    # Entry-proximity (7d) allows it — the 21-DTE close exits before that earnings.
    assert (
        in_blackout("X", date(2026, 7, 31), days_before=5, days_after=2,
                    today=TODAY, entry_avoid_days=7)
        is False
    )
