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
