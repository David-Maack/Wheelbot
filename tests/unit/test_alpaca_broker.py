"""platforms/alpaca_broker — pure-function helpers.

Constructing AlpacaBroker requires real creds, so these tests focus on the
helpers that don't need a session — the OCC parser and the status mapper.
The status mapper bug (returning PENDING for FILLED orders because str(enum)
returns the qualname not the value) is the regression we care about.
"""

from __future__ import annotations

from enum import Enum

import pytest

from core.models import OrderStatus
from platforms.alpaca_broker import _map_status, _parse_occ


# Mimic alpaca-py's OrderStatus shape: subclass of str + Enum, value is the
# canonical lowercase string.
class _FakeAlpacaStatus(str, Enum):
    NEW = "new"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    ACCEPTED = "accepted"


def test_map_status_filled_via_enum_value():
    """The bug: str(_FakeAlpacaStatus.FILLED) returns
    '_FakeAlpacaStatus.FILLED', not 'filled'. Our mapper must use .value."""
    assert _map_status(_FakeAlpacaStatus.FILLED) == OrderStatus.FILLED
    assert _map_status(_FakeAlpacaStatus.PARTIALLY_FILLED) == OrderStatus.PARTIAL
    assert _map_status(_FakeAlpacaStatus.NEW) == OrderStatus.PENDING
    assert _map_status(_FakeAlpacaStatus.ACCEPTED) == OrderStatus.PENDING
    assert _map_status(_FakeAlpacaStatus.CANCELED) == OrderStatus.CANCELLED
    assert _map_status(_FakeAlpacaStatus.REJECTED) == OrderStatus.REJECTED


def test_map_status_accepts_plain_strings():
    """Defensive: callers should be able to pass either the enum or its value."""
    assert _map_status("filled") == OrderStatus.FILLED
    assert _map_status("new") == OrderStatus.PENDING
    assert _map_status("canceled") == OrderStatus.CANCELLED
    assert _map_status("rejected") == OrderStatus.REJECTED


def test_map_status_uppercase_strings_also_work():
    assert _map_status("FILLED") == OrderStatus.FILLED


def test_map_status_unknown_falls_back_to_pending():
    assert _map_status("some_new_alpaca_state") == OrderStatus.PENDING


def test_map_status_done_for_day_and_calculated_are_pending_not_rejected():
    """Finding #9: `calculated` (post-fill settlement accounting) and
    `done_for_day` are NOT rejections. Mapping them to REJECTED would make the
    reconciler treat a filled/unfinished order as cancelled and reset the
    position. They must hold as PENDING until a definitive terminal status."""
    assert _map_status("calculated") == OrderStatus.PENDING
    assert _map_status("done_for_day") == OrderStatus.PENDING
    # Genuine rejections still map to REJECTED.
    assert _map_status("rejected") == OrderStatus.REJECTED
    assert _map_status("suspended") == OrderStatus.REJECTED
    assert _map_status("stopped") == OrderStatus.REJECTED


def test_parse_occ_round_trip():
    parsed = _parse_occ("F250620P00010000")
    assert parsed is not None
    underlying, exp, opt_type, strike = parsed
    from datetime import date
    from core.models import OptionType
    assert underlying == "F"
    assert exp == date(2025, 6, 20)
    assert opt_type == OptionType.PUT
    assert strike == 10.0
