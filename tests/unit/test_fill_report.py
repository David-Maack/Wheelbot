"""scripts/fill_report.summarise — the mid-fill test's arithmetic."""

from __future__ import annotations

import pytest

from scripts.fill_report import classify, summarise


def _order(oid: int, **kw) -> dict:
    base = {
        "id": oid,
        "symbol": "F",
        "contract_symbol": "F1",
        "client_order_id": f"wb-{oid}",
        "order_type": "SELL_TO_OPEN",
        "status": "FILLED",
        "limit_price": 0.40,
        "fill_price": 0.40,
        "placed_at": "2026-11-02T15:00:00",
        "filled_at": "2026-11-02T15:20:00",
        "trigger_reason": None,
        "raw_request": {"wb_quote": {"bid": 0.38, "ask": 0.42, "mid": 0.40, "resting": False}},
    }
    base.update(kw)
    return base


def _tape(oid: int, phase: str, bid: float, ask: float) -> dict:
    return {
        "order_id": oid,
        "phase": phase,
        "bid": bid,
        "ask": ask,
        "logged_at": "2026-11-02T15:05:00",
    }


def test_classify_splits_entries_resting_and_urgent():
    assert classify(_order(1)) == "entries"
    assert classify(_order(2, order_type="BUY_TO_CLOSE", client_order_id="wb-rest-x")) == "resting"
    assert classify(_order(3, order_type="BUY_TO_CLOSE")) == "urgent"


def test_entries_touch_fill_without_touch_spread_position_and_drift():
    orders = [
        _order(1),  # filled at mid, bid touched 0.40
        _order(
            2, placed_at="2026-11-02T15:16:00", status="CANCELLED", fill_price=None, filled_at=None
        ),
        _order(3, symbol="T", fill_price=0.40),  # filled, quote never reached it
    ]
    tape = [
        _tape(1, "working", 0.40, 0.43),
        _tape(1, "post_fill", 0.43, 0.45),
        _tape(2, "working", 0.37, 0.41),
        _tape(3, "working", 0.36, 0.42),
    ]
    s = summarise(orders, tape)["entries"]
    assert s["orders"] == 3 and s["filled"] == 2
    assert s["fill_rate_per_order"] == pytest.approx(2 / 3)
    assert s["symbol_days"] == 2 and s["fill_rate_per_symbol_day"] == 1.0
    assert s["touch_rate"] == pytest.approx(1 / 3)
    assert s["fills_without_touch"] == 1  # order 3: a paper artefact
    assert s["spread_position_median"] == pytest.approx(0.5)
    assert s["post_fill_drift_usd_mean"] == pytest.approx(4.0)  # mid 0.44 vs sold 0.40
    assert s["minutes_to_fill_median"] == pytest.approx(20.0)


def test_resting_and_urgent_close_metrics():
    rest_quote = {"wb_quote": {"bid": 0.18, "ask": 0.24, "mid": 0.21, "resting": True}}
    orders = [
        _order(
            10,
            order_type="BUY_TO_CLOSE",
            client_order_id="wb-rest-a",
            contract_symbol="F1",
            limit_price=0.20,
            fill_price=0.20,
            raw_request=rest_quote,
            trigger_reason="profit",
        ),
        _order(
            11,
            order_type="BUY_TO_CLOSE",
            client_order_id="wb-rest-b",
            contract_symbol="F2",
            limit_price=0.20,
            status="CANCELLED",
            fill_price=None,
            filled_at=None,
            trigger_reason="profit",
        ),
        _order(
            12,
            order_type="BUY_TO_CLOSE",
            contract_symbol="F2",
            limit_price=0.90,
            fill_price=0.90,
            trigger_reason="stop_loss",
            raw_request={"wb_quote": {"bid": 0.80, "ask": 0.90, "mid": 0.85}},
        ),
    ]
    tape = [_tape(10, "working", 0.17, 0.20), _tape(11, "working", 0.30, 0.34)]
    out = summarise(orders, tape)
    r = out["resting"]
    assert r["positions"] == 2 and r["positions_closed_by_resting_fill"] == 1
    assert r["cancelled_for_urgent_close"] == 1
    assert r["touch_rate"] == pytest.approx(0.5)
    assert r["spread_position_median"] == pytest.approx(4 / 6)  # bought 0.20 inside 0.18/0.24
    u = out["urgent"]
    assert u["by_trigger"] == {"stop_loss": 1}
    assert u["cost_vs_mid_usd_mean"] == pytest.approx(5.0)
