"""risk/limits.py — one test per §8 rule plus a composite happy path."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from core.models import (
    OptionContract,
    OptionType,
    OrderType,
    Position,
    PositionState,
    UniverseEntry,
)
from data import earnings as earnings_module
from platforms.paper_broker import PaperBroker
from risk.limits import RiskCheckFailed, RiskGate
from strategies.wheel import Proposal


@pytest.fixture(autouse=True)
def _clear_earnings_cache():
    earnings_module._clear_cache()


def _config(**wheel_overrides: Any) -> dict:
    base = {
        "account": {"id": "test", "broker": "paper"},
        "wheel": {
            "buying_power_floor_pct": 20,
            "max_position_pct_of_account": 30,
            "max_concurrent_positions": 4,
            "open_interest_min": 100,
            "volume_min": 50,
            "bid_ask_spread_max_pct": 10.0,
            "earnings_blackout_days_before": 5,
            "earnings_blackout_days_after": 2,
        },
        "regime": {"enabled": False},
    }
    base["wheel"].update(wheel_overrides)
    return base


def _universe() -> dict:
    return {
        "tickers": [UniverseEntry(symbol="F", name="Ford", tier=1, overrides={})],
        "banned": [],
        "banned_rules": [],
    }


def _put_contract(strike: float = 9.5, dte: int = 35, oi: int = 1000, vol: int = 200, bid: float = 0.39, ask: float = 0.41) -> OptionContract:
    today = date(2025, 6, 1)
    return OptionContract(
        underlying="F",
        occ_symbol="F250706P00009500",
        strike=strike,
        expiration=today + timedelta(days=dte),
        option_type=OptionType.PUT,
        bid=bid,
        ask=ask,
        delta=-0.25,
        open_interest=oi,
        volume=vol,
    )


def _proposal(contract: OptionContract | None = None, qty: int = 1) -> Proposal:
    contract = contract or _put_contract()
    return Proposal(
        symbol="F",
        contract=contract,
        order_type=OrderType.SELL_TO_OPEN,
        quantity=qty,
        rationale="test",
    )


@pytest.mark.asyncio
async def test_buying_power_floor_passes_when_room(db_repos, monkeypatch):
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)
    broker = PaperBroker(cash=10_000)
    gate = RiskGate(broker, db_repos, _config(), _universe())
    res = await gate.evaluate(_proposal(), today=date(2025, 6, 1), raise_on_fail=False)
    statuses = {r.rule: r.status for r in res.results}
    assert statuses["buying_power_floor"] == "pass"


@pytest.mark.asyncio
async def test_buying_power_floor_fails_when_too_tight(db_repos, monkeypatch):
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)
    # Strike 9.5 × 100 = $950 needed; at $1k cash floor 20% of equity ($1k) = $200
    # → BP after = 50, less than floor 200 → fail.
    broker = PaperBroker(cash=1_000)
    gate = RiskGate(broker, db_repos, _config(), _universe())
    res = await gate.evaluate(_proposal(), today=date(2025, 6, 1), raise_on_fail=False)
    statuses = {r.rule: r.status for r in res.results}
    assert statuses["buying_power_floor"] == "fail"


@pytest.mark.asyncio
async def test_per_position_cap_fails_for_oversized_notional(db_repos, monkeypatch):
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)
    broker = PaperBroker(cash=2_000)  # equity ~$2k; cap 30% = $600
    gate = RiskGate(broker, db_repos, _config(), _universe())
    res = await gate.evaluate(_proposal(qty=2), today=date(2025, 6, 1), raise_on_fail=False)
    statuses = {r.rule: r.status for r in res.results}
    assert statuses["per_position_cap"] == "fail"


@pytest.mark.asyncio
async def test_concurrent_positions_cap_blocks_fifth(db_repos, monkeypatch):
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)
    broker = PaperBroker(cash=20_000)
    now = datetime.now(UTC).replace(tzinfo=None)
    for sym in ("BAC", "SOFI", "NOK", "T"):
        await db_repos.positions.insert(
            Position(
                account_id="test",
                symbol=sym,
                state=PositionState.CSP_OPEN,
                shares=0,
                state_changed_at=now,
            )
        )
    gate = RiskGate(broker, db_repos, _config(max_concurrent_positions=4), _universe())
    res = await gate.evaluate(_proposal(), today=date(2025, 6, 1), raise_on_fail=False)
    statuses = {r.rule: r.status for r in res.results}
    assert statuses["concurrent_positions_cap"] == "fail"


@pytest.mark.asyncio
async def test_concurrent_cap_does_not_double_count_existing_symbol(db_repos, monkeypatch):
    """Rolling on F shouldn't count F as a new slot if it's already open."""
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)
    broker = PaperBroker(cash=20_000)
    now = datetime.now(UTC).replace(tzinfo=None)
    for sym in ("F", "BAC", "SOFI", "NOK"):
        await db_repos.positions.insert(
            Position(
                account_id="test",
                symbol=sym,
                state=PositionState.CSP_OPEN,
                shares=0,
                state_changed_at=now,
            )
        )
    gate = RiskGate(broker, db_repos, _config(max_concurrent_positions=4), _universe())
    res = await gate.evaluate(_proposal(), today=date(2025, 6, 1), raise_on_fail=False)
    statuses = {r.rule: r.status for r in res.results}
    assert statuses["concurrent_positions_cap"] == "pass"


@pytest.mark.asyncio
async def test_earnings_blackout_skipped_when_no_data(db_repos, monkeypatch):
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)
    broker = PaperBroker(cash=20_000)
    gate = RiskGate(broker, db_repos, _config(), _universe())
    res = await gate.evaluate(_proposal(), today=date(2025, 6, 1), raise_on_fail=False)
    statuses = {r.rule: r.status for r in res.results}
    assert statuses["earnings_blackout"] == "skip"


@pytest.mark.asyncio
async def test_earnings_blackout_fails_when_in_window(db_repos, monkeypatch):
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: True)
    broker = PaperBroker(cash=20_000)
    gate = RiskGate(broker, db_repos, _config(), _universe())
    res = await gate.evaluate(_proposal(), today=date(2025, 6, 1), raise_on_fail=False)
    statuses = {r.rule: r.status for r in res.results}
    assert statuses["earnings_blackout"] == "fail"


@pytest.mark.asyncio
async def test_liquidity_rule_fails_on_wide_spread(db_repos, monkeypatch):
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)
    broker = PaperBroker(cash=20_000)
    gate = RiskGate(broker, db_repos, _config(), _universe())
    contract = _put_contract(bid=0.30, ask=0.60)  # ~67% spread
    res = await gate.evaluate(_proposal(contract), today=date(2025, 6, 1), raise_on_fail=False)
    statuses = {r.rule: r.status for r in res.results}
    assert statuses["liquidity"] == "fail"


@pytest.mark.asyncio
async def test_regime_skipped_when_disabled(db_repos, monkeypatch):
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)
    broker = PaperBroker(cash=20_000)
    gate = RiskGate(broker, db_repos, _config(), _universe())
    res = await gate.evaluate(_proposal(), today=date(2025, 6, 1), raise_on_fail=False)
    statuses = {r.rule: r.status for r in res.results}
    assert statuses["regime"] == "skip"


@pytest.mark.asyncio
async def test_regime_blocks_csp_when_csps_disallowed(db_repos, monkeypatch):
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)
    conn = await db_repos.db.connect()
    await conn.execute(
        "INSERT INTO regime_snapshots (snapshot_date, csps_allowed) VALUES (?, ?)",
        ("2025-06-01", 0),
    )
    await conn.commit()

    broker = PaperBroker(cash=20_000)
    cfg = _config()
    cfg["regime"] = {"enabled": True}
    gate = RiskGate(broker, db_repos, cfg, _universe())
    res = await gate.evaluate(_proposal(), today=date(2025, 6, 1), raise_on_fail=False)
    statuses = {r.rule: r.status for r in res.results}
    assert statuses["regime"] == "fail"


@pytest.mark.asyncio
async def test_evaluate_raises_on_first_failure_by_default(db_repos, monkeypatch):
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)
    broker = PaperBroker(cash=1_000)  # too tight → BP failure
    gate = RiskGate(broker, db_repos, _config(), _universe())
    with pytest.raises(RiskCheckFailed):
        await gate.evaluate(_proposal(), today=date(2025, 6, 1))


@pytest.mark.asyncio
async def test_happy_path_all_pass_or_skip(db_repos, monkeypatch):
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: False)
    broker = PaperBroker(cash=20_000)
    gate = RiskGate(broker, db_repos, _config(), _universe())
    res = await gate.evaluate(_proposal(), today=date(2025, 6, 1), raise_on_fail=False)
    assert res.passed


# -- Regime-gate dispatch on multi-leg direction ----------------------------


def _multi_leg_proposal(*, direction: str, order_type: OrderType = OrderType.MULTI_LEG_OPEN):
    """Minimal MultiLegProposal for regime-gate dispatch tests."""
    from core.models import OrderLeg
    from strategies.spreads import MultiLegProposal

    today = date(2025, 6, 1)
    if direction == "bear_call":
        legs = [
            OrderLeg(
                contract_symbol="F250706C00010000",
                underlying="F",
                option_type=OptionType.CALL,
                strike=10.0,
                expiration=today + timedelta(days=35),
                action=OrderType.SELL_TO_OPEN
                if order_type == OrderType.MULTI_LEG_OPEN
                else OrderType.BUY_TO_CLOSE,
            ),
            OrderLeg(
                contract_symbol="F250706C00011000",
                underlying="F",
                option_type=OptionType.CALL,
                strike=11.0,
                expiration=today + timedelta(days=35),
                action=OrderType.BUY_TO_OPEN
                if order_type == OrderType.MULTI_LEG_OPEN
                else OrderType.SELL_TO_CLOSE,
            ),
        ]
    else:
        legs = [
            OrderLeg(
                contract_symbol="F250706P00010000",
                underlying="F",
                option_type=OptionType.PUT,
                strike=10.0,
                expiration=today + timedelta(days=35),
                action=OrderType.SELL_TO_OPEN
                if order_type == OrderType.MULTI_LEG_OPEN
                else OrderType.BUY_TO_CLOSE,
            ),
            OrderLeg(
                contract_symbol="F250706P00009000",
                underlying="F",
                option_type=OptionType.PUT,
                strike=9.0,
                expiration=today + timedelta(days=35),
                action=OrderType.BUY_TO_OPEN
                if order_type == OrderType.MULTI_LEG_OPEN
                else OrderType.SELL_TO_CLOSE,
            ),
        ]
    return MultiLegProposal(
        symbol="F",
        legs=legs,
        net_credit_per_spread=0.30,
        max_loss_per_spread=70.0,
        width_dollars=1.0,
        quantity=1,
        rationale="test",
        strategy_id="put_spread" if direction == "bull_put" else "bear_call_spread",
        order_type=order_type,
        direction=direction,
    )


async def _seed_regime(db_repos, **flags: int) -> None:
    cols = ", ".join(["snapshot_date", *flags.keys()])
    placeholders = ", ".join(["?"] * (1 + len(flags)))
    conn = await db_repos.db.connect()
    await conn.execute(
        f"INSERT INTO regime_snapshots ({cols}) VALUES ({placeholders})",
        ("2025-06-01", *flags.values()),
    )
    await conn.commit()


@pytest.mark.asyncio
async def test_regime_gate_uses_bear_calls_allowed_for_bear_call_proposal(
    db_repos, monkeypatch
):
    """bear_call OPEN with bear_calls_allowed=0 fails the regime gate."""
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)
    await _seed_regime(db_repos, csps_allowed=1, bear_calls_allowed=0)

    cfg = _config()
    cfg["regime"] = {"enabled": True}
    gate = RiskGate(PaperBroker(cash=20_000), db_repos, cfg, _universe())
    res = await gate.evaluate(
        _multi_leg_proposal(direction="bear_call"),
        today=date(2025, 6, 1),
        raise_on_fail=False,
    )
    statuses = {r.rule: r.status for r in res.results}
    assert statuses["regime"] == "fail"


@pytest.mark.asyncio
async def test_regime_gate_uses_csps_allowed_for_bull_put_proposal(
    db_repos, monkeypatch
):
    """bull_put OPEN with csps_allowed=0 fails — bear_calls_allowed=1 must not unblock it."""
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)
    await _seed_regime(db_repos, csps_allowed=0, bear_calls_allowed=1)

    cfg = _config()
    cfg["regime"] = {"enabled": True}
    gate = RiskGate(PaperBroker(cash=20_000), db_repos, cfg, _universe())
    res = await gate.evaluate(
        _multi_leg_proposal(direction="bull_put"),
        today=date(2025, 6, 1),
        raise_on_fail=False,
    )
    statuses = {r.rule: r.status for r in res.results}
    assert statuses["regime"] == "fail"


@pytest.mark.asyncio
async def test_regime_gate_passes_bear_call_when_bear_calls_allowed(
    db_repos, monkeypatch
):
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)
    await _seed_regime(db_repos, csps_allowed=0, bear_calls_allowed=1)

    cfg = _config()
    cfg["regime"] = {"enabled": True}
    gate = RiskGate(PaperBroker(cash=20_000), db_repos, cfg, _universe())
    res = await gate.evaluate(
        _multi_leg_proposal(direction="bear_call"),
        today=date(2025, 6, 1),
        raise_on_fail=False,
    )
    statuses = {r.rule: r.status for r in res.results}
    assert statuses["regime"] == "pass"


@pytest.mark.asyncio
async def test_regime_gate_skips_close_proposals(db_repos, monkeypatch):
    """Closes always bypass the regime gate so we never get stuck holding a position."""
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)
    await _seed_regime(db_repos, csps_allowed=0, bear_calls_allowed=0)

    cfg = _config()
    cfg["regime"] = {"enabled": True}
    gate = RiskGate(PaperBroker(cash=20_000), db_repos, cfg, _universe())
    for direction in ("bull_put", "bear_call"):
        res = await gate.evaluate(
            _multi_leg_proposal(direction=direction, order_type=OrderType.MULTI_LEG_CLOSE),
            today=date(2025, 6, 1),
            raise_on_fail=False,
        )
        statuses = {r.rule: r.status for r in res.results}
        assert statuses["regime"] == "skip", f"close for {direction} should bypass regime"
