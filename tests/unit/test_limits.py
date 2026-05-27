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
    # Existing position on F → managing, not adding a new slot → rule skips.
    assert statuses["concurrent_positions_cap"] == "skip"


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
async def test_regime_bypasses_buy_to_close_even_when_csps_disallowed(db_repos, monkeypatch):
    """Profit-close of a CSP must not be blocked by an unfavorable regime —
    we never want to be unable to exit a position because SPY fell."""
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

    close_proposal = Proposal(
        symbol="F",
        contract=_put_contract(),
        order_type=OrderType.BUY_TO_CLOSE,
        quantity=1,
        rationale="profit close",
    )
    res = await gate.evaluate(close_proposal, today=date(2025, 6, 1), raise_on_fail=False)
    statuses = {r.rule: r.status for r in res.results}
    assert statuses["regime"] == "skip"


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


# -- Global concurrent-total cap (Sprint 12 sub-sprint 4) -------------------


async def _seed_active_position(
    db_repos, *, symbol: str, strategy_id: str, state: str = "CSP_OPEN"
) -> None:
    """Persist a non-IDLE position so list_active counts it."""
    now = datetime.now(UTC).replace(tzinfo=None)
    await db_repos.positions.insert(
        Position(
            account_id="test",
            symbol=symbol,
            strategy_id=strategy_id,
            state=PositionState(state),
            shares=0,
            state_changed_at=now,
        )
    )


@pytest.mark.asyncio
async def test_concurrent_total_skips_when_setting_absent(db_repos, monkeypatch):
    """Backwards-compat: configs without account.max_concurrent_total skip the rule."""
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)
    cfg = _config()  # no max_concurrent_total in account section
    gate = RiskGate(PaperBroker(cash=20_000), db_repos, cfg, _universe())
    res = await gate.evaluate(_proposal(), today=date(2025, 6, 1), raise_on_fail=False)
    statuses = {r.rule: r.status for r in res.results}
    assert statuses["concurrent_total_cap"] == "skip"


@pytest.mark.asyncio
async def test_concurrent_total_passes_below_global_cap(db_repos, monkeypatch):
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)
    # Seed 2 active positions across two different strategies. Cap is 4.
    await _seed_active_position(db_repos, symbol="AAA", strategy_id="monthly_wheel")
    await _seed_active_position(db_repos, symbol="BBB", strategy_id="put_spread", state="SPREAD_OPEN")

    cfg = _config()
    cfg["account"]["max_concurrent_total"] = 4
    gate = RiskGate(PaperBroker(cash=20_000), db_repos, cfg, _universe())
    res = await gate.evaluate(_proposal(), today=date(2025, 6, 1), raise_on_fail=False)
    statuses = {r.rule: r.status for r in res.results}
    assert statuses["concurrent_total_cap"] == "pass"


@pytest.mark.asyncio
async def test_concurrent_total_blocks_new_entry_at_cap_across_strategies(db_repos, monkeypatch):
    """Cap is 4. 4 different symbols already active (any strategies). New entry rejected
    even if the per-strategy cap would still allow it."""
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)
    await _seed_active_position(db_repos, symbol="AAA", strategy_id="monthly_wheel")
    await _seed_active_position(db_repos, symbol="BBB", strategy_id="weekly_wheel")
    await _seed_active_position(db_repos, symbol="CCC", strategy_id="put_spread", state="SPREAD_OPEN")
    await _seed_active_position(db_repos, symbol="DDD", strategy_id="bear_call_spread", state="SPREAD_OPEN")

    cfg = _config()
    cfg["account"]["max_concurrent_total"] = 4
    gate = RiskGate(PaperBroker(cash=20_000), db_repos, cfg, _universe())
    # Proposal is on F (a different symbol from the 4 active). Adds a new slot.
    res = await gate.evaluate(_proposal(), today=date(2025, 6, 1), raise_on_fail=False)
    statuses = {r.rule: r.status for r in res.results}
    assert statuses["concurrent_total_cap"] == "fail"


@pytest.mark.asyncio
async def test_concurrent_total_allows_proposal_on_existing_symbol_at_cap(db_repos, monkeypatch):
    """At cap, but the proposal is on a symbol we already hold → managing an
    existing position, cap skipped entirely. Critical for orphan-position
    management when the account is over-cap from historical opens."""
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)
    # 4 active including F (the proposal symbol from _proposal()).
    await _seed_active_position(db_repos, symbol="F", strategy_id="monthly_wheel")
    await _seed_active_position(db_repos, symbol="BBB", strategy_id="weekly_wheel")
    await _seed_active_position(db_repos, symbol="CCC", strategy_id="put_spread", state="SPREAD_OPEN")
    await _seed_active_position(db_repos, symbol="DDD", strategy_id="bear_call_spread", state="SPREAD_OPEN")

    cfg = _config()
    cfg["account"]["max_concurrent_total"] = 4
    gate = RiskGate(PaperBroker(cash=20_000), db_repos, cfg, _universe())
    res = await gate.evaluate(_proposal(), today=date(2025, 6, 1), raise_on_fail=False)
    statuses = {r.rule: r.status for r in res.results}
    assert statuses["concurrent_total_cap"] == "skip"


@pytest.mark.asyncio
async def test_concurrent_total_blocks_new_strategy_on_same_symbol_at_cap(db_repos, monkeypatch):
    """Regression for 2026-05-27 COIN incident: orphan-management cap-skip
    must NOT leak through to a new position on a different strategy that
    happens to share the symbol. Same-symbol cap-skip only applies to the
    same (symbol, strategy) pair."""
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)
    # 4 active positions, including COIN under weekly_wheel (SHARES_HELD).
    await _seed_active_position(db_repos, symbol="COIN", strategy_id="weekly_wheel", state="SHARES_HELD")
    await _seed_active_position(db_repos, symbol="BBB", strategy_id="weekly_wheel")
    await _seed_active_position(db_repos, symbol="CCC", strategy_id="put_spread", state="SPREAD_OPEN")
    await _seed_active_position(db_repos, symbol="DDD", strategy_id="bear_call_spread", state="SPREAD_OPEN")

    cfg = _config()
    cfg["account"]["max_concurrent_total"] = 4
    gate = RiskGate(PaperBroker(cash=20_000), db_repos, cfg, _universe())

    # Build a MultiLegProposal on COIN under put_spread (different strategy
    # from the existing COIN/weekly_wheel SHARES_HELD position).
    coin_proposal = _multi_leg_proposal(direction="bull_put")
    # _multi_leg_proposal builds with symbol="F" by default; override fields
    # to make it a COIN proposal under put_spread.
    from core.models import OrderLeg
    from strategies.spreads import MultiLegProposal
    coin_proposal = MultiLegProposal(
        symbol="COIN",
        legs=coin_proposal.legs,
        net_credit_per_spread=1.50,
        max_loss_per_spread=350.0,
        width_dollars=5.0,
        quantity=1,
        rationale="put_spread on COIN",
        strategy_id="put_spread",
        direction="bull_put",
    )

    res = await gate.evaluate(coin_proposal, today=date(2025, 6, 1), raise_on_fail=False)
    statuses = {r.rule: r.status for r in res.results}
    # Different strategy on same symbol = NEW exposure, must NOT skip the cap.
    # Account is at-cap (4 active) and this would add a 5th → fail.
    assert statuses["concurrent_total_cap"] == "fail"


@pytest.mark.asyncio
async def test_concurrent_total_allows_management_when_over_cap(db_repos, monkeypatch):
    """Regression for 2026-05-27 COIN situation: account is OVER cap due to
    historical positions opened before the cap was tightened. CC proposal on
    a SHARES_HELD position must still pass — managing existing exposure is
    not subject to the cap."""
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)
    # 5 active positions including F (over the cap of 4).
    await _seed_active_position(db_repos, symbol="F", strategy_id="monthly_wheel", state="SHARES_HELD")
    await _seed_active_position(db_repos, symbol="AAA", strategy_id="monthly_wheel")
    await _seed_active_position(db_repos, symbol="BBB", strategy_id="weekly_wheel")
    await _seed_active_position(db_repos, symbol="CCC", strategy_id="put_spread", state="SPREAD_OPEN")
    await _seed_active_position(db_repos, symbol="DDD", strategy_id="bear_call_spread", state="SPREAD_OPEN")

    cfg = _config()
    cfg["account"]["max_concurrent_total"] = 4
    gate = RiskGate(PaperBroker(cash=20_000), db_repos, cfg, _universe())
    res = await gate.evaluate(_proposal(), today=date(2025, 6, 1), raise_on_fail=False)
    # Proposal on F (existing symbol) — both caps should skip, not fail.
    statuses = {r.rule: r.status for r in res.results}
    assert statuses["concurrent_total_cap"] == "skip"
    assert statuses["concurrent_positions_cap"] == "skip"
