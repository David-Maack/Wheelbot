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
async def test_close_passes_all_gates_on_tiny_account_near_bp_floor(db_repos, monkeypatch):
    """Finding #1: a CSP buyback must NOT be blocked by buying-power floor,
    per-position cap, or earnings. A buyback frees collateral; gating it would
    trap the bot in a losing position it's trying to stop out of."""
    # Earnings says we're inside a blackout window (would block an OPEN).
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: True)
    # Tiny account, high BP floor — an OPEN of this strike would be rejected.
    broker = PaperBroker(cash=1_000)
    cfg = _config(buying_power_floor_pct=90, max_position_pct_of_account=1)
    gate = RiskGate(broker, db_repos, cfg, _universe())

    close_proposal = Proposal(
        symbol="F",
        contract=_put_contract(),
        order_type=OrderType.BUY_TO_CLOSE,
        quantity=1,
        rationale="stop-loss close",
    )
    res = await gate.evaluate(close_proposal, today=date(2025, 6, 1), raise_on_fail=False)
    statuses = {r.rule: r.status for r in res.results}
    assert statuses["buying_power_floor"] == "skip"
    assert statuses["per_position_cap"] == "skip"
    assert statuses["earnings_blackout"] == "skip"
    # The close must pass overall — nothing should block an exit.
    assert res.passed


@pytest.mark.asyncio
async def test_open_still_blocked_by_bp_floor_on_tiny_account(db_repos, monkeypatch):
    """Counterpart: a new CSP entry IS still gated by the BP floor."""
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)
    broker = PaperBroker(cash=1_000)
    gate = RiskGate(broker, db_repos, _config(buying_power_floor_pct=90), _universe())
    res = await gate.evaluate(_proposal(), today=date(2025, 6, 1), raise_on_fail=False)
    statuses = {r.rule: r.status for r in res.results}
    assert statuses["buying_power_floor"] == "fail"


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
async def test_regime_gate_iron_condor_passes_when_both_wings_allowed(
    db_repos, monkeypatch
):
    """2026-07-01 audit: condor allowed wherever bullish premium is (csps_allowed
    — BULL_TREND or NEUTRAL). NEUTRAL (both flags) passes."""
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)
    await _seed_regime(db_repos, csps_allowed=1, bear_calls_allowed=1)

    cfg = _config()
    cfg["regime"] = {"enabled": True}
    gate = RiskGate(PaperBroker(cash=20_000), db_repos, cfg, _universe())
    res = await gate.evaluate(
        _multi_leg_proposal(direction="iron_condor"),
        today=date(2025, 6, 1),
        raise_on_fail=False,
    )
    statuses = {r.rule: r.status for r in res.results}
    assert statuses["regime"] == "pass"


@pytest.mark.asyncio
async def test_regime_gate_iron_condor_blocked_when_put_wing_disallowed(
    db_repos, monkeypatch
):
    """csps_allowed=0 (put wing blocked) → condor fails even if bear_calls=1."""
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)
    await _seed_regime(db_repos, csps_allowed=0, bear_calls_allowed=1)

    cfg = _config()
    cfg["regime"] = {"enabled": True}
    gate = RiskGate(PaperBroker(cash=20_000), db_repos, cfg, _universe())
    res = await gate.evaluate(
        _multi_leg_proposal(direction="iron_condor"),
        today=date(2025, 6, 1),
        raise_on_fail=False,
    )
    rule_results = {r.rule: r for r in res.results}
    assert rule_results["regime"].status == "fail"
    assert "BEAR_TREND/HIGH_VOL" in rule_results["regime"].detail


@pytest.mark.asyncio
async def test_regime_gate_iron_condor_passes_in_bull_trend(
    db_repos, monkeypatch
):
    """2026-07-01 audit fix: BULL_TREND (csps=1, bear_calls=0) now PASSES a
    condor — the old AND-combine required NEUTRAL, which a grinding bull never
    produces, making the strategy structurally zero-trade."""
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)
    await _seed_regime(db_repos, csps_allowed=1, bear_calls_allowed=0)

    cfg = _config()
    cfg["regime"] = {"enabled": True}
    gate = RiskGate(PaperBroker(cash=20_000), db_repos, cfg, _universe())
    res = await gate.evaluate(
        _multi_leg_proposal(direction="iron_condor"),
        today=date(2025, 6, 1),
        raise_on_fail=False,
    )
    statuses = {r.rule: r.status for r in res.results}
    assert statuses["regime"] == "pass"


@pytest.mark.asyncio
async def test_regime_gate_calendar_passes_when_range_bound(db_repos, monkeypatch):
    """Calendar passes wherever bullish premium is allowed (NEUTRAL here)."""
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)
    await _seed_regime(db_repos, csps_allowed=1, bear_calls_allowed=1)
    cfg = _config()
    cfg["regime"] = {"enabled": True}
    gate = RiskGate(PaperBroker(cash=20_000), db_repos, cfg, _universe())
    res = await gate.evaluate(
        _multi_leg_proposal(direction="calendar"),
        today=date(2025, 6, 1), raise_on_fail=False,
    )
    statuses = {r.rule: r.status for r in res.results}
    assert statuses["regime"] == "pass"


@pytest.mark.asyncio
async def test_regime_gate_calendar_blocked_when_bearish(db_repos, monkeypatch):
    """BEAR_TREND/HIGH_VOL (csps_allowed=0) still blocks the calendar; BULL_TREND
    no longer does (2026-07-01 audit fix)."""
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)
    await _seed_regime(db_repos, csps_allowed=0, bear_calls_allowed=1)
    cfg = _config()
    cfg["regime"] = {"enabled": True}
    gate = RiskGate(PaperBroker(cash=20_000), db_repos, cfg, _universe())
    res = await gate.evaluate(
        _multi_leg_proposal(direction="calendar"),
        today=date(2025, 6, 1), raise_on_fail=False,
    )
    rule_results = {r.rule: r for r in res.results}
    assert rule_results["regime"].status == "fail"
    assert "BEAR_TREND/HIGH_VOL" in rule_results["regime"].detail


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


# -- Tier-2 LLM screen gate (Sprint 14) -------------------------------------


def _tier2_universe() -> dict:
    return {
        "tickers": [
            UniverseEntry(symbol="F", name="Ford", tier=1, overrides={}),
            UniverseEntry(symbol="HOOD", name="Robinhood", tier=2, overrides={}),
        ],
        "banned": [],
        "banned_rules": [],
    }


def _hood_proposal() -> Proposal:
    today = date(2025, 6, 1)
    contract = OptionContract(
        underlying="HOOD",
        occ_symbol="HOOD250706P00070000",
        strike=70.0,
        expiration=today + timedelta(days=35),
        option_type=OptionType.PUT,
        bid=0.39, ask=0.41, delta=-0.25,
        open_interest=1000, volume=200,
    )
    return Proposal(
        symbol="HOOD",
        contract=contract,
        order_type=OrderType.SELL_TO_OPEN,
        quantity=1,
        rationale="csp test",
        strategy_id="weekly_wheel",
    )


async def _seed_candidate(db_repos, *, symbol: str, score: float, run_date: str | None = None):
    """Insert a candidates row for today (or given date).

    Uses the UTC date so it matches the rule's SQLite `date('now')` (UTC).
    date.today() is local and drifts off by a day near the UTC boundary.
    """
    import datetime as _dt
    if run_date is None:
        run_date = _dt.datetime.now(_dt.UTC).date().isoformat()
    conn = await db_repos.db.connect()
    await conn.execute(
        "INSERT INTO candidates (run_date, symbol, score, rationale) "
        "VALUES (?, ?, ?, ?)",
        (run_date, symbol, score, "test rationale"),
    )
    await conn.commit()


@pytest.mark.asyncio
async def test_tier2_screen_skips_when_screener_disabled(db_repos, monkeypatch):
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)
    cfg = _config()
    cfg["intelligence"] = {"llm_screener_enabled": False}
    gate = RiskGate(PaperBroker(cash=20_000), db_repos, cfg, _tier2_universe())
    res = await gate.evaluate(_hood_proposal(), today=date(2025, 6, 1), raise_on_fail=False)
    statuses = {r.rule: r.status for r in res.results}
    assert statuses["tier2_screen"] == "skip"


@pytest.mark.asyncio
async def test_tier2_screen_bypasses_tier1(db_repos, monkeypatch):
    """F is tier-1 — the screener gate must skip it."""
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)
    cfg = _config()
    cfg["intelligence"] = {"llm_screener_enabled": True, "tier2_min_score": 50}
    gate = RiskGate(PaperBroker(cash=20_000), db_repos, cfg, _tier2_universe())
    res = await gate.evaluate(_proposal(), today=date(2025, 6, 1), raise_on_fail=False)
    statuses = {r.rule: r.status for r in res.results}
    assert statuses["tier2_screen"] == "skip"


@pytest.mark.asyncio
async def test_tier2_screen_fails_when_no_candidate_row(db_repos, monkeypatch):
    """Tier-2 entry with no LLM screener row today → fail (screener didn't run)."""
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)
    cfg = _config()
    cfg["intelligence"] = {"llm_screener_enabled": True, "tier2_min_score": 50}
    gate = RiskGate(PaperBroker(cash=20_000), db_repos, cfg, _tier2_universe())
    res = await gate.evaluate(_hood_proposal(), today=date(2025, 6, 1), raise_on_fail=False)
    statuses = {r.rule: r.status for r in res.results}
    assert statuses["tier2_screen"] == "fail"


@pytest.mark.asyncio
async def test_tier2_screen_passes_with_high_score(db_repos, monkeypatch):
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)
    await _seed_candidate(db_repos, symbol="HOOD", score=72.0)
    cfg = _config()
    cfg["intelligence"] = {"llm_screener_enabled": True, "tier2_min_score": 50}
    gate = RiskGate(PaperBroker(cash=20_000), db_repos, cfg, _tier2_universe())
    res = await gate.evaluate(_hood_proposal(), today=date(2025, 6, 1), raise_on_fail=False)
    statuses = {r.rule: r.status for r in res.results}
    assert statuses["tier2_screen"] == "pass"


@pytest.mark.asyncio
async def test_tier2_screen_fails_when_score_below_threshold(db_repos, monkeypatch):
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)
    await _seed_candidate(db_repos, symbol="HOOD", score=42.0)
    cfg = _config()
    cfg["intelligence"] = {"llm_screener_enabled": True, "tier2_min_score": 50}
    gate = RiskGate(PaperBroker(cash=20_000), db_repos, cfg, _tier2_universe())
    res = await gate.evaluate(_hood_proposal(), today=date(2025, 6, 1), raise_on_fail=False)
    statuses = {r.rule: r.status for r in res.results}
    assert statuses["tier2_screen"] == "fail"


@pytest.mark.asyncio
async def test_tier2_screen_skips_bear_call_direction(db_repos, monkeypatch):
    """Bear-call direction must bypass — current screener prompt is bull-biased."""
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)
    # No candidate row, low score, etc. — none of it matters; bear-call bypasses.
    cfg = _config()
    cfg["intelligence"] = {"llm_screener_enabled": True, "tier2_min_score": 50}
    bear_call_universe = {
        "tickers": [UniverseEntry(symbol="WBA", name="Walgreens", tier=2, overrides={})],
        "banned": [], "banned_rules": [],
    }
    gate = RiskGate(PaperBroker(cash=20_000), db_repos, cfg, bear_call_universe)
    bc_proposal = _multi_leg_proposal(direction="bear_call")
    # Override symbol to WBA for this test.
    from strategies.spreads import MultiLegProposal
    bc_proposal = MultiLegProposal(
        symbol="WBA",
        legs=bc_proposal.legs,
        net_credit_per_spread=bc_proposal.net_credit_per_spread,
        max_loss_per_spread=bc_proposal.max_loss_per_spread,
        width_dollars=bc_proposal.width_dollars,
        quantity=bc_proposal.quantity,
        rationale=bc_proposal.rationale,
        strategy_id="bear_call_spread",
        direction="bear_call",
    )
    res = await gate.evaluate(bc_proposal, today=date(2025, 6, 1), raise_on_fail=False)
    statuses = {r.rule: r.status for r in res.results}
    assert statuses["tier2_screen"] == "skip"


@pytest.mark.asyncio
async def test_tier2_screen_skips_closes(db_repos, monkeypatch):
    """BUY_TO_CLOSE on a tier-2 position must bypass — this is an entry gate."""
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)
    cfg = _config()
    cfg["intelligence"] = {"llm_screener_enabled": True, "tier2_min_score": 50}
    gate = RiskGate(PaperBroker(cash=20_000), db_repos, cfg, _tier2_universe())
    close_proposal = Proposal(
        symbol="HOOD",
        contract=_hood_proposal().contract,
        order_type=OrderType.BUY_TO_CLOSE,
        quantity=1,
        rationale="close",
        strategy_id="weekly_wheel",
    )
    res = await gate.evaluate(close_proposal, today=date(2025, 6, 1), raise_on_fail=False)
    statuses = {r.rule: r.status for r in res.results}
    assert statuses["tier2_screen"] == "skip"


def _itm_call(strike: float = 700.0, bid: float = 45.0, ask: float = 45.2) -> OptionContract:
    return OptionContract(
        underlying="SPY", occ_symbol="SPY260821C00700000", strike=strike,
        expiration=date(2026, 8, 21), option_type=OptionType.CALL,
        bid=bid, ask=ask, delta=0.9, underlying_price=733.0,
    )


def _swing_proposal() -> Proposal:
    return Proposal(
        symbol="SPY", contract=_itm_call(), order_type=OrderType.BUY_TO_OPEN,
        quantity=1, rationale="swing test", strategy_id="spy_swing_opt",
    )


def test_notional_long_option_is_premium_not_underlying():
    # A deep-ITM SPY long: premium ~$45.10/sh → $4,510, NOT ~$73,300 underlying.
    from risk.limits import _notional
    assert _notional(_swing_proposal()) == pytest.approx(45.1 * 100)


@pytest.mark.asyncio
async def test_regime_gate_skips_swing(db_repos):
    # Swing carries its own 200-SMA gate → the CSP regime rule must skip it.
    from risk.limits import RiskCheckResult
    cfg = {
        "account": {"id": "test"}, "regime": {"enabled": True},
        "strategies": [{"id": "spy_swing_opt", "type": "swing"}],
    }
    gate = RiskGate(PaperBroker(cash=100_000), db_repos, cfg, _universe())
    p = _swing_proposal()
    res = RiskCheckResult(proposal=p)
    await gate._rule_regime(res, p, {})
    assert res.results[-1].status == "skip"
    assert "swing" in res.results[-1].detail


# -- per-strategy concurrent cap (2026-07-08 fix) ------------------------------


@pytest.mark.asyncio
async def test_concurrent_cap_uses_strategy_block_max_concurrent(db_repos, monkeypatch):
    """The strategies-block max_concurrent (2) beats the wheel-section
    max_concurrent_positions (4) — pre-fix, EVERY strategy got the wheel
    default and put_spread piled up 3-4 pendings past its cap of 2."""
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)
    broker = PaperBroker(cash=20_000)
    now = datetime.now(UTC).replace(tzinfo=None)
    for sym in ("BAC", "SOFI"):
        await db_repos.positions.insert(
            Position(
                account_id="test",
                symbol=sym,
                strategy_id="monthly_wheel",
                state=PositionState.CSP_OPEN,
                shares=0,
                state_changed_at=now,
            )
        )
    cfg = _config(max_concurrent_positions=4)  # wheel fallback alone would PASS
    cfg["strategies"] = [{"id": "monthly_wheel", "max_concurrent": 2}]
    gate = RiskGate(broker, db_repos, cfg, _universe())
    res = await gate.evaluate(_proposal(), today=date(2025, 6, 1), raise_on_fail=False)
    statuses = {r.rule: r.status for r in res.results}
    assert statuses["concurrent_positions_cap"] == "fail"  # projected 3 > cap 2


@pytest.mark.asyncio
async def test_concurrent_cap_falls_back_to_wheel_param(db_repos, monkeypatch):
    """Strategy without a max_concurrent (or absent from the strategies
    block) keeps the wheel-section fallback — backwards compatible."""
    monkeypatch.setattr("risk.limits.in_blackout", lambda *a, **k: None)
    broker = PaperBroker(cash=20_000)
    now = datetime.now(UTC).replace(tzinfo=None)
    for sym in ("BAC", "SOFI"):
        await db_repos.positions.insert(
            Position(
                account_id="test",
                symbol=sym,
                strategy_id="monthly_wheel",
                state=PositionState.CSP_OPEN,
                shares=0,
                state_changed_at=now,
            )
        )
    cfg = _config(max_concurrent_positions=4)
    cfg["strategies"] = [{"id": "some_other_strategy", "max_concurrent": 1}]
    gate = RiskGate(broker, db_repos, cfg, _universe())
    res = await gate.evaluate(_proposal(), today=date(2025, 6, 1), raise_on_fail=False)
    statuses = {r.rule: r.status for r in res.results}
    assert statuses["concurrent_positions_cap"] == "pass"  # projected 3 <= 4
